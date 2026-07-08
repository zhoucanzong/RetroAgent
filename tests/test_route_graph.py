import unittest

from retroagent.blackboard import SharedBlackboard
from retroagent.route_graph import RouteGraph
from retroagent.schemas import build_reaction_record


class RouteGraphTests(unittest.TestCase):
    def test_route_graph_inserts_reaction_as_and_or_nodes(self):
        graph = RouteGraph()
        target_id = graph.add_target("CC=O")
        reaction = build_reaction_record(
            reaction_id="rxn_oxidation",
            product_smiles="CC=O",
            precursor_smiles_list=["CCO", "O=O"],
            classification="Oxidation",
        )

        reaction_id = graph.add_reaction(reaction)

        self.assertEqual(reaction_id, "rxn_oxidation")
        self.assertEqual(graph.target_id, target_id)
        self.assertIn(reaction_id, graph.product_to_reactions[target_id])
        self.assertEqual(len(graph.reaction_to_precursors[reaction_id]), 2)
        self.assertEqual(graph.stats()["reaction_count"], 1)
        self.assertEqual(graph.stats()["molecule_count"], 3)

    def test_route_graph_reuses_canonical_molecule_nodes(self):
        graph = RouteGraph()
        first = graph.add_molecule("CCO")
        second = graph.add_molecule("OCC")

        self.assertEqual(first, second)
        self.assertEqual(graph.stats()["molecule_count"], 1)

    def test_stock_marking_removes_molecule_from_unsolved_set(self):
        graph = RouteGraph()
        graph.add_target("CC=O")
        graph.add_reaction(build_reaction_record(
            product_smiles="CC=O",
            precursor_smiles_list=["CCO"],
        ))

        self.assertEqual(graph.stats()["unsolved_molecule_count"], 1)
        graph.mark_stock("CCO", source="test_stock")
        self.assertEqual(graph.stats()["unsolved_molecule_count"], 0)

    def test_blackboard_updates_route_graph_from_proposals_and_stock(self):
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="retrosynthesis")
        blackboard.update("propose", {
            "reactions": [
                build_reaction_record(
                    reaction_id="rxn_1",
                    product_smiles="CC=O",
                    precursor_smiles_list=["CCO"],
                )
            ]
        })

        self.assertEqual(blackboard.route_graph.stats()["reaction_count"], 1)
        self.assertEqual(blackboard.route_graph.stats()["unsolved_molecule_count"], 1)

        blackboard.update("check_stock", {"in_stock": {"CCO": "test_stock"}})

        self.assertEqual(blackboard.route_graph.stats()["unsolved_molecule_count"], 0)

    def test_route_graph_enumerates_and_scores_solved_routes_first(self):
        graph = RouteGraph()
        graph.add_target("CC=O")
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_solved",
            product_smiles="CC=O",
            precursor_smiles_list=["CCO"],
            classification="Oxidation",
        ))
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_partial",
            product_smiles="CC=O",
            precursor_smiles_list=["CCCl"],
            classification="Reduction",
        ))
        graph.mark_stock("CCO", source="test_stock")

        routes = graph.scored_routes(max_routes=5)

        self.assertEqual(len(routes), 2)
        self.assertEqual(routes[0]["reaction_ids"], ["rxn_solved"])
        self.assertTrue(routes[0]["metadata"]["solved"])
        self.assertFalse(routes[1]["metadata"]["solved"])
        self.assertGreater(routes[0]["score"], routes[1]["score"])

    def test_route_graph_enumerates_recursive_two_step_route(self):
        graph = RouteGraph()
        graph.add_target("CC=O")
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_target",
            product_smiles="CC=O",
            precursor_smiles_list=["CCO"],
            classification="Oxidation",
        ))
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_precursor",
            product_smiles="CCO",
            precursor_smiles_list=["CCBr"],
            classification="Substitution",
        ))
        graph.mark_stock("CCBr", source="test_stock")

        routes = graph.scored_routes(max_depth=3)

        self.assertEqual(routes[0]["reaction_ids"], ["rxn_target", "rxn_precursor"])
        self.assertTrue(routes[0]["metadata"]["solved"])
        self.assertEqual(routes[0]["metadata"]["step_count"], 2)


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from retroagent.schemas import build_reaction_record
from retroagent.tools.route_scorer import RouteScorerTool


class RouteScorerToolTests(unittest.TestCase):
    def test_score_routes_returns_ranked_candidate_routes(self):
        tool = RouteScorerTool()
        result = json.loads(tool.execute({
            "target_smiles": "CC=O",
            "reactions": [
                build_reaction_record(
                    reaction_id="rxn_solved",
                    product_smiles="CC=O",
                    precursor_smiles_list=["CCO"],
                    classification="Oxidation",
                ),
                build_reaction_record(
                    reaction_id="rxn_partial",
                    product_smiles="CC=O",
                    precursor_smiles_list=["CCCl"],
                    classification="Reduction",
                ),
            ],
            "stock_hits": {"CCO": "test_stock"},
        }))

        self.assertEqual(result["route_count"], 2)
        self.assertEqual(result["summary"]["solved_route_count"], 1)
        self.assertEqual(result["routes"][0]["reaction_ids"], ["rxn_solved"])
        self.assertTrue(result["routes"][0]["metadata"]["solved"])
        self.assertIn("candidate_routes", result["route_graph"])

    def test_parameters_schema_requires_target_and_reactions(self):
        schema = RouteScorerTool().parameters_schema()

        self.assertEqual(schema["required"], ["target_smiles", "reactions"])
        self.assertIn("stock_hits", schema["properties"])


if __name__ == "__main__":
    unittest.main()

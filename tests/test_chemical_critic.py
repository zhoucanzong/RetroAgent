import json
import unittest

from retroagent.route_graph import RouteGraph
from retroagent.schemas import build_reaction_record
from retroagent.tools.chemical_critic import ChemicalCriticTool, build_critic_report


class ChemicalCriticToolTests(unittest.TestCase):
    def test_critic_flags_empty_portfolio(self):
        report = build_critic_report(
            target_smiles="CC=O",
            hypotheses=[],
            reactions=[],
            route_scores={},
            stock_hits={},
            literature_hits=[],
        )

        self.assertTrue(report["needs_revision"])
        self.assertEqual(report["critical_count"], 1)
        self.assertIn("hypothesis_count", report["metadata"])

    def test_critic_flags_speculative_hypothesis_and_low_score(self):
        report = build_critic_report(
            target_smiles="CC=O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Speculative oxidation route.",
                    "evidence": [
                        {
                            "level": "speculative",
                            "source": "model",
                            "summary": "Plausible but unsupported.",
                        }
                    ],
                    "risk_flags": [
                        {
                            "severity": "warning",
                            "category": "selectivity",
                            "message": "Overoxidation possible.",
                        }
                    ],
                    "next_experiments": [
                        {
                            "objective": "Screen oxidants",
                            "procedure": "Run small-scale oxidant screen.",
                        }
                    ],
                }
            ],
            reactions=[
                {
                    "id": "rxn_1",
                    "product_smiles": "CC=O",
                    "precursor_smiles_list": ["CCO"],
                    "library_occurrence": 2,
                }
            ],
            route_scores={"rxn_1": {"total": 0.22}},
            stock_hits={},
            literature_hits=[],
        )

        messages = [issue["message"] for issue in report["issues"]]

        self.assertTrue(report["needs_revision"])
        self.assertGreaterEqual(report["warning_count"], 2)
        self.assertTrue(any("speculative-only" in message for message in messages))
        self.assertTrue(any("low route score" in message for message in messages))
        self.assertIn("reaction_records_present", report["passed_checks"])

    def test_critic_flags_template_product_mismatch_and_unassigned_stereo(self):
        report = build_critic_report(
            target_smiles="CC(F)Cl",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Use a template-classified amide route.",
                    "evidence": [
                        {
                            "level": "tool-supported",
                            "source": "propose",
                            "summary": "Template proposal exists.",
                        }
                    ],
                    "risk_flags": [{"severity": "warning", "category": "stereo", "message": "needs control"}],
                    "next_experiments": [
                        {
                            "objective": "Check enantioselectivity",
                            "procedure": "Run chiral HPLC assay after a small-scale test.",
                            "decision_point": "Advance only if one enantiomer dominates.",
                        }
                    ],
                }
            ],
            reactions=[
                {
                    "id": "rxn_stereo",
                    "product_smiles": "CC(F)Cl",
                    "precursor_smiles_list": ["CC(F)O"],
                    "classification": "N-acylation to amide",
                    "library_occurrence": 50,
                }
            ],
            route_scores={"rxn_stereo": {"total": 0.8}},
            stock_hits={"CC(F)O": "stock"},
            literature_hits=[{"title": "analog"}],
        )

        messages = [issue["message"] for issue in report["issues"]]
        categories = [issue["category"] for issue in report["issues"]]

        self.assertIn("mechanism", categories)
        self.assertIn("stereochemistry", categories)
        self.assertTrue(any("does not match obvious product functional groups" in msg for msg in messages))
        self.assertTrue(any("unassigned stereocenter" in msg for msg in messages))

    def test_critic_flags_reactive_precursors_mapping_and_condition_gaps(self):
        report = build_critic_report(
            target_smiles="CC(=O)O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Acyl chloride route to acid derivative.",
                    "evidence": [
                        {
                            "level": "analog-supported",
                            "source": "literature",
                            "summary": "Related acylation precedent.",
                        }
                    ],
                    "risk_flags": [{"severity": "warning", "category": "safety", "message": "reactive reagent"}],
                    "next_experiments": [
                        {
                            "objective": "Probe controlled hydrolysis",
                            "procedure": "Run a cooled microscale quench and monitor acid formation.",
                            "decision_point": "Stop if uncontrolled exotherm or byproducts dominate.",
                        }
                    ],
                }
            ],
            reactions=[
                {
                    "id": "rxn_acyl",
                    "product_smiles": "CC(=O)O",
                    "precursor_smiles_list": ["CC(=O)Cl"],
                    "classification": "Acyl substitution",
                    "library_occurrence": 20,
                }
            ],
            route_scores={"rxn_acyl": {"total": 0.7}},
            stock_hits={"CC(=O)Cl": "stock"},
            literature_hits=[{"title": "acyl chloride hydrolysis"}],
            mapping_reports=[{"reaction_id": "rxn_acyl", "confidence": 0.31}],
            condition_reports=[{"reaction_id": "rxn_acyl", "conditions": {"catalyst": "to be determined"}}],
        )

        categories = [issue["category"] for issue in report["issues"]]
        messages = [issue["message"] for issue in report["issues"]]

        self.assertIn("safety", categories)
        self.assertIn("mechanism", categories)
        self.assertIn("conditions", categories)
        self.assertTrue(any("reactive alert" in msg for msg in messages))
        self.assertTrue(any("mapping confidence is low" in msg for msg in messages))
        self.assertTrue(any("condition report is incomplete" in msg for msg in messages))
        self.assertEqual(report["metadata"]["mapping_report_count"], 1)
        self.assertEqual(report["metadata"]["condition_report_count"], 1)

    def test_critic_audits_unresolved_low_diversity_route_graph(self):
        graph = RouteGraph()
        graph.add_target("CC=O")
        for idx in range(1, 4):
            graph.add_reaction(build_reaction_record(
                reaction_id=f"rxn_{idx}",
                product_smiles="CC=O",
                precursor_smiles_list=["CCO"],
                classification="Oxidation",
            ))

        report = build_critic_report(
            target_smiles="CC=O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Several linear oxidations could access the target.",
                    "evidence": [{"level": "tool-supported", "source": "propose", "summary": "three proposals"}],
                    "risk_flags": [{"severity": "warning", "category": "diversity", "message": "similar routes"}],
                    "next_experiments": [
                        {
                            "objective": "Compare oxidation precursors",
                            "procedure": "Run small-scale screens for three alcohol precursors.",
                            "decision_point": "Advance the cleanest conversion only.",
                        }
                    ],
                }
            ],
            reactions=list(graph.to_dict()["reactions"].values()),
            route_scores={"rxn_1": {"total": 0.7}, "rxn_2": {"total": 0.6}, "rxn_3": {"total": 0.55}},
            stock_hits={},
            literature_hits=[{"title": "oxidation analog"}],
            route_graph=graph.to_dict(),
        )

        categories = [issue["category"] for issue in report["issues"]]

        self.assertIn("route_graph", categories)
        self.assertIn("diversity", categories)
        self.assertIn("convergence", categories)
        self.assertEqual(report["metadata"]["route_graph"]["unsolved_molecule_count"], 1)
        self.assertEqual(report["metadata"]["route_graph"]["target_reaction_count"], 3)

    def test_critic_accepts_resolved_diverse_route_graph(self):
        graph = RouteGraph()
        graph.add_target("CC(=O)OCC")
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_ester",
            product_smiles="CC(=O)OCC",
            precursor_smiles_list=["CC(=O)O", "CCO"],
            classification="Esterification",
        ))
        graph.add_reaction(build_reaction_record(
            reaction_id="rxn_acyl",
            product_smiles="CC(=O)OCC",
            precursor_smiles_list=["CC(=O)Cl", "CCO"],
            classification="Acyl substitution",
        ))
        graph.mark_stock("CC(=O)O", "stock")
        graph.mark_stock("CC(=O)Cl", "stock")
        graph.mark_stock("CCO", "stock")

        report = build_critic_report(
            target_smiles="CC(=O)OCC",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Compare esterification and acyl substitution entries.",
                    "evidence": [{"level": "analog-supported", "source": "literature", "summary": "precedents exist"}],
                    "risk_flags": [{"severity": "info", "category": "conditions", "message": "screen acid activation"}],
                    "next_experiments": [
                        {
                            "objective": "Benchmark two convergent ester routes",
                            "procedure": "Run microscale esterification and acyl substitution in parallel.",
                            "decision_point": "Advance route with higher conversion and simpler workup.",
                        }
                    ],
                }
            ],
            reactions=list(graph.to_dict()["reactions"].values()),
            route_scores={"rxn_ester": {"total": 0.72}, "rxn_acyl": {"total": 0.66}},
            stock_hits={"CC(=O)O": "stock", "CC(=O)Cl": "stock", "CCO": "stock"},
            literature_hits=[{"title": "esterification analog"}],
            route_graph=graph.to_dict(),
        )

        self.assertEqual(report["metadata"]["route_graph"]["unsolved_molecule_count"], 0)
        self.assertIn("route_graph_precursors_resolved", report["passed_checks"])
        self.assertIn("route_graph_route_diversity_present", report["passed_checks"])
        self.assertIn("route_graph_convergent_step_present", report["passed_checks"])

    def test_critic_passes_reviewed_portfolio_with_signals(self):
        report = build_critic_report(
            target_smiles="CC=O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Analog-supported oxidation route.",
                    "evidence": [
                        {
                            "level": "analog-supported",
                            "source": "search_literature",
                            "summary": "Related substrate precedent.",
                        }
                    ],
                    "risk_flags": [
                        {
                            "severity": "info",
                            "category": "conditions",
                            "message": "Condition screen still needed.",
                        }
                    ],
                    "next_experiments": [
                        {
                            "objective": "Validate mild oxidation selectivity",
                            "procedure": "Run three-condition microscale screen and monitor by LCMS.",
                            "decision_point": "Advance if aldehyde area percent exceeds 70%.",
                        }
                    ],
                }
            ],
            reactions=[
                {
                    "id": "rxn_1",
                    "product_smiles": "CC=O",
                    "precursor_smiles_list": ["CCO"],
                    "library_occurrence": 12,
                }
            ],
            route_scores={"rxn_1": {"total": 0.75}},
            stock_hits={"CCO": "stock"},
            literature_hits=[{"title": "oxidation precedent"}],
        )

        self.assertFalse(report["needs_revision"])
        self.assertEqual(report["critical_count"], 0)
        self.assertEqual(report["warning_count"], 0)
        self.assertIn("literature_signal_present", report["passed_checks"])

    def test_tool_execute_returns_json_report(self):
        result = json.loads(ChemicalCriticTool().execute({"target_smiles": "CC=O"}))

        self.assertIn("issues", result)
        self.assertTrue(result["needs_revision"])


if __name__ == "__main__":
    unittest.main()

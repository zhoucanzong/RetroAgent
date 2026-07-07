import json
import unittest

from retroagent.tools.experiment_designer import ExperimentDesignerTool, build_experiment_plan


class ExperimentDesignerToolTests(unittest.TestCase):
    def test_designs_targeted_experiments_from_critic_issues(self):
        plan = build_experiment_plan(
            target_smiles="CC=O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Oxidation route.",
                    "evidence": [{"level": "tool-supported", "source": "propose", "summary": "candidate"}],
                }
            ],
            reactions=[
                {
                    "id": "rxn_1",
                    "product_smiles": "CC=O",
                    "precursor_smiles_list": ["CCO"],
                    "classification": "Oxidation",
                }
            ],
            critic_report={
                "issues": [
                    {
                        "severity": "warning",
                        "category": "mechanism",
                        "message": "mapping confidence is low",
                    },
                    {
                        "severity": "warning",
                        "category": "stereochemistry",
                        "message": "unassigned stereocenter",
                    },
                    {
                        "severity": "info",
                        "category": "conditions",
                        "message": "condition report is incomplete",
                    },
                ],
                "metadata": {"route_graph": {"provided": True, "unsolved_molecule_count": 0}},
            },
            route_graph={"stats": {"unsolved_molecule_count": 0}},
        )

        objectives = [step["objective"] for step in plan["steps"]]

        self.assertTrue(plan["ready_for_report"])
        self.assertEqual(plan["metadata"]["priority"], "warning")
        self.assertIn("mechanism", plan["linked_issue_categories"])
        self.assertIn("Validate reaction mechanism and product connectivity", objectives)
        self.assertIn("Test stereochemical control", objectives)
        self.assertIn("Screen runnable reaction conditions", objectives)
        self.assertGreaterEqual(len(plan["controls"]), 2)
        self.assertGreaterEqual(len(plan["success_criteria"]), 3)

    def test_designs_general_probe_when_critic_has_no_issues(self):
        plan = build_experiment_plan(
            target_smiles="CC=O",
            hypotheses=[{"id": "h1", "claim": "Oxidation route."}],
            reactions=[{"id": "rxn_1", "classification": "Oxidation"}],
            critic_report={"issues": []},
        )

        self.assertTrue(plan["ready_for_report"])
        self.assertEqual(plan["metadata"]["priority"], "routine")
        self.assertEqual(plan["steps"][0]["objective"], "Run first-pass route validation")
        self.assertEqual(plan["controls"][0]["objective"], "Baseline control")

    def test_tool_execute_returns_json_plan(self):
        result = json.loads(ExperimentDesignerTool().execute({
            "target_smiles": "CC=O",
            "critic_report": {
                "issues": [
                    {"severity": "info", "category": "availability", "message": "unresolved precursor"}
                ],
                "metadata": {"route_graph": {"provided": True, "unsolved_molecule_count": 2}},
            },
            "route_graph": {"stats": {"unsolved_molecule_count": 2}},
        }))

        self.assertIn("steps", result)
        self.assertTrue(result["ready_for_report"])
        self.assertIn("availability", result["linked_issue_categories"])

    def test_schema_exposes_required_target(self):
        schema = ExperimentDesignerTool().parameters_schema()

        self.assertEqual(schema["required"], ["target_smiles"])
        self.assertIn("critic_report", schema["properties"])


if __name__ == "__main__":
    unittest.main()

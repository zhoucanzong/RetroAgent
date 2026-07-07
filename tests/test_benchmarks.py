import json
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem

from retroagent.benchmark import (
    analyze_trajectory,
    analyze_trajectory_file,
    load_discovery_cases,
    run_discovery_cases,
    score_discovery_submission,
    score_discovery_rubric,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent


class BenchmarkTests(unittest.TestCase):
    def test_discovery_small_jsonl_is_valid_and_nontrivial(self):
        path = PROJECT_DIR / "benchmarks" / "discovery_small.jsonl"
        lines = [line for line in path.read_text().splitlines() if line.strip()]

        self.assertGreaterEqual(len(lines), 5)
        for line in lines:
            case = json.loads(line)
            with self.subTest(case=case["id"]):
                self.assertEqual(case["task_type"], "discovery")
                self.assertNotEqual(case["target_name"].lower(), "aspirin")
                self.assertGreaterEqual(case["desired_route_count"], 3)
                self.assertIn("objective", case)
                self.assertIn("success_criteria", case)
                self.assertIn("required_evidence_levels", case)
                self.assertTrue(case["success_criteria"])
                self.assertTrue(case["required_evidence_levels"])
                self.assertIsNotNone(Chem.MolFromSmiles(case["target_smiles"]))

    def test_discovery_case_loader_and_dry_run_runner(self):
        path = PROJECT_DIR / "benchmarks" / "discovery_small.jsonl"
        cases = load_discovery_cases(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_discovery_cases(
                cases[:2],
                output_dir=Path(tmpdir),
                runner=lambda case, trajectory_path: {"exit_status": "unused"},
                dry_run=True,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].exit_status, "DryRun")
        self.assertIn("task_prompt_chars", results[0].metrics)

    def test_discovery_submission_shape_scoring(self):
        case = load_discovery_cases(PROJECT_DIR / "benchmarks" / "discovery_small.jsonl")[0]
        submission = """
Route A: known literature-backed route.
Evidence: known. Risk: chirality and racemization must be controlled.
Next experiment: validate enantiomeric ratio under pilot conditions.

Route B: analog-supported route.
Evidence: analog-supported. Risk: chemoselectivity.
Next validation: run small-scale control experiment.

Route C: speculative route.
Evidence: tool-supported and speculative.
Risk: unstable intermediate possible.
Next experiment: test precursor stability.
"""

        metrics = score_discovery_submission(submission, case)

        self.assertTrue(metrics["has_route_portfolio"])
        self.assertGreaterEqual(metrics["evidence_label_coverage"], 0.5)
        self.assertTrue(metrics["mentions_risk"])
        self.assertTrue(metrics["mentions_next_experiment"])
        self.assertTrue(metrics["passes_shape_check"])

    def test_trajectory_analysis_extracts_tool_and_route_metrics(self):
        messages = [
            {
                "role": "assistant",
                "content": "call tools",
                "extra": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            {
                "role": "user",
                "content": (
                    "<tool>research_memory</tool>\n<output>\n"
                    "{\"status\":\"ok\",\"count\":0,\"records\":[]}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<tool>propose</tool>\n"
                    "<returncode>0</returncode>\n"
                    "<output>\n"
                    "{\"reactions\":["
                    "{\"id\":\"rxn_1\",\"template_index\":1,\"classification\":\"Oxidation\","
                    "\"precursor_smiles_list\":[\"CCO\"]},"
                    "{\"id\":\"rxn_2\",\"template_index\":2,\"classification\":\"Reduction\","
                    "\"precursor_smiles_list\":[\"CCCl\"]}"
                    "]}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<tool>evaluate</tool>\n<output>\n"
                    "{\"scores\":{\"rxn_1\":{\"total\":0.8},\"rxn_2\":{\"total\":0.5}}}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": "<tool>check_stock</tool>\n<output>\n{\"in_stock\":{\"CCO\":\"test\"}}\n</output>",
            },
            {
                "role": "user",
                "content": "<tool>web_search</tool>\n<output>\n{\"results\":[{\"title\":\"paper\"}]}\n</output>",
            },
            {
                "role": "user",
                "content": "<tool>functional_groups</tool>\n<output>\n{\"functional_groups_present\":{}}\n</output>",
            },
            {
                "role": "user",
                "content": (
                    "<tool>chemical_critic</tool>\n<output>\n"
                    "{\"warning_count\":2,\"critical_count\":0,\"issues\":[],"
                    "\"metadata\":{\"route_graph\":{\"provided\":true,"
                    "\"unsolved_molecule_count\":3}}}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<tool>design_experiments</tool>\n<output>\n"
                    "{\"ready_for_report\":true,"
                    "\"steps\":[{\"objective\":\"Validate mechanism\"},{\"objective\":\"Resolve precursor\"}],"
                    "\"controls\":[{\"objective\":\"No reagent control\"}]}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<tool>build_research_report</tool>\n<output>\n"
                    "{\"ready_for_review\":true,\"hypothesis_count\":3}\n"
                    "</output>"
                ),
            },
            {
                "role": "user",
                "content": (
                    "<tool>research_memory</tool>\n<output>\n"
                    "{\"status\":\"recorded\",\"record\":{\"id\":\"mem_1\"}}\n"
                    "</output>"
                ),
            },
            {"role": "user", "content": "<role:reviewer>\nISSUE: [WARNING] risk\n</role:reviewer>"},
        ]

        metrics = analyze_trajectory(messages)

        self.assertEqual(metrics["tool_call_count"], 10)
        self.assertEqual(metrics["proposal_reaction_count"], 2)
        self.assertEqual(metrics["reaction_class_count"], 2)
        self.assertEqual(metrics["evaluated_route_count"], 2)
        self.assertEqual(metrics["stock_hit_count"], 1)
        self.assertEqual(metrics["literature_hit_count"], 1)
        self.assertEqual(metrics["critic_report_count"], 1)
        self.assertEqual(metrics["critic_warning_count"], 2)
        self.assertEqual(metrics["critic_critical_count"], 0)
        self.assertEqual(metrics["critic_route_graph_count"], 1)
        self.assertEqual(metrics["critic_unresolved_molecule_count"], 3)
        self.assertEqual(metrics["experiment_plan_count"], 1)
        self.assertEqual(metrics["experiment_step_count"], 2)
        self.assertEqual(metrics["experiment_control_count"], 1)
        self.assertEqual(metrics["memory_search_count"], 1)
        self.assertEqual(metrics["memory_record_count"], 1)
        self.assertEqual(metrics["structured_report_count"], 1)
        self.assertEqual(metrics["complete_structured_report_count"], 1)
        self.assertTrue(metrics["used_evidence_tools"])
        self.assertTrue(metrics["used_risk_tools"])
        self.assertTrue(metrics["used_validation_tools"])
        self.assertEqual(metrics["reviewer_messages"], 1)
        self.assertEqual(metrics["total_tokens"], 15)

    def test_discovery_rubric_passes_complete_result(self):
        case = load_discovery_cases(PROJECT_DIR / "benchmarks" / "discovery_small.jsonl")[0]
        metrics = {
            "has_route_portfolio": True,
            "evidence_label_coverage": 0.75,
            "mentions_risk": True,
            "mentions_next_experiment": True,
            "passes_shape_check": True,
            "trajectory": {
                "proposal_reaction_count": 3,
                "reaction_class_count": 2,
                "precursor_set_count": 3,
                "template_count": 3,
                "used_evidence_tools": True,
                "literature_hit_count": 1,
                "used_validation_tools": True,
                "evaluated_route_count": 3,
                "stock_hit_count": 1,
                "used_risk_tools": True,
                "reviewer_messages": 1,
                "warning_messages": 0,
            },
        }

        rubric = score_discovery_rubric(metrics, case)

        self.assertTrue(rubric["overall_pass"])
        self.assertEqual(rubric["passed_checks"], rubric["required_checks"])

    def test_discovery_rubric_fails_without_trace_grounding(self):
        case = load_discovery_cases(PROJECT_DIR / "benchmarks" / "discovery_small.jsonl")[0]
        metrics = {
            "has_route_portfolio": True,
            "evidence_label_coverage": 0.75,
            "mentions_risk": True,
            "mentions_next_experiment": True,
            "passes_shape_check": True,
            "trajectory": {
                "proposal_reaction_count": 0,
                "reaction_class_count": 0,
                "precursor_set_count": 0,
                "template_count": 0,
                "used_evidence_tools": False,
                "used_validation_tools": False,
                "used_risk_tools": False,
            },
        }

        rubric = score_discovery_rubric(metrics, case)

        self.assertFalse(rubric["overall_pass"])
        self.assertFalse(rubric["route_diversity_pass"])
        self.assertFalse(rubric["evidence_grounding_pass"])
        self.assertFalse(rubric["validation_tool_pass"])
        self.assertFalse(rubric["risk_review_pass"])

    def test_runner_includes_trajectory_metrics_when_file_exists(self):
        case = load_discovery_cases(PROJECT_DIR / "benchmarks" / "discovery_small.jsonl")[0]

        def runner(_case, trajectory_path):
            trajectory_path.write_text(json.dumps([
                {
                    "role": "user",
                    "content": "<tool>propose</tool>\n<output>\n{\"reactions\":[]}\n</output>",
                }
            ]))
            return {
                "exit_status": "Submitted",
                "submission": (
                    "Route A: known. Risk: stereo. Next experiment: test.\n"
                    "Route B: analog-supported. Risk: safety. Next experiment: validate.\n"
                    "Route C: speculative tool-supported. Risk: selectivity. Next validation: control."
                ),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_discovery_cases(
                [case],
                output_dir=Path(tmpdir),
                runner=runner,
                dry_run=False,
            )
            trajectory_metrics = analyze_trajectory_file(Path(results[0].trajectory_path))

        self.assertIn("trajectory", results[0].metrics)
        self.assertIn("rubric", results[0].metrics)
        self.assertEqual(results[0].metrics["trajectory"]["tool_call_count"], 1)
        self.assertEqual(trajectory_metrics["tool_call_count"], 1)


if __name__ == "__main__":
    unittest.main()

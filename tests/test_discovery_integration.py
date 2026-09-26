import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from typer.testing import CliRunner

from retroagent.agents.discovery_controller import DiscoveryController
from retroagent.agents.planner import RetroPlanner
from retroagent.benchmark import (
    analyze_research_events,
    load_blind_cases,
    load_feedback_cases,
    score_closed_loop_events,
)
from retroagent.config import load_config
from retroagent.research_execution import PermissionGrant, PermissionMode
from retroagent.run.retro import _build_agent_for_mode, _build_blind_environment, app
from retroagent.tools.feedback_oracle import FeedbackOracleTool


PROJECT_DIR = Path(__file__).resolve().parent.parent


class _Model:
    pass


class _Env:
    def get_tools_spec(self):
        return []

    def get_tool_graph_summary(self):
        return "| Tool | Category |\n|---|---|"


class DiscoveryIntegrationTests(unittest.TestCase):
    def test_agent_factory_routes_discovery_only_to_v2_controller(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            discovery = _build_agent_for_mode(
                "discovery", _Model(), _Env(), cfg,
                max_steps=7, output=None, run_workspace=Path(tmp),
                ledger_output=None, permission_mode="standard", permission_scopes="",
            )
            retrosynthesis = _build_agent_for_mode(
                "retrosynthesis", _Model(), _Env(), cfg,
                max_steps=7, output=None, run_workspace=Path(tmp),
                ledger_output=None, permission_mode="standard", permission_scopes="",
            )
            design = _build_agent_for_mode(
                "design", _Model(), _Env(), cfg,
                max_steps=7, output=None, run_workspace=Path(tmp),
                ledger_output=None, permission_mode="standard", permission_scopes="",
            )

        self.assertIsInstance(discovery, DiscoveryController)
        self.assertEqual(discovery.max_cycles, 7)
        self.assertIsInstance(retrosynthesis, RetroPlanner)
        self.assertIsInstance(design, RetroPlanner)

    def test_full_permission_mode_requires_explicit_scopes(self):
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "explicit permission_scopes"):
                _build_agent_for_mode(
                    "discovery", _Model(), _Env(), cfg,
                    max_steps=7, output=None, run_workspace=Path(tmp),
                    ledger_output=None, permission_mode="full", permission_scopes="",
                )

    def test_feedback_benchmark_case_is_valid(self):
        cases = load_feedback_cases(PROJECT_DIR / "benchmarks" / "discovery_feedback.jsonl")

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].initial_routes[0]["id"], "route_a")
        self.assertEqual(cases[0].feedback["reaction_ids"], ["rxn_a"])

        result = json.loads(FeedbackOracleTool(cases[0].feedback).execute({"route_id": "route_a"}))
        self.assertEqual(result["failure"]["category"], "selectivity")
        wrong_route = json.loads(FeedbackOracleTool(cases[0].feedback).execute({"route_id": "route_b"}))
        self.assertNotIn("failure", wrong_route)

    def test_feedback_case_rejects_nonpreferred_route_a_and_unlinked_feedback(self):
        base = json.loads((PROJECT_DIR / "benchmarks" / "discovery_feedback.jsonl").read_text())
        base["initial_routes"][0]["scores"]["feasibility"] = 0.1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.jsonl"
            path.write_text(json.dumps(base) + "\n")
            with self.assertRaisesRegex(ValueError, "route_a must be the highest"):
                load_feedback_cases(path)

        base = json.loads((PROJECT_DIR / "benchmarks" / "discovery_feedback.jsonl").read_text())
        base["feedback"]["reaction_ids"] = ["rxn_not_in_route_a"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.jsonl"
            path.write_text(json.dumps(base) + "\n")
            with self.assertRaisesRegex(ValueError, "must belong to route_a"):
                load_feedback_cases(path)

    def test_feedback_benchmark_dry_run_writes_validated_results(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "feedback"
            result = runner.invoke(app, [
                "benchmark-discovery-feedback",
                "--cases", str(PROJECT_DIR / "benchmarks" / "discovery_feedback.jsonl"),
                "--output-dir", str(output_dir),
                "--dry-run",
            ])
            rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(rows[0]["exit_status"], "DryRun")
        self.assertEqual(rows[0]["metrics"]["initial_route_count"], 2)

    def test_blind_study_benchmark_dry_run_validates_withheld_case(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "blind"
            result = runner.invoke(app, [
                "benchmark-discovery-blind",
                "--cases", str(PROJECT_DIR / "benchmarks" / "blind_studies.jsonl"),
                "--output-dir", str(output_dir),
                "--dry-run",
            ])
            rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["exit_status"], "DryRun")
        self.assertEqual(rows[0]["published_date"], "2026-07-15")
        self.assertEqual(rows[0]["metrics"]["registered_experiment_count"], 4)
        self.assertNotIn("expected_outcome", rows[0])

    def test_blind_study_benchmark_can_dry_run_one_case_by_id(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "blind"
            result = runner.invoke(app, [
                "benchmark-discovery-blind",
                "--cases", str(PROJECT_DIR / "benchmarks" / "blind_studies.jsonl"),
                "--case-id", "boronate-amination-2026-002",
                "--output-dir", str(output_dir),
                "--dry-run",
            ])
            rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([row["case_id"] for row in rows], ["boronate-amination-2026-002"])

    def test_blind_study_benchmark_rejects_unknown_case_id(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.invoke(app, [
                "benchmark-discovery-blind",
                "--cases", str(PROJECT_DIR / "benchmarks" / "blind_studies.jsonl"),
                "--case-id", "not-a-case",
                "--output-dir", str(Path(tmp) / "blind"),
                "--dry-run",
            ])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown blind case id", result.output)

    def test_blind_study_environment_excludes_answer_leaking_tools(self):
        case = load_blind_cases(PROJECT_DIR / "benchmarks" / "blind_studies.jsonl")[0]

        env = _build_blind_environment(case)
        names = {tool["name"] for tool in env.get_tools_spec()}

        self.assertEqual(names, {
            "blind_study_oracle", "think", "functional_groups", "molecule_properties",
        })
        self.assertTrue({
            "research_memory", "web_search", "fetch_url", "search_literature",
            "convert_identifier", "check_patent",
        }.isdisjoint(names))
        self.assertNotEqual(
            env.execute({"tool": "bash", "parameters": {"command": "printf must-not-run"}})["returncode"],
            0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            controller = DiscoveryController(
                _Model(),
                env,
                workspace=tmp,
                permission_grant=PermissionGrant(mode=PermissionMode.RESTRICTED),
                allow_execution_tools=False,
            )
        self.assertEqual({tool["name"] for tool in controller._tools_spec()}, names)

    def test_discovery_v3_trajectory_cli_scores_event_contract(self):
        events = [
            {"type": "hypothesis_proposed", "payload": {"id": "H1"}},
            {"type": "hypothesis_proposed", "payload": {"id": "H2"}},
            {"type": "evidence_added", "payload": {"id": "evidence_1"}},
            {"type": "experiment_selected", "payload": {
                "id": "exp_1", "rank": 1, "expected_information_gain": 0.5,
            }},
            {"type": "hypothesis_posterior_updated", "payload": {
                "id": "H1", "posterior_probability": 0.2, "entropy_reduction": 0.3,
            }},
            {"type": "hypothesis_posterior_updated", "payload": {
                "id": "H2", "posterior_probability": 0.8, "entropy_reduction": 0.3,
            }},
            {"type": "route_portfolio_built", "payload": {"multi_source_route_count": 1}},
            {"type": "route_validated", "payload": {"id": "route_1", "status": "passed"}},
            {"type": "research_completed", "payload": {
                "target_ids": ["route_1"], "evidence_ids": ["evidence_1"],
            }},
        ]
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            trajectory = Path(tmp) / "trajectory.json"
            trajectory.write_text(json.dumps(events), encoding="utf-8")
            result = runner.invoke(app, [
                "score-discovery-v3", str(trajectory), "--require-pass",
            ])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertTrue(json.loads(result.output)["discovery_v3_pass"])

    def test_event_metrics_prove_failure_driven_replanning(self):
        events = [
            {"id": "e1", "type": "decision_made", "payload": {
                "action_type": "tool",
                "action": {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}},
                "target_ids": ["route_a"],
            }},
            {"id": "e2", "type": "evidence_added", "payload": {
                "id": "evidence_0001", "direction": "opposes", "target_ids": ["rxn_a", "route_a"],
            }},
            {"id": "e3", "type": "reaction_status_changed", "payload": {
                "id": "rxn_a", "status": "failed",
            }},
            {"id": "e4", "type": "route_status_changed", "payload": {
                "id": "route_a", "status": "failed",
            }},
            {"id": "e5", "type": "action_failed", "payload": {
                "fingerprint": "failed", "action": {"tool": "feedback_oracle"},
                "evidence_id": "evidence_0001",
            }},
            {"id": "e6", "type": "replan_required", "payload": {
                "affected_route_ids": ["route_a"],
            }},
            {"id": "e7", "type": "decision_made", "payload": {
                "action_type": "tool",
                "action": {"tool": "inspect_route", "parameters": {"route_id": "route_b"}},
                "target_ids": ["route_b"],
            }},
            {"id": "e8", "type": "research_completed", "payload": {
                "conclusion": "Route B is preferred", "target_ids": ["route_b"],
                "evidence_ids": ["evidence_0001"],
            }},
        ]

        metrics = score_closed_loop_events(events)
        trajectory = analyze_research_events(events)

        self.assertTrue(metrics["failure_recorded"])
        self.assertTrue(metrics["failed_route_abandoned"])
        self.assertTrue(metrics["distinct_replan"])
        self.assertTrue(metrics["evidence_cited"])
        self.assertTrue(metrics["closed_loop_pass"])
        self.assertEqual(trajectory["decision_count"], 2)

    def test_closed_loop_metrics_reject_unrelated_failure_evidence(self):
        events = [
            {"type": "evidence_added", "payload": {
                "id": "ev_unrelated", "direction": "opposes", "target_ids": ["rxn_other"],
            }},
            {"type": "reaction_status_changed", "payload": {"id": "rxn_a", "status": "failed"}},
            {"type": "route_status_changed", "payload": {"id": "route_a", "status": "failed"}},
            {"type": "replan_required", "payload": {"affected_route_ids": ["route_a"]}},
            {"type": "decision_made", "payload": {"target_ids": ["route_b"], "action": {}}},
            {"type": "research_completed", "payload": {
                "target_ids": ["route_b"], "evidence_ids": ["ev_unrelated"],
            }},
        ]

        metrics = score_closed_loop_events(events)

        self.assertFalse(metrics["failure_recorded"])
        self.assertFalse(metrics["closed_loop_pass"])

    def test_closed_loop_metrics_reject_final_selection_of_failed_route(self):
        events = [
            {"type": "evidence_added", "payload": {
                "id": "ev_fail", "direction": "opposes", "target_ids": ["rxn_a", "route_a"],
            }},
            {"type": "reaction_status_changed", "payload": {"id": "rxn_a", "status": "failed"}},
            {"type": "route_status_changed", "payload": {"id": "route_a", "status": "failed"}},
            {"type": "replan_required", "payload": {"affected_route_ids": ["route_a"]}},
            {"type": "decision_made", "payload": {"target_ids": ["route_b"], "action": {}}},
            {"type": "research_completed", "payload": {
                "target_ids": ["route_a"], "evidence_ids": ["ev_fail"],
            }},
        ]

        metrics = score_closed_loop_events(events)

        self.assertFalse(metrics["failed_route_abandoned"])
        self.assertFalse(metrics["closed_loop_pass"])

    def test_closed_loop_metrics_reject_repeated_failed_action_and_unrelated_citation(self):
        events = [
            {"type": "evidence_added", "payload": {
                "id": "ev_fail", "direction": "opposes", "target_ids": ["rxn_a", "route_a"],
            }},
            {"type": "reaction_status_changed", "payload": {"id": "rxn_a", "status": "failed"}},
            {"type": "route_status_changed", "payload": {"id": "route_a", "status": "failed"}},
            {"type": "action_failed", "payload": {"fingerprint": "fp_a", "evidence_id": "ev_fail"}},
            {"type": "replan_required", "payload": {"affected_route_ids": ["route_a"]}},
            {"type": "execution_started", "payload": {
                "fingerprint": "fp_a", "action": {"tool": "feedback_oracle"},
            }},
            {"type": "action_failed", "payload": {
                "fingerprint": "fp_a", "evidence_id": "ev_fail",
            }},
            {"type": "evidence_added", "payload": {
                "id": "ev_b", "direction": "observes", "target_ids": ["route_b"],
            }},
            {"type": "decision_made", "payload": {"target_ids": ["route_b"], "action": {}}},
            {"type": "research_completed", "payload": {
                "target_ids": ["route_b"], "evidence_ids": ["ev_b"],
            }},
        ]

        metrics = score_closed_loop_events(events)

        self.assertFalse(metrics["failed_action_not_repeated"])
        self.assertFalse(metrics["failure_evidence_cited"])
        self.assertFalse(metrics["closed_loop_pass"])

    def test_event_analyzer_preserves_legacy_tool_metrics(self):
        events = [
            {"type": "execution_started", "payload": {
                "action": {"tool": "propose", "parameters": {}}, "fingerprint": "p",
            }},
            {"type": "execution_finished", "payload": {
                "action": {"tool": "propose", "parameters": {}},
                "result": {"returncode": 0, "output": json.dumps({"reactions": [
                    {"id": "rxn_1", "classification": "Oxidation", "template_index": 1,
                     "precursor_smiles_list": ["CCO"]},
                    {"id": "rxn_2", "classification": "Reduction", "template_index": 2,
                     "precursor_smiles_list": ["CCCl"]},
                ]})},
            }},
            {"type": "execution_started", "payload": {
                "action": {"tool": "score_routes", "parameters": {}}, "fingerprint": "s",
            }},
            {"type": "execution_finished", "payload": {
                "action": {"tool": "score_routes", "parameters": {}},
                "result": {"returncode": 0, "output": json.dumps({
                    "routes": [{"id": "route_1", "score": 0.8}],
                    "summary": {"solved_route_count": 0, "best_score": 0.8},
                })},
            }},
        ]

        metrics = analyze_research_events(events)

        self.assertEqual(metrics["proposal_reaction_count"], 2)
        self.assertEqual(metrics["reaction_class_count"], 2)
        self.assertEqual(metrics["scored_route_count"], 1)
        self.assertTrue(metrics["used_validation_tools"])

    def test_event_analyzer_measures_retrosynthesis_ensemble_usage(self):
        events = [
            {"type": "execution_started", "payload": {
                "action": {"tool": "retrochimera_predict", "parameters": {}},
            }},
            {"type": "execution_finished", "payload": {
                "action": {"tool": "retrochimera_predict", "parameters": {}},
                "result": {"returncode": 0, "output": json.dumps({
                    "reactions": [{"id": "chimera_1", "precursor_smiles_list": ["CCO"]}],
                })},
            }},
            {"type": "execution_started", "payload": {
                "action": {"tool": "ensemble_disconnections", "parameters": {}},
            }},
            {"type": "execution_finished", "payload": {
                "action": {"tool": "ensemble_disconnections", "parameters": {}},
                "result": {"returncode": 0, "output": json.dumps({
                    "reactions": [
                        {"id": "ensemble_1", "precursor_smiles_list": ["CCO"], "unanimous": True},
                        {"id": "ensemble_2", "precursor_smiles_list": ["CCCl"], "unanimous": False},
                    ],
                })},
            }},
        ]

        metrics = analyze_research_events(events)

        self.assertEqual(metrics["retrochimera_call_count"], 1)
        self.assertEqual(metrics["retrochimera_candidate_count"], 1)
        self.assertEqual(metrics["ensemble_call_count"], 1)
        self.assertEqual(metrics["ensemble_candidate_count"], 2)
        self.assertEqual(metrics["consensus_candidate_count"], 1)
        self.assertEqual(metrics["proposal_reaction_count"], 0)
        self.assertFalse(metrics["used_validation_tools"])

    def test_component_predictions_count_as_proposals_not_validation(self):
        events = []
        for name in ("neuralloc_predict", "transformer_predict"):
            events.extend([
                {"type": "execution_started", "payload": {"action": {"tool": name}}},
                {"type": "execution_finished", "payload": {
                    "action": {"tool": name}, "result": {"output": "{}", "returncode": 0},
                }},
            ])
        metrics = analyze_research_events(events)
        self.assertEqual(metrics["component_call_counts"], {"neuralloc_predict": 1, "transformer_predict": 1})
        self.assertFalse(metrics["used_validation_tools"])

    def test_run_cli_passes_discovery_permission_and_workspace_options(self):
        fake_agent = Mock()
        fake_agent.run.return_value = {
            "exit_status": "Submitted",
            "conclusion": "route selected",
            "trajectory": [{"type": "research_completed", "payload": {}}],
        }
        cfg = load_config()
        cfg.llm_api_key = "test-key"
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "retroagent.run.retro.get_config", return_value=cfg,
        ), patch(
            "retroagent.run.retro.logging.basicConfig",
        ), patch(
            "retroagent.run.retro._build_environment", return_value=_Env(),
        ), patch(
            "retroagent.run.retro.LLMClient", return_value=_Model(),
        ), patch(
            "retroagent.run.retro._build_agent_for_mode", return_value=fake_agent,
        ) as factory:
            output = Path(tmp) / "trajectory.json"
            result = runner.invoke(app, [
                "run", "CC=O", "--mode", "discovery",
                "--permission-mode", "full",
                "--permission-scopes", "network,destructive",
                "--run-workspace", tmp,
                "--output", str(output),
            ])

            written_trajectory = json.loads(output.read_text())

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(factory.call_args.kwargs["permission_mode"], "full")
        self.assertEqual(factory.call_args.kwargs["permission_scopes"], "network,destructive")
        self.assertEqual(factory.call_args.kwargs["run_workspace"], Path(tmp))
        self.assertEqual(written_trajectory[0]["type"], "research_completed")

    def test_run_cli_reports_missing_full_scopes_without_traceback(self):
        cfg = load_config()
        cfg.llm_api_key = "test-key"
        runner = CliRunner()

        with patch(
            "retroagent.run.retro.get_config", return_value=cfg,
        ), patch(
            "retroagent.run.retro.logging.basicConfig",
        ), patch(
            "retroagent.run.retro._build_environment", return_value=_Env(),
        ), patch(
            "retroagent.run.retro.LLMClient", return_value=_Model(),
        ):
            result = runner.invoke(app, [
                "run", "CC=O", "--mode", "discovery", "--permission-mode", "full",
            ])

        self.assertEqual(result.exit_code, 2)
        self.assertIn("explicit permission_scopes", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_existing_discovery_benchmark_run_uses_v2_event_trajectory(self):
        fake_agent = Mock()
        fake_agent.run.return_value = {
            "exit_status": "Submitted",
            "conclusion": "Route B is preferred after feedback.",
            "trajectory": [
                {"id": "e1", "type": "decision_made", "payload": {"target_ids": ["route_b"]}},
                {"id": "e2", "type": "evidence_added", "payload": {
                    "id": "evidence_1", "direction": "observes",
                }},
                {"id": "e3", "type": "research_completed", "payload": {
                    "evidence_ids": ["evidence_1"], "target_ids": ["route_b"],
                }},
            ],
        }
        cfg = load_config()
        cfg.llm_api_key = "test-key"
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "retroagent.run.retro.get_config", return_value=cfg,
        ), patch(
            "retroagent.run.retro.logging.basicConfig",
        ), patch(
            "retroagent.run.retro._build_environment", return_value=_Env(),
        ), patch(
            "retroagent.run.retro.LLMClient", return_value=_Model(),
        ), patch(
            "retroagent.run.retro._build_agent_for_mode", return_value=fake_agent,
        ) as factory:
            output_dir = Path(tmp) / "benchmark"
            result = runner.invoke(app, [
                "benchmark-discovery", "--run", "--limit", "1",
                "--cases", str(PROJECT_DIR / "benchmarks" / "discovery_small.jsonl"),
                "--output-dir", str(output_dir),
            ])
            row = json.loads((output_dir / "results.jsonl").read_text().splitlines()[0])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(factory.call_args.args[0], "discovery")
        self.assertEqual(row["metrics"]["trajectory"]["decision_count"], 1)


if __name__ == "__main__":
    unittest.main()

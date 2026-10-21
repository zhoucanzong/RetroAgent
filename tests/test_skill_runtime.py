import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from retroagent.environments import RetroEnvironment
from retroagent.project import ResearchProject, ProjectError, apply_batch
from retroagent.skill_runtime import SkillSession, execute_action, finish_research
from retroagent.research_state import ResearchLedger
from retroagent.mcp_server import handle_message, serve
from retroagent.run.retro import app


class OutputTool:
    description = "Test computation"

    def __init__(self, output):
        self.output, self.calls, self.closed = output, 0, False

    def parameters_schema(self):
        return {"type": "object", "properties": {}}

    def execute(self, parameters):
        self.calls += 1
        return json.dumps(self.output)

    def close(self):
        self.closed = True


def environment(name, tool):
    env = RetroEnvironment(include_bash=False)
    env.register(name, tool)
    return env


class SkillRuntimeTests(unittest.TestCase):
    def test_direct_tool_registers_reactions_routes_and_linked_evidence(self):
        tool = OutputTool({"reactions": [{"id": "rx1", "product_smiles": "CC=O"}],
                           "routes": [{"id": "route1", "reaction_ids": ["rx1"]}]})
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            result = execute_action(project, "tool", {"tool": "predict", "parameters": {}},
                                    env=environment("predict", tool), target_ids=["H1"],
                                    research_question="Which disconnection?", rationale="Compare evidence")
            state = project.ledger().replay()
        self.assertTrue(result["success"])
        self.assertIn("rx1", state.reactions)
        self.assertIn("route1", state.routes)
        self.assertEqual(set(state.evidence[result["evidence_id"]]["target_ids"]), {"rx1", "route1", "H1"})
        self.assertEqual(state.decisions[-1]["research_question"], "Which disconnection?")

    def test_direct_negative_feedback_marks_route_failed_and_requests_replan(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            apply_batch(project, [{"operation": "register_route", "payload": {
                "id": "r1", "reaction_ids": ["rx1"],
            }}])
            tool = OutputTool({"failure": {"reaction_ids": ["rx1"], "reason": "Control contradicted route"}})
            result = execute_action(project, "tool", {"tool": "feedback", "parameters": {}},
                                    env=environment("feedback", tool))
            state = project.ledger().replay()
        self.assertFalse(result["success"])
        self.assertEqual(state.routes["r1"]["status"], "failed")
        self.assertTrue(state.replan_required)

    def test_direct_posterior_and_experiment_events_are_assimilated(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            select = OutputTool({"selected_experiment": {"id": "exp1", "rank": 1}})
            execute_action(project, "tool", {"tool": "select_experiment", "parameters": {}},
                           env=environment("select_experiment", select))
            update = OutputTool({"hypothesis_updates": [{"id": "H1", "posterior_probability": 0.8}],
                                 "posterior_entropy": 0.7, "entropy_reduction": 0.3})
            execute_action(project, "tool", {"tool": "update_hypotheses", "parameters": {}},
                           env=environment("update_hypotheses", update))
            state = project.ledger().replay()
        self.assertEqual(state.experiments["exp1"]["status"], "selected")
        self.assertEqual(state.hypotheses["H1"]["posterior_probability"], 0.8)

    def test_malformed_result_does_not_partially_register_routes(self):
        output = {"routes": [{"id": "bad_route"}], "validation_reports": [
            {"route_id": "bad_route", "confidence": "not a number"},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            result = execute_action(project, "tool", {"tool": "validate_route_portfolio", "parameters": {}},
                                    env=environment("validate_route_portfolio", OutputTool(output)))
            state = project.ledger().replay()
        self.assertFalse(result["success"])
        self.assertNotIn("bad_route", state.routes)
        self.assertTrue(state.replan_required)

    def test_transport_retries_are_idempotent_and_reject_reused_ids(self):
        tool = OutputTool({"summary": "measured"})
        action = {"tool": "measure", "parameters": {}}
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            env = environment("measure", tool)
            first = execute_action(project, "tool", action, env=env, request_id="request-1")
            second = execute_action(project, "tool", action, env=env, request_id="request-1")
            with self.assertRaisesRegex(ProjectError, "different action"):
                execute_action(project, "tool", {**action, "parameters": {"different": True}}, env=env, request_id="request-1")
        self.assertEqual(tool.calls, 1)
        self.assertTrue(second["replayed"])
        self.assertEqual(first["evidence_id"], second["evidence_id"])

    def test_interrupted_request_is_not_automatically_reexecuted(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            action = {"tool": "measure", "parameters": {}}
            project.ledger().append("execution_started", {"request_id": "interrupted", "action": {"type": "tool", **action}})
            with self.assertRaisesRegex(ProjectError, "interrupted"):
                execute_action(project, "tool", action, request_id="interrupted")

    def test_concurrent_same_request_executes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResearchProject.initialize(tmp)
            tool = OutputTool({"summary": "observation"})
            env = environment("measure", tool)
            def run(_):
                return execute_action(ResearchProject(tmp), "tool", {"tool": "measure", "parameters": {}},
                                      env=env, request_id="same")
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run, range(2)))
        self.assertEqual(tool.calls, 1)
        self.assertEqual(sum(bool(x.get("replayed")) for x in results), 1)

    def test_approved_call_is_consumed_once_without_internal_agent(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            tool = OutputTool({"summary": "source fetched"})
            env = environment("web_search", tool)
            action = {"tool": "web_search", "parameters": {"query": "kinetics"}}
            with self.assertRaisesRegex(ProjectError, "Approval required"):
                execute_action(project, "tool", action, env=env)
            self.assertEqual(runner.invoke(app, ["approve", "--project", tmp, "--json"]).exit_code, 0)
            self.assertTrue(execute_action(project, "tool", action, env=env)["success"])
            with self.assertRaisesRegex(ProjectError, "Approval required"):
                execute_action(project, "tool", action, env=env)
            consumed = [e for e in project.ledger().events if e.type == "approval_consumed"]
        self.assertEqual(tool.calls, 1)
        self.assertEqual(len(consumed), 1)

    def test_persistent_mcp_keeps_lazy_tool_and_reports_failure(self):
        tool = OutputTool({"summary": "OK"})
        constructed = []
        env = RetroEnvironment(include_bash=False)
        def factory():
            constructed.append(True)
            return tool
        env.register_lazy("measure", OutputTool, factory)
        session = SkillSession(env)
        with tempfile.TemporaryDirectory() as tmp:
            ResearchProject.initialize(tmp)
            discovery = handle_message({"id": 1, "method": "tools/call", "params": {
                "name": "retroagent_tools", "arguments": {},
            }}, session=session)
            self.assertFalse(discovery["result"]["isError"])
            self.assertEqual(constructed, [])
            request = {"id": 2, "method": "tools/call", "params": {
                "name": "retroagent_tool", "arguments": {"project": tmp, "tool": "measure", "parameters": {}},
            }}
            handle_message(request, session=session)
            handle_message(request, session=session)
            self.assertEqual(len(constructed), 1)
            self.assertEqual(tool.calls, 2)
            tool.output = {"error": "calculation unavailable"}
            response = handle_message(request, session=session)
            self.assertTrue(response["result"]["isError"])
            self.assertTrue(ResearchProject(tmp).ledger().replay().replan_required)
        session.close()
        self.assertTrue(tool.closed)

    def test_host_research_cli_never_queries_model_and_finishes_with_evidence(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp, patch("retroagent.run.retro.LLMClient.query", side_effect=AssertionError("Nested LLM")):
            project = ResearchProject.initialize(tmp)
            start = runner.invoke(app, ["research", "start", "Compare structures", "--project", tmp, "--json"])
            self.assertEqual(start.exit_code, 0, start.output)
            params = Path(tmp) / "input.json"
            params.write_text(json.dumps({"smiles": "CCO"}))
            result = runner.invoke(app, ["tool", "call", "molecule_properties", "--parameters-file", str(params),
                                        "--target", "H1", "--question", "What are the computed descriptors?",
                                        "--project", tmp, "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            evidence = json.loads(result.output)["data"]["evidence_id"]
            conclusion = Path(tmp) / "conclusion.json"
            conclusion.write_text(json.dumps({"conclusion": "Descriptors computed; synthesis untested.",
                                              "evidence_ids": [evidence], "target_ids": ["H1"]}))
            finish = runner.invoke(app, ["research", "finish", str(conclusion), "--project", tmp, "--json"])
            self.assertEqual(finish.exit_code, 0, finish.output)
            context = runner.invoke(app, ["research", "context", "--project", tmp, "--json"])
            self.assertTrue(json.loads(context.output)["data"]["conclusion"])
            self.assertEqual(project.load_manifest().status, "completed")

    def test_failed_cli_call_has_nonzero_status_and_recorded_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            result = CliRunner().invoke(app, ["tool", "call", "missing_tool", "--project", tmp, "--json"])
            self.assertEqual(result.exit_code, 3, result.output)
            self.assertFalse(json.loads(result.output)["ok"])
            self.assertEqual(len(project.ledger().replay().evidence), 1)

    def test_conclusion_rejects_unknown_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            with self.assertRaisesRegex(ProjectError, "unknown evidence"):
                finish_research(project, {"conclusion": "Unsupported", "evidence_ids": ["invented"]})
            self.assertEqual(project.ledger().replay().conclusion, {})

    def test_host_can_withdraw_without_forging_user_denial(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            with self.assertRaisesRegex(ProjectError, "Approval required"):
                execute_action(project, "command", {"command": "printf not-executed"})
            with self.assertRaisesRegex(ProjectError, "pending approvals"):
                finish_research(project, {"conclusion": "done", "evidence_ids": ["not-real"]})
            result = runner.invoke(app, ["research", "withdraw", "--reason", "Use existing evidence", "--project", tmp, "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            event = project.ledger().events[-1]
            self.assertEqual(event.type, "approval_withdrawn")
            self.assertEqual(event.actor, "host_agent")
            self.assertIsNone(project.status()["pending_approval"])

    def test_placeholder_conditions_are_not_calibrated_evidence(self):
        from retroagent.tools.condition import ConditionTool
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            result = execute_action(project, "tool", {"tool": "recommend_conditions", "parameters": {
                "product": "CCOC(C)=O", "precursors": ["CCO", "CC(=O)O"],
                "reaction_classification": "not a supported class",
            }}, env=environment("recommend_conditions", ConditionTool()))
            evidence = project.ledger().replay().evidence[result["evidence_id"]]
        self.assertEqual(evidence["assessment_status"], "needs-evidence")
        self.assertIsNone(evidence["confidence"])
        self.assertIn("rdkit", evidence["execution_metadata"]["installed_packages"])

    def test_skill_export_contains_references_and_executable_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "installed-skill"
            result = CliRunner().invoke(app, ["capability", "skill", "-o", str(output), "--json"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue((output / "references/models.md").exists())
            self.assertTrue((output / "scripts/retro.py").exists())
            self.assertIn("host", (output / "SKILL.md").read_text())


if __name__ == "__main__":
    unittest.main()

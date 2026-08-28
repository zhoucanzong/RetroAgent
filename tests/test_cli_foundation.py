import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from retroagent.agents.discovery_controller import DiscoveryController
from retroagent.capabilities import command_schema, skill_markdown
from retroagent.mcp_server import handle_message
from retroagent.project import ProjectError, ResearchProject, apply_batch
from retroagent.research_execution import PermissionGrant, PermissionMode
from retroagent.resources import ResourceRef, query_resources, resolve_resource
from retroagent.run.retro import app
from retroagent.views import render_view


PROJECT_DIR = Path(__file__).resolve().parent.parent


def _decision(action_type, action, *, stop=False, evidence_ids=None):
    return {
        "research_question": "What evidence should be collected next?",
        "action_type": action_type,
        "action": action,
        "target_ids": [],
        "rationale": "Reduce uncertainty with an auditable action.",
        "expected_observation": "A reproducible observation.",
        "falsification_condition": "The observation contradicts the claim.",
        "estimated_cost": {"time_minutes": 0.1, "compute": "low"},
        "stop": stop,
        "conclusion": "Approved action completed." if stop else "",
        "evidence_ids": evidence_ids or [],
        "satisfied_criteria": [],
    }


class _ScriptedModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def query(self, messages, tools=None):
        return {"content": json.dumps(self.decisions.pop(0)), "extra": {}}


class _EmptyEnv:
    def get_tools_spec(self):
        return []

    def execute(self, action):
        raise AssertionError(f"Unexpected tool action: {action}")


class ProjectFoundationTests(unittest.TestCase):
    def test_project_initialization_has_durable_manifest_and_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp, name="novel-route")

            manifest = project.load_manifest()

            self.assertEqual(manifest.name, "novel-route")
            self.assertEqual(manifest.schema_version, "1.0")
            self.assertTrue(project.ledger_path.exists())
            self.assertTrue(project.artifacts_dir.is_dir())
            self.assertTrue(project.cache_dir.is_dir())
            self.assertTrue(project.reports_dir.is_dir())
            self.assertEqual(ResearchProject.discover(project.artifacts_dir).root, project.root)

    def test_full_permission_requires_explicit_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ProjectError, "explicit permission scopes"):
                ResearchProject.initialize(tmp, permission_mode="full")

    def test_atomic_batch_dry_run_and_failed_batch_leave_ledger_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            actions = [
                {"operation": "set_goal", "payload": {"task": "discover an unknown route"}},
                {"operation": "register_molecule", "payload": {"smiles": "CCO", "role": "target"}},
                {"operation": "propose_hypothesis", "payload": {"claim": "H1 is plausible"}},
            ]

            preview = apply_batch(project, actions, dry_run=True)
            self.assertEqual(preview["event_count_after"], 3)
            self.assertEqual(len(project.ledger().events), 0)

            with self.assertRaisesRegex(ProjectError, "unsupported operation"):
                apply_batch(project, actions + [{"operation": "delete_everything"}])
            self.assertEqual(len(project.ledger().events), 0)

            applied = apply_batch(project, actions)
            state = project.ledger().replay()
            self.assertTrue(applied["atomic"])
            self.assertEqual(list(state.molecules), ["mol_0001"])
            self.assertEqual(list(state.hypotheses), ["hyp_0001"])

    def test_resources_use_stable_kind_ids_and_deterministic_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            apply_batch(project, [
                {"operation": "register_route", "payload": {
                    "id": "route_0001", "status": "active", "reaction_ids": ["rxn_0001"],
                    "scores": {"feasibility": 0.81},
                }},
                {"operation": "register_route", "payload": {
                    "id": "route_0002", "status": "failed", "reaction_ids": [],
                    "scores": {"feasibility": 0.1},
                }},
            ])
            state = project.ledger().replay()

            resolved = resolve_resource(ResourceRef.parse("route:route_0001"), state)
            queried = query_resources("route", state, filters=["status=active", "scores.feasibility>=0.8"])

            self.assertEqual(resolved["id"], "route_0001")
            self.assertEqual([item["id"] for item in queried], ["route_0001"])

    def test_checkpoint_diff_restore_and_replay_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            apply_batch(project, [{"operation": "set_goal", "payload": {"task": "test"}}])
            checkpoint = project.create_checkpoint("initial")
            apply_batch(project, [{"operation": "propose_hypothesis", "payload": {"claim": "new"}}])

            diff = project.checkpoint_ledger(checkpoint["id"]).replay()
            current = project.ledger().replay()
            self.assertNotEqual(diff.event_count, current.event_count)

            restored = project.restore_checkpoint(checkpoint["id"])
            self.assertEqual(restored["event_count"], 1)
            self.assertEqual(project.ledger().replay().event_count, 1)
            self.assertTrue(project.checkpoint_ledger(restored["backup"]).replay().hypotheses)

    def test_route_and_molecule_views_produce_inspectable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            apply_batch(project, [
                {"operation": "register_molecule", "payload": {"id": "mol_1", "smiles": "CCO"}},
                {"operation": "register_reaction", "payload": {"id": "rxn_1", "status": "active"}},
                {"operation": "register_route", "payload": {"id": "route_1", "reaction_ids": ["rxn_1"]}},
                {"operation": "register_experiment", "payload": {
                    "id": "exp_1", "status": "planned", "objective": "Discriminate H1 and H2",
                }},
            ])
            route_path = project.views_dir / "route.svg"
            molecule_path = project.views_dir / "molecule.svg"
            molecule_png_path = project.views_dir / "molecule.png"
            evidence_path = project.views_dir / "evidence.html"
            experiments_path = project.views_dir / "experiments.html"

            render_view(project, "route:route_1", output_format="svg", output=route_path)
            render_view(project, "molecule:mol_1", output_format="svg", output=molecule_path)
            render_view(project, "molecule:mol_1", output_format="png", output=molecule_png_path)
            render_view(project, "evidence", output_format="html", output=evidence_path)
            render_view(project, "experiments", output_format="html", output=experiments_path)

            self.assertIn("rxn_1", route_path.read_text())
            self.assertIn("<svg", molecule_path.read_text())
            self.assertTrue(molecule_png_path.read_bytes().startswith(b"\x89PNG"))
            self.assertIn("<table>", evidence_path.read_text())
            self.assertIn("Discriminate H1 and H2", experiments_path.read_text())


class CliContractTests(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_project_init_status_help_and_validation_return_json_envelopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            initialized = self.runner.invoke(app, ["project", "init", tmp, "--json"])
            status = self.runner.invoke(app, ["status", "--project", tmp, "--json"])
            help_result = self.runner.invoke(app, ["help", "agent", "--json"])
            validation = self.runner.invoke(app, ["validate", "--project", tmp, "--json"])

            for result in (initialized, status, help_result, validation):
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertTrue(json.loads(result.output)["ok"])
            self.assertEqual(json.loads(status.output)["data"]["event_count"], 0)
            self.assertTrue(json.loads(validation.output)["data"]["valid"])

    def test_batch_query_view_and_replay_work_through_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, ["project", "init", tmp])
            batch_path = Path(tmp) / "batch.json"
            batch_path.write_text(json.dumps([
                {"operation": "set_goal", "payload": {"task": "test CLI"}},
                {"operation": "register_route", "payload": {"id": "route_cli", "status": "active", "reaction_ids": []}},
            ]))

            batch = self.runner.invoke(app, ["batch", str(batch_path), "--project", tmp, "--json"])
            query = self.runner.invoke(app, ["query", "route", "--where", "status=active", "--project", tmp, "--json"])
            view = self.runner.invoke(app, ["view", "route:route_cli", "--format", "html", "--project", tmp, "--json"])
            events = self.runner.invoke(app, ["events", "--type", "route_registered", "--project", tmp, "--json"])
            explain = self.runner.invoke(app, ["explain", "route:route_cli", "--project", tmp, "--json"])
            checkpoint = self.runner.invoke(app, ["checkpoint", "create", "--label", "cli", "--project", tmp, "--json"])
            replay = self.runner.invoke(app, ["replay", "--project", tmp, "--json"])

            self.assertEqual(batch.exit_code, 0, batch.output)
            self.assertEqual(json.loads(query.output)["data"]["count"], 1)
            self.assertEqual(json.loads(events.output)["data"]["count"], 1)
            self.assertEqual(json.loads(explain.output)["data"]["value"]["id"], "route_cli")
            self.assertEqual(checkpoint.exit_code, 0, checkpoint.output)
            view_path = Path(json.loads(view.output)["data"]["output"])
            self.assertTrue(view_path.exists())
            self.assertEqual(json.loads(replay.output)["data"]["goal"]["task"], "test CLI")

            checkpoint_id = json.loads(checkpoint.output)["data"]["id"]
            ResearchProject(tmp).ledger().append("route_status_changed", {"id": "route_cli", "status": "failed"})
            diff = self.runner.invoke(app, [
                "checkpoint", "diff", checkpoint_id, "current", "--project", tmp, "--json",
            ])
            restored = self.runner.invoke(app, [
                "checkpoint", "restore", checkpoint_id, "--yes", "--project", tmp, "--json",
            ])
            self.assertEqual(json.loads(diff.output)["data"]["routes"]["changed"], ["route_cli"])
            self.assertEqual(restored.exit_code, 0, restored.output)
            self.assertEqual(ResearchProject(tmp).ledger().replay().routes["route_cli"]["status"], "active")

    def test_high_risk_exec_requires_approval_or_explicit_one_time_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, ["project", "init", tmp])
            denied = self.runner.invoke(app, [
                "exec", "command", "printf denied", "--project", tmp, "--json",
            ])
            self.assertEqual(denied.exit_code, 4)
            self.assertEqual(json.loads(denied.output)["error"]["code"], "E_APPROVAL_REQUIRED")
            self.assertIsNotNone(ResearchProject(tmp).status()["pending_approval"])

        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, ["project", "init", tmp])
            allowed = self.runner.invoke(app, [
                "exec", "command", "printf approved", "--project", tmp, "--approve", "--json",
            ])
            self.assertEqual(allowed.exit_code, 0, allowed.output)
            data = json.loads(allowed.output)["data"]
            self.assertEqual(data["result"]["stdout"], "approved")

        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, ["project", "init", tmp, "--permission-mode", "restricted"])
            forbidden = self.runner.invoke(app, [
                "exec", "command", "printf forbidden", "--project", tmp, "--approve", "--json",
            ])
            self.assertNotEqual(forbidden.exit_code, 0)
            self.assertIn("restricted mode", json.loads(forbidden.output)["error"]["message"])

        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, [
                "project", "init", tmp, "--permission-mode", "full", "--permission-scopes", "system",
            ])
            full = self.runner.invoke(app, [
                "exec", "command", "printf full", "--project", tmp, "--json",
            ])
            self.assertEqual(full.exit_code, 0, full.output)
            self.assertEqual(json.loads(full.output)["data"]["result"]["stdout"], "full")

    def test_checked_in_skill_and_mcp_description_share_the_schema(self):
        checked_in = (PROJECT_DIR / "skills" / "retroagent" / "SKILL.md").read_text()
        schema_names = {item["name"] for item in command_schema()["commands"]}
        response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        self.assertEqual(checked_in, skill_markdown())
        self.assertIn("agent resume", schema_names)
        self.assertEqual(response["result"]["tools"][0]["name"], "retroagent_command")

    def test_mcp_tool_call_executes_the_same_json_cli_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResearchProject.initialize(tmp)
            response = handle_message({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "retroagent_command",
                    "arguments": {"command": "status", "project": tmp},
                },
            })

            self.assertFalse(response["result"]["isError"])
            output = json.loads(response["result"]["content"][0]["text"])
            self.assertTrue(output["ok"])
            self.assertEqual(output["command"], "status")

    def test_reject_and_stop_are_durable_agent_lifecycle_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.runner.invoke(app, ["project", "init", tmp])
            pending = self.runner.invoke(app, [
                "exec", "command", "printf pending", "--project", tmp, "--json",
            ])
            rejected = self.runner.invoke(app, [
                "reject", "--reason", "Use a safer analysis", "--project", tmp, "--json",
            ])
            stopped = self.runner.invoke(app, [
                "agent", "stop", "--reason", "Pause the study", "--project", tmp, "--json",
            ])

            self.assertEqual(pending.exit_code, 4)
            self.assertEqual(rejected.exit_code, 0, rejected.output)
            self.assertEqual(stopped.exit_code, 0, stopped.output)
            state = ResearchProject(tmp).ledger().replay()
            self.assertTrue(state.replan_required)
            self.assertTrue(state.stopped)

    def test_agent_run_and_resume_commands_route_through_project_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            ResearchProject.initialize(tmp)
            with patch("retroagent.cli_foundation._run_agent", return_value={"exit_status": "Submitted"}) as run_agent:
                started = self.runner.invoke(app, [
                    "agent", "run", "discover a new route", "--project", tmp, "--json",
                ])
            self.assertEqual(started.exit_code, 0, started.output)
            self.assertEqual(run_agent.call_args.args[1], "discover a new route")

        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            project.ledger().append("goal_set", {"task": "resume this study"}, actor="user")
            with patch("retroagent.cli_foundation._run_agent", return_value={"exit_status": "StepLimit"}) as resume_agent:
                resumed = self.runner.invoke(app, ["agent", "resume", "--project", tmp, "--json"])
            self.assertEqual(resumed.exit_code, 0, resumed.output)
            self.assertEqual(resume_agent.call_args.args[1], "resume this study")

    def test_agent_run_discovery_v3_persists_completion_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            with patch("retroagent.cli_foundation._run_agent", return_value={"exit_status": "Submitted"}):
                result = self.runner.invoke(app, [
                    "agent", "run", "discover a new route", "--project", tmp,
                    "--discovery-v3", "--max-posterior-entropy", "0.4",
                    "--min-route-novelty", "0.2", "--json",
                ])

            goal = project.ledger().replay().goal

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(goal["discovery_v3"]["max_posterior_entropy"], 0.4)
        self.assertEqual(goal["discovery_v3"]["min_route_novelty"], 0.2)


class ApprovalResumeTests(unittest.TestCase):
    def test_approved_pending_action_executes_exactly_once_before_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = ResearchProject.initialize(tmp)
            first_model = _ScriptedModel([
                _decision("command", {"command": "printf approved > result.txt"}),
            ])
            first = DiscoveryController(
                first_model,
                _EmptyEnv(),
                ledger=project.ledger(),
                workspace=project.runs_dir / "current",
                permission_grant=PermissionGrant(mode=PermissionMode.STANDARD),
                max_cycles=1,
            ).run("test approval resume")
            self.assertEqual(first["exit_status"], "AwaitingApproval")

            approved = CliRunner().invoke(app, ["approve", "--project", tmp, "--json"])
            self.assertEqual(approved.exit_code, 0, approved.output)
            second_model = _ScriptedModel([
                _decision("stop", {}, stop=True, evidence_ids=["evidence_0001"]),
            ])
            resumed = DiscoveryController(
                second_model,
                _EmptyEnv(),
                ledger=project.ledger(),
                workspace=project.runs_dir / "current",
                permission_grant=PermissionGrant(mode=PermissionMode.STANDARD),
                max_cycles=2,
            ).run("test approval resume")

            self.assertEqual(resumed["exit_status"], "Submitted")
            self.assertEqual((project.runs_dir / "current" / "result.txt").read_text(), "approved")
            event_types = [event.type for event in project.ledger().events]
            self.assertEqual(event_types.count("approval_consumed"), 1)
            self.assertEqual(event_types.count("execution_started"), 1)


if __name__ == "__main__":
    unittest.main()

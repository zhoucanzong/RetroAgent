import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from retroagent.research_execution import (
    PermissionGrant,
    PermissionMode,
    PolicyEngine,
    ResearchExecutor,
)


class ResearchExecutionTests(unittest.TestCase):
    def test_restricted_mode_allows_tools_but_rejects_code_and_commands(self):
        policy = PolicyEngine()
        grant = PermissionGrant(mode=PermissionMode.RESTRICTED)

        tool = policy.authorize({"type": "tool", "tool": "molecule_properties"}, grant)
        code = policy.authorize({"type": "python", "code": "print(1)"}, grant)

        self.assertTrue(tool.allowed)
        self.assertFalse(code.allowed)
        self.assertFalse(code.requires_approval)

        network = policy.authorize({"type": "tool", "tool": "web_search"}, grant)
        self.assertFalse(network.allowed)
        self.assertFalse(network.requires_approval)

        bash = policy.authorize({
            "type": "tool", "tool": "bash", "parameters": {"command": "echo bypass"},
        }, grant)
        self.assertFalse(bash.allowed)
        self.assertFalse(bash.requires_approval)

        state = policy.authorize({
            "type": "state", "operation": "propose_hypothesis",
        }, grant)
        self.assertTrue(state.allowed)

    def test_standard_mode_requires_approval_for_high_risk_command(self):
        decision = PolicyEngine().authorize(
            {"type": "command", "command": "rm -rf generated-output"},
            PermissionGrant(mode=PermissionMode.STANDARD),
        )

        self.assertFalse(decision.allowed)
        self.assertTrue(decision.requires_approval)
        self.assertEqual(decision.risk, "high")
        self.assertEqual(decision.required_scope, "system")

    def test_standard_mode_requires_approval_for_python_execution(self):
        policy = PolicyEngine()
        grant = PermissionGrant(mode=PermissionMode.STANDARD)

        relative = policy.authorize({
            "type": "python", "code": "from pathlib import Path\nPath('result.json').write_text('{}')",
        }, grant)
        absolute = policy.authorize({
            "type": "python", "code": "from pathlib import Path\nPath('/tmp/escape').write_text('x')",
        }, grant)

        self.assertFalse(relative.allowed)
        self.assertTrue(relative.requires_approval)
        self.assertEqual(relative.required_scope, "code_execution")
        self.assertFalse(absolute.allowed)
        self.assertTrue(absolute.requires_approval)
        self.assertEqual(absolute.required_scope, "code_execution")

        dynamic = policy.authorize({
            "type": "python", "code": "p='/tmp/escape'; open(p, 'w').write('x')",
        }, PermissionGrant(mode=PermissionMode.FULL, scopes={"code_execution"}))
        self.assertTrue(dynamic.allowed)

    def test_standard_mode_requires_approval_for_bash_tool_and_memory_write(self):
        policy = PolicyEngine()
        grant = PermissionGrant(mode=PermissionMode.STANDARD)

        bash = policy.authorize({
            "type": "tool", "tool": "bash", "parameters": {"command": "echo x"},
        }, grant)
        memory = policy.authorize({
            "type": "tool", "tool": "research_memory", "parameters": {},
        }, grant)

        self.assertEqual((bash.allowed, bash.required_scope), (False, "system"))
        self.assertEqual((memory.allowed, memory.required_scope), (False, "project_write"))

    def test_full_mode_honors_scope_and_expiration(self):
        action = {"type": "command", "command": "rm -rf generated-output"}
        future = datetime.now(timezone.utc) + timedelta(minutes=5)
        expired = datetime.now(timezone.utc) - timedelta(minutes=1)
        policy = PolicyEngine()

        allowed = policy.authorize(action, PermissionGrant(
            mode=PermissionMode.FULL, scopes={"system"}, expires_at=future.isoformat(),
        ))
        wrong_scope = policy.authorize(action, PermissionGrant(
            mode=PermissionMode.FULL, scopes={"network"}, expires_at=future.isoformat(),
        ))
        too_late = policy.authorize(action, PermissionGrant(
            mode=PermissionMode.FULL, scopes={"system"}, expires_at=expired.isoformat(),
        ))

        self.assertTrue(allowed.allowed)
        self.assertFalse(allowed.requires_approval)
        self.assertTrue(wrong_scope.requires_approval)
        self.assertTrue(too_late.requires_approval)

    def test_python_execution_captures_output_and_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = ResearchExecutor(Path(tmp), timeout=2)

            result = executor.execute_python(
                "from pathlib import Path\n"
                "Path('result.json').write_text('{\"score\": 0.8}')\n"
                "print('computed')\n"
            )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["status"], "completed")
            self.assertIn("computed", result["stdout"])
            self.assertIn("result.json", result["artifacts"])
            self.assertTrue(Path(result["script_path"]).is_relative_to(Path(tmp)))

    def test_executor_rejects_cwd_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = ResearchExecutor(Path(tmp), timeout=1)

            with self.assertRaisesRegex(ValueError, "outside research workspace"):
                executor.execute_command("pwd", cwd=Path(tmp).parent)

    def test_timeout_is_reported_and_process_is_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ResearchExecutor(Path(tmp), timeout=0.05).execute_python(
                "import time\ntime.sleep(2)\n"
            )

            self.assertEqual(result["status"], "timed_out")
            self.assertNotEqual(result["returncode"], 0)
            self.assertIn("timed out", result["exception_info"].lower())


if __name__ == "__main__":
    unittest.main()

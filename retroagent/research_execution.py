"""Permission policy and bounded execution for discovery research actions."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class PermissionMode(str, Enum):
    RESTRICTED = "restricted"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class PermissionGrant:
    mode: PermissionMode = PermissionMode.STANDARD
    scopes: set[str] = field(default_factory=set)
    expires_at: str = ""

    def is_active(self) -> bool:
        if not self.expires_at:
            return True
        try:
            expiry = datetime.fromisoformat(self.expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) < expiry
        except ValueError:
            return False

    def covers(self, scope: str) -> bool:
        return self.is_active() and (scope in self.scopes or "*" in self.scopes)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    risk: str
    reason: str = ""
    required_scope: str = ""


class PolicyEngine:
    NETWORK_TOOLS = {"web_search", "fetch_url", "convert_identifier"}

    def authorize(self, action: dict[str, Any], grant: PermissionGrant) -> PolicyDecision:
        action_type = str(action.get("type", "tool"))
        if grant.mode == PermissionMode.RESTRICTED and action_type not in {"tool", "state"}:
            return PolicyDecision(False, False, "forbidden", "restricted mode only permits registered tools")

        risk, scope = self.classify(action)
        if grant.mode == PermissionMode.RESTRICTED and risk != "low":
            return PolicyDecision(False, False, "forbidden", "restricted mode rejects high-risk tools")
        if risk == "low":
            return PolicyDecision(True, False, risk)
        if grant.mode == PermissionMode.FULL and grant.covers(scope):
            return PolicyDecision(True, False, risk, required_scope=scope)
        return PolicyDecision(
            False,
            True,
            risk,
            f"action requires explicit '{scope}' authorization",
            required_scope=scope,
        )

    def classify(self, action: dict[str, Any]) -> tuple[str, str]:
        action_type = str(action.get("type", "tool"))
        if action_type == "tool":
            tool_name = str(action.get("tool", ""))
            parameters = action.get("parameters", {}) or {}
            if not isinstance(parameters, dict):
                return "high", "invalid"
            if tool_name == "bash":
                return "high", "system"
            if tool_name == "research_memory" and parameters.get("operation", "record") == "record":
                return "high", "project_write"
            memory_path = str(parameters.get("memory_path", ""))
            if tool_name == "research_memory" and memory_path and (
                Path(memory_path).is_absolute() or ".." in Path(memory_path).parts
            ):
                return "high", "filesystem"
            if tool_name in self.NETWORK_TOOLS:
                return "high", "network"
            return "low", ""
        if action_type == "state":
            return "low", ""
        if action_type == "python":
            return "high", "code_execution"
        if action_type == "command":
            return "high", "system"
        return "low", ""


class ResearchExecutor:
    """Run temporary analyses and capture reproducibility metadata."""

    def __init__(self, workspace: Path | str, *, timeout: float = 120):
        self.workspace = Path(workspace).absolute()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._script_count = 0

    def execute_python(self, code: str) -> dict[str, Any]:
        self._script_count += 1
        script_path = self.workspace / f"analysis_{self._script_count:04d}.py"
        script_path.write_text(code, encoding="utf-8")
        result = self._run([sys.executable, str(script_path)], cwd=self.workspace)
        result["script_path"] = str(script_path)
        return result

    def execute_command(self, command: str, *, cwd: Path | str | None = None) -> dict[str, Any]:
        resolved_cwd = Path(cwd).resolve() if cwd else self.workspace
        self._ensure_contained(resolved_cwd)
        return self._run(command, cwd=resolved_cwd, shell=True)

    def _ensure_contained(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.workspace.resolve())
        except ValueError as exc:
            raise ValueError(f"Execution cwd is outside research workspace: {path}") from exc

    def _run(
        self,
        command: list[str] | str,
        *,
        cwd: Path,
        shell: bool = False,
    ) -> dict[str, Any]:
        before = _relative_files(self.workspace)
        try:
            process = subprocess.Popen(
                command,
                shell=shell,
                cwd=cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                status = "completed" if process.returncode == 0 else "failed"
                exception_info = ""
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                status = "timed_out"
                exception_info = f"Command timed out after {self.timeout}s"
            after = _relative_files(self.workspace)
            return {
                "status": status,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "exception_info": exception_info,
                "command": command if isinstance(command, str) else list(command),
                "cwd": str(cwd),
                "artifacts": sorted(after - before),
            }
        except Exception as exc:
            return {
                "status": "failed",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "exception_info": f"{type(exc).__name__}: {exc}",
                "command": command if isinstance(command, str) else list(command),
                "cwd": str(cwd),
                "artifacts": [],
            }


def _relative_files(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }

"""Minimal MCP stdio bridge backed by the RetroAgent CLI contract."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

from retroagent.capabilities import mcp_tools
from retroagent.project import ResearchProject
from retroagent.skill_runtime import SkillSession
from retroagent.tools.retrochimera import _diagnostic_output
from retroagent import __version__


_SESSION = SkillSession()


MCP_PROTOCOL_VERSION = "2024-11-05"


def handle_message(message: dict[str, Any], *, cwd: Path | str = ".", session=None) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "retroagent", "version": __version__},
        })
    if method == "tools/list":
        return _result(request_id, {"tools": mcp_tools()})
    if method == "tools/call":
        params = message.get("params", {}) or {}
        if not isinstance(params, dict):
            return _error(request_id, -32602, "params must be an object")
        tool_name = params.get("name")
        if tool_name not in {"retroagent_command", "retroagent_tool", "retroagent_tools"}:
            return _error(request_id, -32602, "Unknown MCP tool")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "arguments must be an object")
        try:
            active_session = session or _SESSION
            if tool_name == "retroagent_tools":
                with _diagnostic_output():
                    data = {"tools": active_session.environment.get_tools_spec()}
                return _result(request_id, {"content": [{"type": "text", "text": json.dumps(data)}], "isError": False})
            if tool_name == "retroagent_tool":
                parameters = arguments.get("parameters", {})
                targets = arguments.get("target_ids", [])
                if not isinstance(parameters, dict) or not isinstance(targets, list) or any(not isinstance(x, str) for x in targets):
                    raise ValueError("parameters must be an object and target_ids an array of strings")
                with _diagnostic_output():
                    data = active_session.call(
                        ResearchProject.discover(arguments.get("project") or cwd),
                        str(arguments.get("tool", "")), parameters,
                        target_ids=targets, research_question=str(arguments.get("question", "")),
                        rationale=str(arguments.get("rationale", "")),
                        request_id=str(arguments.get("request_id", "")),
                    )
                return _result(request_id, {"content": [{"type": "text", "text": json.dumps(data)}],
                                           "isError": not data["success"]})
            content, is_error = run_cli_command(
                str(arguments.get("command", "")),
                cwd=Path(str(arguments.get("project") or cwd)),
            )
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            return _error(request_id, -32602, str(exc))
        return _result(request_id, {
            "content": [{"type": "text", "text": content}],
            "isError": is_error,
        })
    if method == "ping":
        return _result(request_id, {})
    return _error(request_id, -32601, f"Method not found: {method}")


def run_cli_command(command: str, *, cwd: Path) -> tuple[str, bool]:
    args = shlex.split(command)
    if not args:
        raise ValueError("command cannot be empty")
    if args[:2] == ["mcp", "serve"]:
        raise ValueError("Nested MCP servers are not allowed")
    if "--json" not in args:
        args.append("--json")
    process = subprocess.run(
        [sys.executable, "-m", "retroagent.run.retro", *args],
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if process.stderr:
        sys.stderr.write(process.stderr)
    return process.stdout, process.returncode != 0


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, *, cwd: Path | str = ".") -> None:
    try:
        _serve(stdin, stdout, cwd=cwd)
    finally:
        _SESSION.close()


def _serve(stdin, stdout, *, cwd):
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("MCP message must be an object")
            response = handle_message(message, cwd=cwd)
        except (json.JSONDecodeError, ValueError) as exc:
            response = _error(None, -32700, str(exc))
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdout.flush()


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

"""Minimal MCP stdio bridge backed by the RetroAgent CLI contract."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, TextIO

from retroagent.capabilities import mcp_tools


MCP_PROTOCOL_VERSION = "2024-11-05"


def handle_message(message: dict[str, Any], *, cwd: Path | str = ".") -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        return _result(request_id, {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "retroagent", "version": "0.2.0"},
        })
    if method == "tools/list":
        return _result(request_id, {"tools": mcp_tools()})
    if method == "tools/call":
        params = message.get("params", {}) or {}
        if params.get("name") != "retroagent_command":
            return _error(request_id, -32602, "Unknown MCP tool")
        arguments = params.get("arguments", {}) or {}
        try:
            content, is_error = run_cli_command(
                str(arguments.get("command", "")),
                cwd=Path(str(arguments.get("project") or cwd)),
            )
        except ValueError as exc:
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
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    return process.stdout, process.returncode != 0


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout, *, cwd: Path | str = ".") -> None:
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

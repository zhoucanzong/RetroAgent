"""Single source of truth for agent-facing CLI, Skill, and MCP capabilities."""

from __future__ import annotations

import json
from typing import Any


COMMAND_SCHEMA_VERSION = "1.0"


def _command(name: str, risk: str, summary: str, usage: str) -> dict[str, str]:
    return {"name": name, "risk": risk, "summary": summary, "usage": usage}


COMMANDS: list[dict[str, Any]] = [
    _command("research start", "low", "Record a research goal for the host Agent; no internal LLM call.", "retroagent research start TASK [--project PATH] [--json]"),
    _command("research context", "low", "Restore research context for the host Agent.", "retroagent research context [--project PATH] [--json]"),
    _command("research withdraw", "low", "Withdraw a pending host action without attributing rejection to the user.", "retroagent research withdraw --reason TEXT [--project PATH] [--json]"),
    _command("research finish", "low", "Record an evidence-linked conclusion after deterministic reference checks.", "retroagent research finish CONCLUSION.json [--project PATH] [--json]"),
    _command("project init", "low", "Create a persistent .retroagent research workspace.", "retroagent project init [PATH] [--name NAME] [--permission-mode MODE] [--permission-scopes CSV] [--json]"),
    _command("status", "low", "Inspect project, run, entity, and pending-approval status.", "retroagent status [--project PATH] [--json]"),
    _command("agent run", "controlled", "Start an autonomous evidence-seeking chemistry loop.", "retroagent agent run GOAL [--discovery-v3] [--max-posterior-entropy N] [--min-route-novelty N] [--project PATH] [--model MODEL] [--max-cycles N] [--json]"),
    _command("agent resume", "controlled", "Resume a ledger-backed run, including an approved pending action.", "retroagent agent resume [--project PATH] [--model MODEL] [--max-cycles N] [--json]"),
    _command("agent stop", "low", "Mark a project stopped without deleting its ledger.", "retroagent agent stop [--reason TEXT] [--project PATH] [--json]"),
    _command("approve", "high", "Approve exactly one pending high-risk action.", "retroagent approve [--project PATH] [--json]"),
    _command("reject", "low", "Reject a pending action and require replanning.", "retroagent reject [--reason TEXT] [--project PATH] [--json]"),
    _command("events", "low", "Read the append-only research event stream.", "retroagent events [--type TYPE] [--limit N] [--project PATH] [--json]"),
    _command("view", "low", "Render project or stable resources as text, JSON, HTML, SVG, or PNG.", "retroagent view RESOURCE [--format text|json|html|svg|png] [-o PATH] [--project PATH] [--json]"),
    _command("query", "low", "Filter resources using deterministic field predicates.", "retroagent query KIND [--where FIELD=VALUE ...] [--contains TEXT] [--project PATH] [--json]"),
    _command("explain", "low", "Resolve an entity and show linked events and evidence.", "retroagent explain KIND:ID [--project PATH] [--json]"),
    _command("validate", "low", "Validate manifest, ledger replay, references, and approval state.", "retroagent validate [--strict] [--project PATH] [--json]"),
    _command("batch", "controlled", "Atomically apply structured state operations; supports dry-run.", "retroagent batch ACTIONS.json [--dry-run] [--project PATH] [--json]"),
    _command("checkpoint create", "low", "Snapshot the current event ledger.", "retroagent checkpoint create [--label TEXT] [--project PATH] [--json]"),
    _command("checkpoint diff", "low", "Compare current or checkpoint-projected research states.", "retroagent checkpoint diff LEFT [RIGHT] [--project PATH] [--json]"),
    _command("checkpoint restore", "high", "Restore state after creating an automatic backup checkpoint.", "retroagent checkpoint restore ID --yes [--project PATH] [--json]"),
    _command("replay", "low", "Rebuild deterministic research state from ledger events.", "retroagent replay [--checkpoint ID] [--until-event ID] [--project PATH] [--json]"),
    _command("tool list", "low", "List registered chemistry tools and JSON parameter schemas.", "retroagent tool list [--json]"),
    _command("tool call", "policy", "Execute and assimilate one tool result; no internal LLM call.", "retroagent tool call NAME [--parameters JSON|--parameters-file PATH] [--target ID ...] [--question TEXT] [--rationale TEXT] [--request-id ID] [--approve] [--project PATH] [--json]"),
    _command("exec python", "high", "Run temporary Python in the project run workspace.", "retroagent exec python (--code CODE|--file PATH) [--approve] [--project PATH] [--json]"),
    _command("exec command", "high", "Run a system command in the project run workspace.", "retroagent exec command COMMAND [--approve] [--project PATH] [--json]"),
    _command("help", "low", "Return this machine-readable command schema.", "retroagent help [COMMAND] [--json]"),
    _command("capability skill", "low", "Export the complete maintained research Skill bundle.", "retroagent capability skill [-o PATH] [--json]"),
    _command("capability mcp", "low", "Describe MCP tools generated from this schema.", "retroagent capability mcp [--json]"),
    _command("mcp serve", "policy", "Expose the same CLI contract through MCP stdio.", "retroagent mcp serve [--project PATH]"),
]


def command_schema(command: str = "") -> dict[str, Any]:
    commands = COMMANDS
    if command:
        commands = [item for item in commands if item["name"] == command or item["name"].startswith(command + " ")]
        if not commands:
            raise ValueError(f"Unknown command schema: {command}")
    return {
        "schema_version": COMMAND_SCHEMA_VERSION,
        "program": "retroagent",
        "resource_addressing": "kind:id",
        "output_envelope": {
            "success": {"ok": True, "command": "string", "data": "object", "warnings": "array", "meta": "object"},
            "error": {"ok": False, "command": "string", "error": {"code": "string", "message": "string", "details": "object"}},
        },
        "commands": commands,
    }


def mcp_tools() -> list[dict[str, Any]]:
    return [{
        "name": "retroagent_command",
        "description": (
            "Run one RetroAgent CLI command. Use 'help --json' or 'tool list --json' before guessing syntax. "
            "High-risk actions remain subject to the project's permission and approval policy."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Arguments after the retroagent executable."},
                "project": {"type": "string", "description": "Project root; defaults to the MCP server working directory."},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    }, {
        "name": "retroagent_tools", "description": "Discover chemistry tool schemas without loading model weights.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    }, {
        "name": "retroagent_tool",
        "description": "Execute one host-selected chemistry action, retain model cache, and record linked evidence/state. No internal LLM call. Reuse request_id only when retrying the identical request.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"}, "parameters": {"type": "object"},
                "project": {"type": "string"}, "target_ids": {"type": "array", "items": {"type": "string"}},
                "question": {"type": "string"}, "rationale": {"type": "string"},
                "request_id": {"type": "string"},
            },
            "required": ["tool", "parameters"], "additionalProperties": False,
        },
    }]


def skill_source_dir():
    from pathlib import Path
    package = Path(__file__).resolve().parent
    checkout = package.parent / "skills" / "retroagent"
    return checkout if (checkout / "SKILL.md").exists() else package / "skill_assets"


def skill_markdown() -> str:
    """Load the maintained research Skill; tool schemas remain generated separately."""
    return (skill_source_dir() / "SKILL.md").read_text(encoding="utf-8")


def schema_json() -> str:
    return json.dumps(command_schema(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

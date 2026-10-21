"""Agent-native CLI commands layered over RetroAgent's deterministic core."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import typer

from retroagent import get_config
from retroagent.capabilities import command_schema, mcp_tools, skill_markdown, skill_source_dir
from retroagent.project import (
    ProjectError,
    ResearchProject,
    apply_batch,
    diff_states,
    state_to_dict,
    unresolved_approval,
)
from retroagent.research_execution import PermissionGrant, PermissionMode
from retroagent.research_state import ResearchLedger
from retroagent.resources import RESOURCE_COLLECTIONS, ResourceRef, query_resources, resolve_resource
from retroagent.views import render_view
from retroagent.skill_runtime import project_lock


CONTRACT_VERSION = "1.0"
project_app = typer.Typer(help="Create and configure durable research projects.")
agent_app = typer.Typer(help="Run and resume the autonomous chemistry controller.")
checkpoint_app = typer.Typer(help="Create, compare, and restore ledger checkpoints.")
tool_app = typer.Typer(help="Discover and call registered chemistry tools.")
exec_app = typer.Typer(help="Run policy-controlled code and system commands.")
capability_app = typer.Typer(help="Export schema-derived agent capabilities.")
mcp_app = typer.Typer(help="Expose RetroAgent through MCP stdio.")


def _envelope(command: str, data: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "command": command,
        "data": data,
        "warnings": warnings or [],
        "meta": {"contract_version": CONTRACT_VERSION},
    }


def _emit(
    command: str,
    data: Any,
    *,
    json_output: bool,
    text: str | None = None,
    warnings: list[str] | None = None,
) -> None:
    if json_output:
        typer.echo(json.dumps(_envelope(command, data, warnings), ensure_ascii=False, sort_keys=True))
        return
    if text is not None:
        typer.echo(text)
    elif isinstance(data, str):
        typer.echo(data)
    else:
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    for warning in warnings or []:
        typer.echo(f"WARNING: {warning}", err=True)


def _fail(command: str, code: str, message: str, *, json_output: bool, details: Any = None, exit_code: int = 2) -> None:
    if json_output:
        typer.echo(json.dumps({
            "ok": False,
            "command": command,
            "error": {"code": code, "message": message, "details": details or {}},
            "meta": {"contract_version": CONTRACT_VERSION},
        }, ensure_ascii=False, sort_keys=True))
    else:
        typer.echo(f"ERROR [{code}]: {message}", err=True)
    raise typer.Exit(exit_code)


def _project(path: Path) -> ResearchProject:
    return ResearchProject.discover(path)


@project_app.command("init")
def project_init(
    path: Path = typer.Argument(Path("."), help="Project root."),
    name: str = typer.Option("", "--name"),
    permission_mode: str = typer.Option("standard", "--permission-mode"),
    permission_scopes: str = typer.Option("", "--permission-scopes"),
    force: bool = typer.Option(False, "--force"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Create a `.retroagent` project workspace."""
    try:
        project = ResearchProject.initialize(
            path,
            name=name,
            permission_mode=permission_mode,
            permission_scopes=permission_scopes.split(","),
            force=force,
        )
    except ProjectError as exc:
        _fail("project init", "E_PROJECT", str(exc), json_output=json_output)
    data = project.status()
    _emit("project init", data, json_output=json_output, text=f"Initialized RetroAgent project at {project.control_dir}")


@project_app.command("config")
def project_config(
    project_path: Path = typer.Option(Path("."), "--project"),
    permission_mode: str | None = typer.Option(None, "--permission-mode"),
    permission_scopes: str | None = typer.Option(None, "--permission-scopes"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect or update project permission configuration."""
    try:
        project = _project(project_path)
        manifest = project.load_manifest()
        if permission_mode is not None:
            if permission_mode not in {"restricted", "standard", "full"}:
                raise ProjectError("permission_mode must be restricted, standard, or full")
            manifest.permission_mode = permission_mode
        if permission_scopes is not None:
            manifest.permission_scopes = sorted({item.strip() for item in permission_scopes.split(",") if item.strip()})
        if manifest.permission_mode == "full" and not manifest.permission_scopes:
            raise ProjectError("full permission mode requires explicit permission scopes")
        project.save_manifest(manifest)
    except ProjectError as exc:
        _fail("project config", "E_PROJECT", str(exc), json_output=json_output)
    _emit("project config", manifest.to_dict(), json_output=json_output)


def status_command(
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show current project, entity, run, and approval status."""
    try:
        data = _project(project_path).status()
    except (ProjectError, ValueError) as exc:
        _fail("status", "E_PROJECT", str(exc), json_output=json_output)
    status = data["project"]["status"]
    text = f"{data['project']['name']}: {status} | events={data['event_count']} | pending_approval={bool(data['pending_approval'])}"
    _emit("status", data, json_output=json_output, text=text)


def events_command(
    project_path: Path = typer.Option(Path("."), "--project"),
    event_type: str = typer.Option("", "--type"),
    limit: int = typer.Option(50, "--limit", min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Read the append-only research event stream."""
    try:
        events = list(_project(project_path).ledger().events)
        if event_type:
            events = [event for event in events if event.type == event_type]
        rows = [event.to_dict() for event in events[-limit:]]
    except (ProjectError, ValueError) as exc:
        _fail("events", "E_LEDGER", str(exc), json_output=json_output)
    text = "\n".join(f"{row['id']} {row['type']} actor={row['actor']} cycle={row['cycle']}" for row in rows)
    _emit("events", {"events": rows, "count": len(rows)}, json_output=json_output, text=text)


def view_command(
    resource: str = typer.Argument("state", help="state, project, or kind:id."),
    project_path: Path = typer.Option(Path("."), "--project"),
    output_format: str = typer.Option("text", "--format"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Render a stable resource as text, JSON, HTML, SVG, or PNG."""
    try:
        project = _project(project_path)
        if output is None and output_format in {"html", "svg", "png"}:
            slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", resource)
            output = project.views_dir / f"{slug}.{output_format}"
        data = render_view(project, resource, output_format=output_format, output=output)
    except (ProjectError, ValueError, KeyError) as exc:
        _fail("view", "E_RESOURCE", str(exc), json_output=json_output)
    text = data.get("content") or f"Wrote {data['format']} view to {data['output']}"
    _emit("view", data, json_output=json_output, text=text)


def query_command(
    kind: str = typer.Argument(..., help="molecule, hypothesis, route, reaction, evidence, or experiment."),
    project_path: Path = typer.Option(Path("."), "--project"),
    where: list[str] = typer.Option([], "--where"),
    contains: str = typer.Option("", "--contains"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Query resources with deterministic field predicates."""
    try:
        project = _project(project_path)
        results = query_resources(kind, project.ledger().replay(), filters=where, contains=contains)
    except (ProjectError, ValueError) as exc:
        _fail("query", "E_QUERY", str(exc), json_output=json_output)
    text = "\n".join(f"{kind}:{item.get('id', '?')} {item.get('status', '')}".rstrip() for item in results)
    _emit("query", {"kind": kind, "results": results, "count": len(results)}, json_output=json_output, text=text)


def explain_command(
    resource: str = typer.Argument(..., help="Stable kind:id resource."),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Show a resource together with linked evidence and causal events."""
    try:
        ledger = _project(project_path).ledger()
        state = ledger.replay()
        ref = ResourceRef.parse(resource)
        resolved = resolve_resource(ref, state, ledger.events)
        linked_events = [event.to_dict() for event in ledger.events if _contains_id(event.payload, ref.identifier)]
        linked_evidence = [item for item in state.evidence.values() if ref.identifier in item.get("target_ids", [])]
        data = {"resource": resource, "value": resolved, "evidence": linked_evidence, "events": linked_events}
    except (ProjectError, ValueError, KeyError) as exc:
        _fail("explain", "E_RESOURCE", str(exc), json_output=json_output)
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)
    _emit("explain", data, json_output=json_output, text=text)


def validate_command(
    project_path: Path = typer.Option(Path("."), "--project"),
    strict: bool = typer.Option(False, "--strict"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Validate manifest, ledger replay, references, and approval state."""
    try:
        project = _project(project_path)
        manifest = project.load_manifest()
        ledger = project.ledger()
        state = ledger.replay()
    except (ProjectError, ValueError) as exc:
        _fail("validate", "E_VALIDATION", str(exc), json_output=json_output, exit_code=3)
    warnings = []
    known_ids = set(state.molecules) | set(state.hypotheses) | set(state.routes) | set(state.reactions) | set(state.experiments)
    external_targets = sorted({
        str(target)
        for evidence in state.evidence.values()
        for target in evidence.get("target_ids", [])
        if str(target) not in known_ids
    })
    if external_targets:
        warnings.append(f"Evidence references external or unresolved targets: {', '.join(external_targets)}")
    pending = unresolved_approval(ledger.events)
    if pending:
        warnings.append(f"Pending approval: {pending.request_event_id} scope={pending.scope}")
    data = {
        "valid": not (strict and warnings),
        "schema_version": manifest.schema_version,
        "event_count": len(ledger.events),
        "replay_event_count": state.event_count,
        "warnings": warnings,
    }
    if strict and warnings:
        _fail("validate", "E_VALIDATION_WARNING", "Strict validation found warnings", json_output=json_output, details=data, exit_code=3)
    _emit("validate", data, json_output=json_output, text=f"Valid project ({len(ledger.events)} events)", warnings=warnings)


def batch_command(
    input_path: Path = typer.Argument(..., help="JSON array of structured state operations."),
    project_path: Path = typer.Option(Path("."), "--project"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Atomically apply structured ledger operations."""
    try:
        actions = json.loads(input_path.read_text(encoding="utf-8"))
        data = apply_batch(_project(project_path), actions, dry_run=dry_run)
    except (OSError, json.JSONDecodeError, ProjectError, ValueError) as exc:
        _fail("batch", "E_BATCH", str(exc), json_output=json_output, exit_code=3)
    text = f"{'Validated' if dry_run else 'Applied'} {len(data['created_events'])} operations atomically"
    _emit("batch", data, json_output=json_output, text=text)


def replay_command(
    project_path: Path = typer.Option(Path("."), "--project"),
    checkpoint: str = typer.Option("", "--checkpoint"),
    until_event: str = typer.Option("", "--until-event"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Deterministically rebuild state from current or checkpoint events."""
    try:
        project = _project(project_path)
        source = project.checkpoint_ledger(checkpoint) if checkpoint else project.ledger()
        if until_event:
            scratch = ResearchLedger()
            found = False
            for event in source.events:
                scratch.append_event(event)
                if event.id == until_event:
                    found = True
                    break
            if not found:
                raise ProjectError(f"Unknown event: {until_event}")
            source = scratch
        data = state_to_dict(source.replay())
    except (ProjectError, ValueError) as exc:
        _fail("replay", "E_REPLAY", str(exc), json_output=json_output, exit_code=3)
    _emit("replay", data, json_output=json_output, text=json.dumps(data, indent=2, ensure_ascii=False))


@checkpoint_app.command("create")
def checkpoint_create(
    label: str = typer.Option("", "--label"),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        data = _project(project_path).create_checkpoint(label)
    except (ProjectError, ValueError) as exc:
        _fail("checkpoint create", "E_CHECKPOINT", str(exc), json_output=json_output)
    _emit("checkpoint create", data, json_output=json_output, text=f"Created checkpoint {data['id']}")


@checkpoint_app.command("list")
def checkpoint_list(
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        rows = _project(project_path).list_checkpoints()
    except ProjectError as exc:
        _fail("checkpoint list", "E_CHECKPOINT", str(exc), json_output=json_output)
    _emit("checkpoint list", {"checkpoints": rows}, json_output=json_output)


@checkpoint_app.command("diff")
def checkpoint_diff(
    left: str = typer.Argument(..., help="Checkpoint id or current."),
    right: str = typer.Argument("current", help="Checkpoint id or current."),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    try:
        project = _project(project_path)
        left_ledger = project.ledger() if left == "current" else project.checkpoint_ledger(left)
        right_ledger = project.ledger() if right == "current" else project.checkpoint_ledger(right)
        data = diff_states(left_ledger.replay(), right_ledger.replay())
    except (ProjectError, ValueError) as exc:
        _fail("checkpoint diff", "E_CHECKPOINT", str(exc), json_output=json_output)
    _emit("checkpoint diff", data, json_output=json_output)


@checkpoint_app.command("restore")
def checkpoint_restore(
    checkpoint_id: str = typer.Argument(...),
    project_path: Path = typer.Option(Path("."), "--project"),
    yes: bool = typer.Option(False, "--yes", help="Confirm ledger state replacement."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if not yes:
        _fail("checkpoint restore", "E_APPROVAL_REQUIRED", "Restore requires --yes", json_output=json_output, exit_code=4)
    try:
        data = _project(project_path).restore_checkpoint(checkpoint_id)
    except (ProjectError, ValueError) as exc:
        _fail("checkpoint restore", "E_CHECKPOINT", str(exc), json_output=json_output)
    _emit("checkpoint restore", data, json_output=json_output, text=f"Restored checkpoint {checkpoint_id}")


def approve_command(
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Approve exactly one pending action."""
    try:
        project = _project(project_path)
        with project_lock(project):
            ledger = project.ledger()
            pending = unresolved_approval(ledger.events)
            if pending is None:
                raise ProjectError("No unresolved approval request")
            event = ledger.append(
                "approval_granted",
                {"scope": pending.scope, "action": pending.action, "one_time": True},
                actor="user",
                causal_parent_id=pending.request_event_id,
            )
            project.set_status("approved")
            data = {"approval_event": event.to_dict(), "pending": pending.to_dict()}
    except (ProjectError, ValueError) as exc:
        _fail("approve", "E_APPROVAL", str(exc), json_output=json_output)
    _emit("approve", data, json_output=json_output, text=f"Approved {pending.request_event_id} for one execution")


def reject_command(
    reason: str = typer.Option("Rejected by user", "--reason"),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Reject a pending action and require replanning."""
    try:
        project = _project(project_path)
        with project_lock(project):
            ledger = project.ledger()
            pending = unresolved_approval(ledger.events)
            if pending is None:
                raise ProjectError("No unresolved approval request")
            denied = ledger.append(
                "approval_denied",
                {"reason": reason, "action": pending.action},
                actor="user",
                causal_parent_id=pending.request_event_id,
            )
            ledger.append("decision_rejected", {"reason": reason, "action": pending.action}, actor="controller")
            ledger.append("replan_required", {"reason": reason}, actor="controller")
            project.set_status("active")
    except (ProjectError, ValueError) as exc:
        _fail("reject", "E_APPROVAL", str(exc), json_output=json_output)
    _emit("reject", denied.to_dict(), json_output=json_output, text=f"Rejected {pending.request_event_id}")


@agent_app.command("run")
def agent_run(
    goal: str = typer.Argument(...),
    project_path: Path = typer.Option(Path("."), "--project"),
    model: str | None = typer.Option(None, "--model", "-m"),
    api_key: str | None = typer.Option(None, "--api-key"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_cycles: int = typer.Option(30, "--max-cycles", min=1),
    discovery_v3: bool = typer.Option(False, "--discovery-v3", help="Require the active-learning and independent route-validation loop."),
    max_posterior_entropy: float = typer.Option(0.65, "--max-posterior-entropy", min=0.0),
    min_route_novelty: float = typer.Option(0.1, "--min-route-novelty", min=0.0, max=1.0),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Start or continue an autonomous chemistry research goal."""
    try:
        project = _project(project_path)
        if project.ledger().replay().goal:
            raise ProjectError("Project already has a goal; use 'retroagent agent resume'")
        if discovery_v3:
            project.ledger().append("goal_set", {
                "task": goal,
                "discovery_v3": {
                    "min_competing_hypotheses": 2,
                    "require_active_experiment_selection": True,
                    "require_posterior_update": True,
                    "max_posterior_entropy": max_posterior_entropy,
                    "require_multi_source_route": True,
                    "min_route_novelty": min_route_novelty,
                    "require_independent_validation": True,
                },
            }, actor="user", cycle=0)
        result = _run_agent(project, goal, model=model, api_key=api_key, base_url=base_url, max_cycles=max_cycles)
    except (ProjectError, ValueError) as exc:
        _fail("agent run", "E_AGENT", str(exc), json_output=json_output)
    _emit("agent run", result, json_output=json_output, text=f"Agent exit_status={result.get('exit_status')}")


@agent_app.command("resume")
def agent_resume(
    project_path: Path = typer.Option(Path("."), "--project"),
    model: str | None = typer.Option(None, "--model", "-m"),
    api_key: str | None = typer.Option(None, "--api-key"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_cycles: int = typer.Option(30, "--max-cycles", min=1),
    force: bool = typer.Option(False, "--force"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Resume from the event ledger and consume any exact one-time approval."""
    try:
        project = _project(project_path)
        state = project.ledger().replay()
        if not state.goal.get("task"):
            raise ProjectError("Project has no research goal")
        if state.stopped and not force:
            raise ProjectError("Project is stopped; pass --force to resume")
        result = _run_agent(
            project, str(state.goal["task"]), model=model, api_key=api_key, base_url=base_url, max_cycles=max_cycles,
        )
    except (ProjectError, ValueError) as exc:
        _fail("agent resume", "E_AGENT", str(exc), json_output=json_output)
    _emit("agent resume", result, json_output=json_output, text=f"Agent exit_status={result.get('exit_status')}")


@agent_app.command("stop")
def agent_stop(
    reason: str = typer.Option("Stopped by user", "--reason"),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Stop future resume attempts while preserving all research state."""
    try:
        project = _project(project_path)
        event = project.ledger().append("research_stopped", {"reason": reason}, actor="user")
        project.set_status("stopped")
    except (ProjectError, ValueError) as exc:
        _fail("agent stop", "E_AGENT", str(exc), json_output=json_output)
    _emit("agent stop", event.to_dict(), json_output=json_output, text="Project stopped; ledger preserved")


@agent_app.command("status")
def agent_status(
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    status_command(project_path, json_output)


@agent_app.command("approve")
def agent_approve(
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    approve_command(project_path, json_output)


@agent_app.command("reject")
def agent_reject(
    reason: str = typer.Option("Rejected by user", "--reason"),
    project_path: Path = typer.Option(Path("."), "--project"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    reject_command(reason, project_path, json_output)


@tool_app.command("list")
def tool_list(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """List chemistry tools and their machine-readable parameter schemas."""
    from retroagent.run.retro import _build_environment

    tools = _build_environment().get_tools_spec()
    _emit("tool list", {"tools": tools, "count": len(tools)}, json_output=json_output)


@tool_app.command("call")
def tool_call(
    name: str = typer.Argument(...),
    parameters: str = typer.Option("{}", "--parameters", "--params"),
    project_path: Path = typer.Option(Path("."), "--project"),
    approve: bool = typer.Option(False, "--approve", help="Explicitly approve this exact high-risk call."),
    json_output: bool = typer.Option(False, "--json"),
    parameters_file: Path | None = typer.Option(None, "--parameters-file"),
    target_ids: list[str] = typer.Option([], "--target"),
    research_question: str = typer.Option("", "--question"),
    rationale: str = typer.Option("", "--rationale"),
    request_id: str = typer.Option("", "--request-id"),
) -> None:
    """Call one registered tool under the project permission policy."""
    try:
        if parameters_file and parameters != "{}":
            raise ValueError("Use --parameters or --parameters-file, not both")
        parsed = json.loads(parameters_file.read_text() if parameters_file else parameters)
        if not isinstance(parsed, dict):
            raise ValueError("parameters must be a JSON object")
        action = {"tool": name, "parameters": parsed}
        result = _execute_project_action(_project(project_path), "tool", action, explicitly_approved=approve,
                                        target_ids=target_ids, research_question=research_question,
                                        rationale=rationale, request_id=request_id)
    except (OSError, json.JSONDecodeError, ProjectError, ValueError) as exc:
        code = "E_APPROVAL_REQUIRED" if "approval required" in str(exc).lower() else "E_TOOL"
        _fail("tool call", code, str(exc), json_output=json_output, exit_code=4 if code == "E_APPROVAL_REQUIRED" else 2)
    _emit_action("tool call", result, json_output)


@exec_app.command("python")
def exec_python(
    code: str = typer.Option("", "--code"),
    file: Path | None = typer.Option(None, "--file"),
    project_path: Path = typer.Option(Path("."), "--project"),
    approve: bool = typer.Option(False, "--approve"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run temporary Python with captured artifacts and approval policy."""
    try:
        if file:
            code = file.read_text(encoding="utf-8")
        if not code:
            raise ValueError("Provide --code or --file")
        result = _execute_project_action(_project(project_path), "python", {"code": code}, explicitly_approved=approve)
    except (OSError, ProjectError, ValueError) as exc:
        code_name = "E_APPROVAL_REQUIRED" if "approval required" in str(exc).lower() else "E_EXECUTION"
        _fail("exec python", code_name, str(exc), json_output=json_output, exit_code=4 if code_name == "E_APPROVAL_REQUIRED" else 2)
    _emit_action("exec python", result, json_output)


@exec_app.command("command")
def exec_command(
    command: str = typer.Argument(...),
    project_path: Path = typer.Option(Path("."), "--project"),
    approve: bool = typer.Option(False, "--approve"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Run a system command with captured artifacts and approval policy."""
    try:
        result = _execute_project_action(
            _project(project_path), "command", {"command": command}, explicitly_approved=approve,
        )
    except (ProjectError, ValueError) as exc:
        code_name = "E_APPROVAL_REQUIRED" if "approval required" in str(exc).lower() else "E_EXECUTION"
        _fail("exec command", code_name, str(exc), json_output=json_output, exit_code=4 if code_name == "E_APPROVAL_REQUIRED" else 2)
    _emit_action("exec command", result, json_output)


def help_command(
    command: str = typer.Argument("", help="Optional command path."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Return the machine-readable CLI schema."""
    try:
        data = command_schema(command)
    except ValueError as exc:
        _fail("help", "E_COMMAND", str(exc), json_output=json_output)
    _emit("help", data, json_output=json_output)


@capability_app.command("skill")
def capability_skill(
    output: Path | None = typer.Option(None, "--output", "-o"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Export the complete maintained Skill bundle and its supporting references."""
    content = skill_markdown()
    if output:
        destination = output.parent if output.suffix == ".md" else output
        destination.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != skill_source_dir().resolve():
            shutil.copytree(skill_source_dir(), destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if output.suffix == ".md":
            output.write_text(content, encoding="utf-8")
    data = {"output": str(output.resolve()) if output else None, "content": None if output else content}
    _emit("capability skill", data, json_output=json_output, text=content if not output else f"Wrote {output}")


@capability_app.command("mcp")
def capability_mcp(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Describe MCP tools generated from the command schema."""
    _emit("capability mcp", {"tools": mcp_tools()}, json_output=json_output)


@mcp_app.command("serve")
def mcp_serve(
    project_path: Path = typer.Option(Path("."), "--project"),
) -> None:
    """Serve the RetroAgent MCP bridge over stdio."""
    from retroagent.mcp_server import serve

    serve(cwd=project_path)


def register_cli(app: typer.Typer) -> None:
    from retroagent.skill_commands import research_app
    app.add_typer(research_app, name="research")
    app.add_typer(project_app, name="project")
    app.add_typer(agent_app, name="agent")
    app.add_typer(checkpoint_app, name="checkpoint")
    app.add_typer(tool_app, name="tool")
    app.add_typer(exec_app, name="exec")
    app.add_typer(capability_app, name="capability")
    app.add_typer(mcp_app, name="mcp")
    app.command("status")(status_command)
    app.command("events")(events_command)
    app.command("view")(view_command)
    app.command("query")(query_command)
    app.command("explain")(explain_command)
    app.command("validate")(validate_command)
    app.command("batch")(batch_command)
    app.command("replay")(replay_command)
    app.command("approve")(approve_command)
    app.command("reject")(reject_command)
    app.command("help")(help_command)


def _run_agent(
    project: ResearchProject,
    goal: str,
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    max_cycles: int,
) -> dict[str, Any]:
    from retroagent.agents.discovery_controller import DiscoveryController
    from retroagent.run.retro import LLMClient, _build_environment

    cfg = get_config()
    llm_api_key = api_key or cfg.llm_api_key
    if not llm_api_key:
        raise ProjectError("No API key set; use --api-key or configure LLM_API_KEY")
    manifest = project.load_manifest()
    mode = PermissionMode(manifest.permission_mode)
    grant = PermissionGrant(mode=mode, scopes=set(manifest.permission_scopes or []))
    llm = LLMClient(
        model=model or cfg.llm_model,
        api_key=llm_api_key,
        base_url=base_url or cfg.llm_base_url,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        max_retries=cfg.llm_max_retries,
    )
    controller = DiscoveryController(
        llm,
        _build_environment(),
        ledger=project.ledger(),
        workspace=project.runs_dir / "current",
        permission_grant=grant,
        max_cycles=max_cycles,
        max_estimated_minutes=cfg.env_timeout / 60 * max_cycles,
        timeout=cfg.env_timeout,
    )
    project.set_status("running")
    result = controller.run(goal)
    status = {
        "AwaitingApproval": "awaiting_approval",
        "Submitted": "completed",
        "StepLimit": "step_limit",
    }.get(str(result.get("exit_status")), "active")
    project.set_status(status)
    return result


def _execute_project_action(project, action_type, action, *, explicitly_approved, **context):
    from retroagent.skill_runtime import execute_action
    return execute_action(project, action_type, action, explicitly_approved=explicitly_approved, **context)


def _emit_action(command, result, json_output):
    if not result.get("success", True):
        _fail(command, "E_ACTION_FAILED", "Action failed; inspect recorded evidence and result",
              json_output=json_output, details=result, exit_code=3)
    _emit(command, result, json_output=json_output)






def _contains_id(value: Any, identifier: str) -> bool:
    if isinstance(value, dict):
        return any(_contains_id(item, identifier) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_id(item, identifier) for item in value)
    return str(value) == identifier

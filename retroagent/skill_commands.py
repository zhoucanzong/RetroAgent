"""Host-driven research commands that never invoke an internal language model."""

import json
from pathlib import Path
import typer

from retroagent.project import ResearchProject, ProjectError, apply_batch, unresolved_approval
from retroagent.skill_runtime import finish_research, project_lock


research_app = typer.Typer(help="Record and resume host-Agent research without a second LLM loop.")


@research_app.command("withdraw")
def withdraw(reason: str = typer.Option(..., "--reason"),
             project_path: Path = typer.Option(Path("."), "--project"),
             json_output: bool = typer.Option(False, "--json")):
    """Withdraw the latest pending action as the host Agent, not as a user denial."""
    from retroagent.cli_foundation import _emit, _fail
    try:
        project = ResearchProject.discover(project_path)
        with project_lock(project):
            ledger = project.ledger()
            pending = unresolved_approval(ledger.events)
            if pending is None:
                raise ProjectError("No pending approval to withdraw")
            event = ledger.append("approval_withdrawn", {"reason": reason, "action": pending.action},
                                  actor="host_agent", causal_parent_id=pending.request_event_id)
            project.set_status("active")
        _emit("research withdraw", event.to_dict(), json_output=json_output)
    except (ValueError, OSError) as exc:
        _fail("research withdraw", "E_RESEARCH", str(exc), json_output=json_output)


@research_app.command("start")
def start(task: str = typer.Argument(...), project_path: Path = typer.Option(Path("."), "--project"),
          json_output: bool = typer.Option(False, "--json")):
    from retroagent.cli_foundation import _emit, _fail
    try:
        project = ResearchProject.discover(project_path)
        with project_lock(project):
            if project.ledger().replay().goal:
                raise ProjectError("A research goal exists; use research context to continue it")
            data = apply_batch(project, [{"operation": "set_goal", "payload": {"task": task}}])
        _emit("research start", data, json_output=json_output)
    except (ValueError, OSError) as exc:
        _fail("research start", "E_RESEARCH", str(exc), json_output=json_output)


@research_app.command("context")
def context(project_path: Path = typer.Option(Path("."), "--project"),
            json_output: bool = typer.Option(False, "--json")):
    from retroagent.cli_foundation import _emit, _fail
    try:
        project = ResearchProject.discover(project_path)
        state = project.ledger().replay()
        _emit("research context", {"state": state.to_observation(), "conclusion": state.conclusion,
                                  "pending_approval": project.status()["pending_approval"]}, json_output=json_output)
    except (ValueError, OSError) as exc:
        _fail("research context", "E_RESEARCH", str(exc), json_output=json_output)


@research_app.command("finish")
def finish(input_path: Path = typer.Argument(..., help="JSON with conclusion, evidence_ids, target_ids, and uncertainty."),
           project_path: Path = typer.Option(Path("."), "--project"),
           json_output: bool = typer.Option(False, "--json")):
    from retroagent.cli_foundation import _emit, _fail
    try:
        conclusion = json.loads(input_path.read_text())
        if not isinstance(conclusion, dict):
            raise ProjectError("Conclusion must be an object")
        data = finish_research(ResearchProject.discover(project_path), conclusion)
        _emit("research finish", data, json_output=json_output)
    except (ValueError, OSError) as exc:
        _fail("research finish", "E_RESEARCH", str(exc), json_output=json_output)

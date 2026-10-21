"""Deterministic execution services for a host Agent; no model queries or planning."""

from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from filelock import FileLock

from retroagent.project import ProjectError, ResearchProject, unresolved_approval
from retroagent.research_execution import PermissionGrant, PermissionMode, PolicyEngine, ResearchExecutor
from retroagent.research_state import ResearchLedger
from retroagent.research_results import ResearchResultRecorder


def project_lock(project: ResearchProject):
    if not hasattr(project, "_writer_lock"):
        project._writer_lock = FileLock(str(project.control_dir / "writer.lock"), timeout=30)
    return project._writer_lock


def _copy_ledger(ledger: ResearchLedger) -> ResearchLedger:
    copied = ResearchLedger()
    for event in ledger.events:
        copied.append_event(event)
    return copied


def _matching_grant(ledger, action):
    requests = {e.id: e for e in ledger.events if e.type == "approval_requested"}
    consumed = {e.causal_parent_id for e in ledger.events if e.type == "approval_consumed"}
    denied = {e.causal_parent_id for e in ledger.events if e.type in {"approval_denied", "approval_withdrawn"}}
    for event in reversed(ledger.events):
        request = requests.get(event.causal_parent_id)
        if (event.type == "approval_granted" and event.id not in consumed and request
                and request.id not in denied and request.payload.get("action") == action):
            return event, request
    return None


def execute_action(project: ResearchProject, action_type: str, action: dict[str, Any], *,
                   explicitly_approved: bool = False, env=None, target_ids=(),
                   research_question: str = "", rationale: str = "", request_id: str = "") -> dict[str, Any]:
    """Execute exactly one requested action and atomically assimilate its result."""
    project.require()
    with project_lock(project):
        ledger = project.ledger()
        policy_action = {"type": action_type, **action}
        if request_id:
            for event in ledger.events:
                if event.type == "execution_started" and event.payload.get("request_id") == request_id:
                    if event.payload.get("action") != policy_action or event.payload.get("target_ids", []) != list(target_ids):
                        raise ProjectError("request_id already belongs to a different action or targets")
                    finished = next((e for e in ledger.events if e.type == "action_recorded"
                                     and e.payload.get("request_id") == request_id), None)
                    if finished:
                        return {**finished.payload["response"], "replayed": True}
                    raise ProjectError("request_id has an interrupted/in-progress action; inspect events before retrying")
        manifest = project.load_manifest()
        authorization = PolicyEngine().authorize(policy_action, PermissionGrant(
            mode=PermissionMode(manifest.permission_mode), scopes=set(manifest.permission_scopes or []),
        ))
        decision = {"action_type": action_type, "action": action, "target_ids": list(target_ids),
                    "research_question": research_question, "rationale": rationale,
                    "estimated_cost": {"time_minutes": 0, "compute": "unspecified"}}
        cycle = max((e.cycle for e in ledger.events), default=0) + 1
        if not authorization.allowed:
            if not authorization.requires_approval:
                raise ProjectError(authorization.reason)
            approved = _matching_grant(ledger, policy_action)
            if not approved:
                request = ledger.append("approval_requested", {
                    "action": policy_action, "decision": decision,
                    "required_scope": authorization.required_scope, "reason": authorization.reason,
                }, actor="host_agent", cycle=cycle)
                if not explicitly_approved:
                    project.set_status("awaiting_approval")
                    raise ProjectError(f"Approval required for scope '{authorization.required_scope}'")
                grant = ledger.append("approval_granted", {"scope": authorization.required_scope,
                    "action": policy_action, "one_time": True}, actor="user", causal_parent_id=request.id)
                approved = grant, request
            grant, request = approved
            ledger.append("approval_consumed", {"request_event_id": request.id, "action": action,
                          "one_time": True}, actor="executor", cycle=cycle, causal_parent_id=grant.id)
        ledger.append("decision_made", decision, actor="host_agent", cycle=cycle)
        ledger.append("execution_started", {"action": policy_action, "request_id": request_id,
                      "target_ids": list(target_ids)}, actor="executor", cycle=cycle)
        owned_environment = False
        try:
            if action_type == "tool":
                if env is None:
                    from retroagent.run.retro import _build_environment
                    env = _build_environment()
                    owned_environment = True
                result = dict(env.execute(action))
            elif action_type in {"python", "command"}:
                executor = ResearchExecutor(project.runs_dir / "manual")
                result = (executor.execute_python(str(action.get("code", ""))) if action_type == "python"
                          else executor.execute_command(str(action.get("command", ""))))
            else:
                raise ValueError(f"Unsupported action type: {action_type}")
        except Exception as exc:
            result = {"returncode": -1, "output": json.dumps({"error": str(exc)}),
                      "exception_info": f"{type(exc).__name__}: {exc}"}
        finally:
            if owned_environment and hasattr(env, "close"):
                env.close()
        from retroagent import __version__
        packages = {}
        for name in ("rdkit", "numpy", "torch", "retrochimera"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                pass
        result["runtime"] = {"python": platform.python_version(), "retroagent": __version__,
                             "installed_packages": packages}
        scratch = _copy_ledger(ledger)
        try:
            scratch.append("execution_finished", {"action": policy_action, "result": result}, actor="executor", cycle=cycle)
            ResearchResultRecorder(scratch).assimilate(result, action, decision, cycle)
        except (KeyError, TypeError, ValueError) as exc:
            # Reject the whole derived event set, retaining the original tool result for audit.
            scratch = _copy_ledger(ledger)
            result = {"returncode": -1, "output": json.dumps({"error": f"Invalid tool result: {exc}"}),
                      "original_result": result}
            scratch.append("execution_finished", {"action": policy_action, "result": result}, actor="executor", cycle=cycle)
            ResearchResultRecorder(scratch).assimilate(result, action, decision, cycle)
        new_events = scratch.events[len(ledger.events):]
        evidence_ids = [e.payload["id"] for e in new_events if e.type == "evidence_added"]
        failed = result.get("returncode", 0) != 0 or any(e.type == "action_failed" for e in new_events)
        response = {"action": policy_action, "result": result, "evidence_id": evidence_ids[0] if evidence_ids else None,
                    "evidence_ids": evidence_ids, "success": not failed,
                    "event_ids": [e.id for e in new_events], "request_id": request_id}
        scratch.append("action_recorded", {"request_id": request_id, "response": response}, actor="executor", cycle=cycle)
        project.replace_events(scratch.events)
        project.set_status("active")
        return response


def finish_research(project: ResearchProject, conclusion: dict[str, Any]) -> dict[str, Any]:
    with project_lock(project):
        ledger = project.ledger()
        if unresolved_approval(ledger.events):
            raise ProjectError("Resolve or withdraw pending approvals before finishing")
        for field in ("target_ids", "evidence_ids", "satisfied_criteria"):
            if not isinstance(conclusion.get(field, []), list):
                raise ProjectError(f"{field} must be an array")
        error = ResearchResultRecorder(ledger).completion_error(conclusion)
        if error:
            raise ProjectError(error)
        event = ledger.append("research_completed", conclusion, actor="host_agent")
        project.set_status("completed")
        return event.to_dict()


class SkillSession:
    """A persistent, serial MCP session retaining the environment/model cache."""
    def __init__(self, env=None):
        self._env = env

    @property
    def environment(self):
        if self._env is None:
            from retroagent.run.retro import _build_environment
            self._env = _build_environment()
        return self._env

    def call(self, project: ResearchProject, name: str, parameters: dict, **context):
        return execute_action(project, "tool", {"tool": name, "parameters": parameters},
                              env=self.environment, **context)

    def close(self):
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

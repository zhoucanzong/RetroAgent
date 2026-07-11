"""Autonomous observe-decide-execute-replan loop for discovery mode."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from retroagent.research_execution import PermissionGrant, PolicyEngine, ResearchExecutor
from retroagent.research_feedback import EvidenceAssimilator, action_fingerprint
from retroagent.research_state import ResearchLedger, next_entity_id


_SYSTEM_PROMPT = """You are RetroAgent's autonomous chemistry research controller.
Choose exactly one next research action from the current state. Return one JSON object with:
research_question, action_type, action, target_ids, rationale, expected_observation,
falsification_condition, estimated_cost, stop, conclusion, evidence_ids, and satisfied_criteria.
Use action_type state, tool, python, command, or stop. State actions may propose_hypothesis
or set_subgoal; they organize research but do not count as evidence. Negative evidence must change the next action.
Do not repeat an unchanged failed action. A stop decision must cite recorded evidence ids.
"""


class DiscoveryController:
    def __init__(
        self,
        model: Any,
        env: Any,
        *,
        ledger: ResearchLedger | None = None,
        workspace: Path | str = "logs/research_runs/default",
        permission_grant: PermissionGrant | None = None,
        max_cycles: int = 30,
        max_estimated_minutes: float = 120,
        timeout: float = 120,
    ):
        self.model = model
        self.env = env
        self.ledger = ledger or ResearchLedger()
        self.permission_grant = permission_grant or PermissionGrant()
        self.max_cycles = max_cycles
        self.max_estimated_minutes = float(max_estimated_minutes)
        self.policy = PolicyEngine()
        self.executor = ResearchExecutor(workspace, timeout=timeout)
        self.assimilator = EvidenceAssimilator()

    def run(self, task: str) -> dict[str, Any]:
        if not self.ledger.replay().goal:
            self.ledger.append("goal_set", {"task": task}, actor="user", cycle=0)

        for cycle in range(1, self.max_cycles + 1):
            state = self.ledger.replay()
            response = self.model.query(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "cycle": cycle,
                                "state": state.to_observation(),
                                "available_tools": self._tools_spec(),
                                "instruction": "Choose one evidence-seeking action or submit an evidence-backed conclusion.",
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                    },
                ],
                tools=None,
            )
            try:
                decision = _parse_decision(response)
                _validate_decision(decision)
            except ValueError as exc:
                self.ledger.append(
                    "decision_rejected", {"reason": str(exc)}, actor="controller", cycle=cycle,
                )
                continue

            self.ledger.append("decision_made", decision, actor="agent", cycle=cycle)
            if decision["stop"] or decision["action_type"] == "stop":
                invalid_reason = self._completion_error(decision)
                if invalid_reason:
                    self.ledger.append(
                        "decision_rejected", {"reason": invalid_reason}, actor="controller", cycle=cycle,
                    )
                    continue
                completion = {
                    "conclusion": decision.get("conclusion", ""),
                    "evidence_ids": list(decision.get("evidence_ids", [])),
                    "target_ids": list(decision.get("target_ids", [])),
                    "satisfied_criteria": list(decision.get("satisfied_criteria", [])),
                }
                self.ledger.append("research_completed", completion, actor="agent", cycle=cycle)
                return self._result("Submitted", **completion)

            try:
                action = _execution_action(decision)
            except ValueError as exc:
                self.ledger.append(
                    "decision_rejected", {"reason": str(exc)}, actor="controller", cycle=cycle,
                )
                continue
            fingerprint = action_fingerprint(action)
            current_state = self.ledger.replay()
            if (
                fingerprint in current_state.failed_action_fingerprints
                and not _has_new_evidence_basis(decision, current_state, fingerprint)
            ):
                self.ledger.append(
                    "decision_rejected",
                    {"reason": "unchanged failed action", "fingerprint": fingerprint, "action": action},
                    actor="controller",
                    cycle=cycle,
                )
                continue

            policy_action = {"type": decision["action_type"], **action}
            estimated_minutes = float(decision["estimated_cost"]["time_minutes"])
            if current_state.estimated_minutes_used + estimated_minutes > self.max_estimated_minutes:
                self.ledger.append(
                    "decision_rejected",
                    {
                        "reason": (
                            f"estimated time budget exceeded: used={current_state.estimated_minutes_used}, "
                            f"requested={estimated_minutes}, limit={self.max_estimated_minutes}"
                        ),
                        "action": action,
                    },
                    actor="controller",
                    cycle=cycle,
                )
                continue
            authorization = self.policy.authorize(policy_action, self.permission_grant)
            if not authorization.allowed:
                if authorization.requires_approval:
                    self.ledger.append(
                        "approval_requested",
                        {
                            "action": policy_action,
                            "required_scope": authorization.required_scope,
                            "reason": authorization.reason,
                        },
                        actor="controller",
                        cycle=cycle,
                    )
                    return self._result(
                        "AwaitingApproval",
                        required_scope=authorization.required_scope,
                        pending_action=policy_action,
                    )
                self.ledger.append(
                    "decision_rejected",
                    {"reason": authorization.reason, "action": policy_action},
                    actor="controller",
                    cycle=cycle,
                )
                continue

            self.ledger.append(
                "budget_charged",
                {"time_minutes": estimated_minutes, "action": action},
                actor="controller",
                cycle=cycle,
            )
            self.ledger.append(
                "execution_started",
                {"action": action, "fingerprint": fingerprint},
                actor="controller",
                cycle=cycle,
            )
            try:
                result = self._execute(decision["action_type"], action)
                if not isinstance(result, dict):
                    raise TypeError(f"executor returned {type(result).__name__}, expected dict")
            except Exception as exc:
                result = {
                    "status": "failed",
                    "returncode": -1,
                    "output": json.dumps({"error": str(exc)}),
                    "exception_info": f"{type(exc).__name__}: {exc}",
                }
            self.ledger.append(
                "execution_finished",
                {"action": action, "fingerprint": fingerprint, "result": result},
                actor="executor",
                cycle=cycle,
            )
            self._assimilate(result, action, decision, cycle)

        return self._result("StepLimit", error=f"Discovery cycle limit reached: {self.max_cycles}")

    def _execute(self, action_type: str, action: dict[str, Any]) -> dict[str, Any]:
        if action_type == "tool":
            return dict(self.env.execute(action))
        if action_type == "state":
            return _state_action_result(action)
        if action_type == "python":
            return self.executor.execute_python(str(action.get("code", "")))
        if action_type == "command":
            return self.executor.execute_command(str(action.get("command", "")))
        return {"status": "failed", "returncode": -1, "exception_info": "unsupported action type"}

    def _assimilate(
        self,
        result: dict[str, Any],
        action: dict[str, Any],
        decision: dict[str, Any],
        cycle: int,
    ) -> None:
        data = _result_data(result)
        evidence_targets = {str(item) for item in decision.get("target_ids", [])}
        failure = data.get("failure") if isinstance(data, dict) else None
        if isinstance(failure, dict) and failure.get("reaction_ids"):
            self.assimilator.record_failure(
                self.ledger,
                reaction_ids=[str(item) for item in failure["reaction_ids"]],
                reason=str(failure.get("reason", "computational feedback rejected the reaction")),
                category=str(failure.get("category", "feasibility")),
                repairable=bool(failure.get("repairable", False)),
                action=action,
                cycle=cycle,
                source=str(action.get("tool", decision["action_type"])),
            )
            return

        if isinstance(data, dict):
            collections = {}
            malformed_collection = ""
            for field_name in ("reactions", "routes"):
                value = data.get(field_name, [])
                if not isinstance(value, list):
                    malformed_collection = field_name
                    break
                collections[field_name] = value
            if malformed_collection:
                reason = f"tool output field '{malformed_collection}' must be an array"
                result = {
                    "status": "failed",
                    "returncode": -1,
                    "output": json.dumps({"error": reason}),
                    "exception_info": reason,
                }
                data = {"error": reason}
                collections = {"reactions": [], "routes": []}
            state_event = data.get("state_event")
            if isinstance(state_event, dict):
                allowed_state_events = {"hypothesis_proposed", "subgoal_set"}
                if (
                    decision["action_type"] == "state"
                    and state_event.get("type") in allowed_state_events
                    and isinstance(state_event.get("payload"), dict)
                ):
                    self.ledger.append(
                        str(state_event["type"]),
                        dict(state_event["payload"]),
                        actor="agent",
                        cycle=cycle,
                    )
                    return
                result = {
                    "status": "failed",
                    "returncode": -1,
                    "output": json.dumps({"error": "malformed state_event"}),
                    "exception_info": "malformed state_event",
                }
                data = {"error": "malformed state_event"}
            for reaction in collections.get("reactions", []):
                if isinstance(reaction, dict) and reaction.get("id"):
                    evidence_targets.add(str(reaction["id"]))
                    self.ledger.append(
                        "reaction_registered",
                        {"status": "active", **reaction},
                        actor="assimilator",
                        cycle=cycle,
                    )
            for route in collections.get("routes", []):
                if not isinstance(route, dict):
                    continue
                route_id = route.get("id") or route.get("route_id")
                if route_id:
                    evidence_targets.add(str(route_id))
                    normalized_route = {**route, "id": str(route_id)}
                    normalized_route.pop("route_id", None)
                    normalized_route.setdefault("status", "active")
                    normalized_route.setdefault("reaction_ids", route.get("steps", []))
                    score = route.get("score")
                    if isinstance(score, (int, float)):
                        normalized_route.setdefault("scores", {
                            "feasibility": round(float(score), 3),
                            "uncertainty": round(1.0 - float(score), 3),
                        })
                    self.ledger.append(
                        "route_registered", normalized_route, actor="assimilator", cycle=cycle,
                    )

        state = self.ledger.replay()
        evidence_id = next_entity_id(state.evidence, "evidence")
        summary = _result_summary(data, result)
        evidence_event = self.ledger.append(
            "evidence_added",
            {
                "id": evidence_id,
                "direction": "observes",
                "summary": summary,
                "source": str(action.get("tool", decision["action_type"])),
                "confidence": 0.6 if result.get("returncode", 0) == 0 else 0.2,
                "target_ids": sorted(evidence_targets),
                "result": data,
            },
            actor="assimilator",
            cycle=cycle,
        )
        failed = result.get("returncode", 0) != 0 or result.get("status") in {"failed", "timed_out"}
        if failed:
            fingerprint = action_fingerprint(action)
            reason = str(result.get("exception_info") or summary or "execution failed")
            self.ledger.append(
                "action_failed",
                {
                    "fingerprint": fingerprint,
                    "action": action,
                    "reason": reason,
                    "evidence_id": evidence_id,
                    "target_ids": sorted(evidence_targets),
                },
                actor="assimilator",
                cycle=cycle,
                causal_parent_id=evidence_event.id,
            )
            self.ledger.append(
                "replan_required",
                {"reason": reason, "evidence_id": evidence_id, "affected_route_ids": list(decision.get("target_ids", []))},
                actor="assimilator",
                cycle=cycle,
                causal_parent_id=evidence_event.id,
            )

    def _completion_error(self, decision: dict[str, Any]) -> str:
        if not str(decision.get("conclusion", "")).strip():
            return "completion requires a conclusion"
        evidence_ids = [str(item) for item in decision.get("evidence_ids", [])]
        if not evidence_ids:
            return "completion requires evidence ids"
        state = self.ledger.replay()
        required_criteria = {str(item) for item in state.goal.get("stopping_criteria", [])}
        satisfied_criteria = {str(item) for item in decision.get("satisfied_criteria", [])}
        missing_criteria = sorted(required_criteria - satisfied_criteria)
        if missing_criteria:
            return f"completion has unsatisfied stopping criteria: {', '.join(missing_criteria)}"
        missing = [item for item in evidence_ids if item not in state.evidence]
        if missing:
            return f"completion cites unknown evidence ids: {', '.join(missing)}"
        failed_targets = [
            target_id for target_id in decision.get("target_ids", [])
            if state.routes.get(str(target_id), {}).get("status") == "failed"
        ]
        if failed_targets:
            return f"completion selects failed routes: {', '.join(failed_targets)}"
        selected_targets = {str(item) for item in decision.get("target_ids", [])}
        relevant = False
        for evidence_id in evidence_ids:
            evidence = state.evidence[evidence_id]
            evidence_targets = {str(item) for item in evidence.get("target_ids", [])}
            if evidence_targets.intersection(selected_targets):
                relevant = True
                break
            if evidence.get("direction") == "opposes" and any(
                state.routes.get(target_id, {}).get("status") == "failed"
                or state.reactions.get(target_id, {}).get("status") == "failed"
                for target_id in evidence_targets
            ):
                relevant = True
                break
        if not relevant:
            return "completion requires evidence relevant to selected or failed competing routes"
        return ""

    def _tools_spec(self) -> list[dict[str, Any]]:
        specs = list(self.env.get_tools_spec()) if hasattr(self.env, "get_tools_spec") else []
        specs.extend([
            {"name": "run_python", "description": "Run temporary Python analysis in the research workspace", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
            {"name": "run_command", "description": "Run a system command in the research workspace", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
        ])
        return specs

    def _result(self, exit_status: str, **extra: Any) -> dict[str, Any]:
        state = self.ledger.replay()
        return {
            "exit_status": exit_status,
            "state": state.to_observation(),
            "events": list(self.ledger.events),
            "trajectory": [event.to_dict() for event in self.ledger.events],
            **extra,
        }


def _parse_decision(response: dict[str, Any]) -> dict[str, Any]:
    decision = (response.get("extra", {}) or {}).get("decision")
    if isinstance(decision, dict):
        return dict(decision)
    content = response.get("content", "")
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError) as exc:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", str(content), re.DOTALL)
        if not match:
            raise ValueError("decision must be one JSON object") from exc
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError as fenced_exc:
            raise ValueError("decision must be one JSON object") from fenced_exc
    if not isinstance(parsed, dict):
        raise ValueError("decision must be one JSON object")
    return parsed


def _validate_decision(decision: dict[str, Any]) -> None:
    required = {
        "research_question", "action_type", "action", "target_ids", "rationale",
        "expected_observation", "falsification_condition", "estimated_cost", "stop",
    }
    missing = sorted(required - set(decision))
    if missing:
        raise ValueError(f"decision missing fields: {', '.join(missing)}")
    if decision["action_type"] not in {"state", "tool", "python", "command", "stop"}:
        raise ValueError(f"unsupported action_type: {decision['action_type']}")
    if not isinstance(decision["action"], dict) or not isinstance(decision["target_ids"], list):
        raise ValueError("decision action must be an object and target_ids must be a list")
    estimated_cost = decision["estimated_cost"]
    if not isinstance(estimated_cost, dict):
        raise ValueError("estimated_cost must be an object")
    time_minutes = estimated_cost.get("time_minutes")
    if not isinstance(time_minutes, (int, float)) or isinstance(time_minutes, bool) or time_minutes < 0:
        raise ValueError("estimated_cost.time_minutes must be a non-negative number")
    if not math.isfinite(float(time_minutes)):
        raise ValueError("estimated_cost.time_minutes must be finite")
    if not isinstance(estimated_cost.get("compute"), str):
        raise ValueError("estimated_cost.compute must be a string")


def _execution_action(decision: dict[str, Any]) -> dict[str, Any]:
    action = dict(decision["action"])
    if decision["action_type"] == "tool":
        if not action.get("tool"):
            raise ValueError("tool action requires tool name")
        action.setdefault("parameters", {})
        if not isinstance(action["parameters"], dict):
            raise ValueError("tool action parameters must be an object")
    return action


def _result_data(result: dict[str, Any]) -> Any:
    output = result.get("output")
    if isinstance(output, str):
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return {"raw_output": output}
    if output is not None:
        return output
    return {
        key: value for key, value in result.items()
        if key not in {"command", "cwd", "script_path"}
    }


def _result_summary(data: Any, result: dict[str, Any]) -> str:
    if isinstance(data, dict):
        for key in ("summary", "message", "error", "exception_info"):
            if data.get(key):
                return str(data[key])
    if result.get("exception_info"):
        return str(result["exception_info"])
    encoded = json.dumps(data, ensure_ascii=True, sort_keys=True)
    return encoded[:500]


def _has_new_evidence_basis(decision: dict[str, Any], state: Any, fingerprint: str) -> bool:
    candidate_ids = {str(item) for item in decision.get("new_evidence_ids", [])}
    known_ids = candidate_ids.intersection(state.evidence)
    prior_failure_ids = state.failed_action_evidence.get(fingerprint, set())
    failure_index = state.failed_action_event_index.get(fingerprint, -1)
    failed_targets = state.failed_action_targets.get(fingerprint, set())
    if not failed_targets:
        return False
    for evidence_id in known_ids - prior_failure_ids:
        if state.evidence_event_index.get(evidence_id, -1) <= failure_index:
            continue
        evidence_targets = {str(item) for item in state.evidence[evidence_id].get("target_ids", [])}
        if evidence_targets.intersection(failed_targets):
            return True
    return False


def _state_action_result(action: dict[str, Any]) -> dict[str, Any]:
    operation = action.get("operation")
    if operation == "propose_hypothesis":
        hypothesis = action.get("hypothesis", {})
        required = {"id", "claim", "falsification_condition"}
        missing = sorted(required - set(hypothesis)) if isinstance(hypothesis, dict) else sorted(required)
        if missing:
            return {
                "returncode": -1,
                "output": json.dumps({"error": f"hypothesis missing fields: {', '.join(missing)}"}),
                "exception_info": "invalid state action",
            }
        return {
            "returncode": 0,
            "output": json.dumps({
                "state_event": {"type": "hypothesis_proposed", "payload": hypothesis},
            }),
            "exception_info": "",
        }
    if operation == "set_subgoal":
        subgoal = action.get("subgoal", {})
        if not isinstance(subgoal, dict) or not subgoal.get("objective"):
            return {
                "returncode": -1,
                "output": json.dumps({"error": "subgoal objective is required"}),
                "exception_info": "invalid state action",
            }
        return {
            "returncode": 0,
            "output": json.dumps({"state_event": {"type": "subgoal_set", "payload": subgoal}}),
            "exception_info": "",
        }
    return {
        "returncode": -1,
        "output": json.dumps({"error": f"unsupported state operation: {operation}"}),
        "exception_info": "invalid state action",
    }

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
Choose exactly one next research action from the current state and return only one JSON object.
Every decision must contain research_question, action_type, action, target_ids, rationale,
expected_observation, falsification_condition, estimated_cost, stop, conclusion, evidence_ids,
and satisfied_criteria. estimated_cost must be an object with numeric time_minutes and string compute.

Tool example:
{"research_question":"What evidence discriminates the hypotheses?","action_type":"tool","action":{"tool":"think","parameters":{"thought":"Compare the competing mechanisms."}},"target_ids":[],"rationale":"Reason before choosing a measurement.","expected_observation":"A discriminating prediction.","falsification_condition":"The hypotheses remain indistinguishable.","estimated_cost":{"time_minutes":0.1,"compute":"low"},"stop":false,"conclusion":"","evidence_ids":[],"satisfied_criteria":[]}

State example:
{"research_question":"Which hypothesis should be tracked?","action_type":"state","action":{"operation":"propose_hypothesis","hypothesis":{"id":"hyp_1","claim":"A testable chemical claim.","falsification_condition":"A specific contrary observation."}},"target_ids":[],"rationale":"Track a competing explanation.","expected_observation":"The hypothesis is recorded.","falsification_condition":"The hypothesis cannot be stated testably.","estimated_cost":{"time_minutes":0.1,"compute":"low"},"stop":false,"conclusion":"","evidence_ids":[],"satisfied_criteria":[]}

Stop example:
{"research_question":"Are the stopping criteria met?","action_type":"stop","action":{},"target_ids":["hyp_1"],"rationale":"Recorded evidence supports a conclusion.","expected_observation":"No further action is required.","falsification_condition":"New evidence overturns the conclusion.","estimated_cost":{"time_minutes":0,"compute":"none"},"stop":true,"conclusion":"Decisive evidence-backed answer.","evidence_ids":["evidence_0001"],"satisfied_criteria":[]}

Never return action as a string or estimated_cost as a number. Use action_type state, tool, python,
command, or stop. State actions may propose_hypothesis or set_subgoal; they organize research but do
not count as evidence. Negative evidence must change the next action. Do not repeat an unchanged
failed action. Recent schema or policy failures appear in state.recent_rejections; do not repeat a
rejected action while the reason and permission state are unchanged. A stop decision must cite
recorded evidence ids. For discovery_v3 goals, keep at least two competing hypotheses, rank
candidate experiments with select_experiment before measuring, update probabilities with
update_hypotheses after an observation, combine genuinely different route generators with
build_route_portfolio, and use validate_route_portfolio as an independent gate before selecting
a route. Novelty is a scored uncertainty, never proof of feasibility.
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
        allow_execution_tools: bool = True,
    ):
        self.model = model
        self.env = env
        self.ledger = ledger or ResearchLedger()
        self.permission_grant = permission_grant or PermissionGrant()
        self.max_cycles = max_cycles
        self.max_estimated_minutes = float(max_estimated_minutes)
        self.allow_execution_tools = bool(allow_execution_tools)
        self.policy = PolicyEngine()
        self.executor = ResearchExecutor(workspace, timeout=timeout)
        self.assimilator = EvidenceAssimilator()

    def run(self, task: str) -> dict[str, Any]:
        if not self.ledger.replay().goal:
            self.ledger.append("goal_set", {"task": task}, actor="user", cycle=0)

        self._consume_approved_action()
        base_cycle = max((event.cycle for event in self.ledger.events), default=0)

        for cycle in range(base_cycle + 1, base_cycle + self.max_cycles + 1):
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
            decision: dict[str, Any] | None = None
            try:
                decision = _parse_decision(response)
                _validate_decision(decision)
            except ValueError as exc:
                rejection = {
                    "reason": str(exc),
                    "model_response": str(response.get("content", ""))[:4000],
                }
                if decision is not None:
                    rejection["decision"] = decision
                self.ledger.append(
                    "decision_rejected", rejection, actor="controller", cycle=cycle,
                )
                continue

            decision_event = self.ledger.append("decision_made", decision, actor="agent", cycle=cycle)
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
                            "decision": decision,
                            "required_scope": authorization.required_scope,
                            "reason": authorization.reason,
                        },
                        actor="controller",
                        cycle=cycle,
                        causal_parent_id=decision_event.id,
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

    def _consume_approved_action(self) -> None:
        requests = {event.id: event for event in self.ledger.events if event.type == "approval_requested"}
        consumed = {
            event.causal_parent_id
            for event in self.ledger.events
            if event.type == "approval_consumed" and event.causal_parent_id
        }
        denied_requests = {
            event.causal_parent_id
            for event in self.ledger.events
            if event.type == "approval_denied" and event.causal_parent_id
        }
        approved = [
            event for event in self.ledger.events
            if event.type == "approval_granted"
            and event.id not in consumed
            and event.causal_parent_id in requests
            and event.causal_parent_id not in denied_requests
        ]
        if not approved:
            return
        grant_event = approved[-1]
        request_event = requests[grant_event.causal_parent_id]
        decision = request_event.payload.get("decision")
        if not isinstance(decision, dict):
            self.ledger.append(
                "decision_rejected",
                {"reason": "approved request has no resumable decision"},
                actor="controller",
                causal_parent_id=grant_event.id,
            )
            return
        try:
            action = _execution_action(decision)
            action_type = str(decision["action_type"])
        except (KeyError, TypeError, ValueError) as exc:
            self.ledger.append(
                "decision_rejected",
                {"reason": f"approved request is invalid: {exc}", "decision": decision},
                actor="controller",
                causal_parent_id=grant_event.id,
            )
            return
        cycle = max((event.cycle for event in self.ledger.events), default=0) + 1
        self.ledger.append(
            "approval_consumed",
            {"request_event_id": request_event.id, "action": action, "one_time": True},
            actor="controller",
            cycle=cycle,
            causal_parent_id=grant_event.id,
        )
        estimated = decision.get("estimated_cost", {})
        estimated_minutes = float(estimated.get("time_minutes", 0) or 0)
        self.ledger.append(
            "budget_charged",
            {"time_minutes": estimated_minutes, "action": action},
            actor="controller",
            cycle=cycle,
        )
        self.ledger.append(
            "execution_started",
            {"action": action, "approved_request_id": request_event.id},
            actor="controller",
            cycle=cycle,
        )
        try:
            result = self._execute(action_type, action)
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
            {"action": action, "result": result, "approved_request_id": request_event.id},
            actor="executor",
            cycle=cycle,
        )
        self._assimilate(result, action, decision, cycle)

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
        tool_name = str(action.get("tool", "")) if decision["action_type"] == "tool" else ""
        if isinstance(data, dict) and result.get("returncode", 0) == 0:
            if tool_name == "select_experiment":
                selected = data.get("selected_experiment")
                if isinstance(selected, dict) and selected.get("id"):
                    self.ledger.append(
                        "experiment_selected",
                        {**selected, "id": str(selected["id"]), "evidence_id": evidence_id},
                        actor="assimilator",
                        cycle=cycle,
                        causal_parent_id=evidence_event.id,
                    )
            elif tool_name == "update_hypotheses":
                for update in data.get("hypothesis_updates", []):
                    if isinstance(update, dict) and update.get("id"):
                        self.ledger.append(
                            "hypothesis_posterior_updated",
                            {
                                **update,
                                "id": str(update["id"]),
                                "posterior_entropy": data.get("posterior_entropy"),
                                "entropy_reduction": data.get("entropy_reduction"),
                                "evidence_id": evidence_id,
                            },
                            actor="assimilator",
                            cycle=cycle,
                            causal_parent_id=evidence_event.id,
                        )
            elif tool_name == "build_route_portfolio":
                portfolio = data.get("route_portfolio")
                if isinstance(portfolio, dict):
                    self.ledger.append(
                        "route_portfolio_built",
                        {**portfolio, "evidence_id": evidence_id},
                        actor="assimilator",
                        cycle=cycle,
                        causal_parent_id=evidence_event.id,
                    )
            elif tool_name == "validate_route_portfolio":
                for report in data.get("validation_reports", []):
                    if isinstance(report, dict) and report.get("route_id"):
                        self.ledger.append(
                            "route_validated",
                            {
                                "id": str(report["route_id"]),
                                "report_id": str(report.get("id", "")),
                                "status": str(report.get("status", "needs-evidence")),
                                "confidence": float(report.get("confidence", 0.0) or 0.0),
                                "checks": list(report.get("checks", [])),
                                "issues": list(report.get("issues", [])),
                                "evidence_id": evidence_id,
                            },
                            actor="assimilator",
                            cycle=cycle,
                            causal_parent_id=evidence_event.id,
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
        missing_criteria = sorted(
            criterion for criterion in required_criteria
            if not any(
                reported == criterion or reported.startswith(f"{criterion}:")
                for reported in satisfied_criteria
            )
        )
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
        discovery_v3 = state.goal.get("discovery_v3")
        if isinstance(discovery_v3, dict) and discovery_v3:
            min_hypotheses = int(discovery_v3.get("min_competing_hypotheses", 2))
            if len(state.hypotheses) < min_hypotheses:
                return f"discovery_v3 requires at least {min_hypotheses} competing hypotheses"
            if discovery_v3.get("require_active_experiment_selection", True) and not state.experiment_selections:
                return "discovery_v3 requires information-gain experiment selection"
            if discovery_v3.get("require_posterior_update", True) and not state.hypothesis_posterior_updates:
                return "discovery_v3 requires an evidence-driven posterior update"
            max_entropy = discovery_v3.get("max_posterior_entropy")
            if max_entropy is not None and state.hypothesis_posterior_updates:
                entropy = state.hypothesis_posterior_updates[-1].get("posterior_entropy")
                if entropy is None or float(entropy) > float(max_entropy):
                    return f"discovery_v3 posterior entropy exceeds {float(max_entropy):g}"
            selected_routes = [state.routes[item] for item in selected_targets if item in state.routes]
            if not selected_routes:
                return "discovery_v3 completion must select at least one recorded route"
            if discovery_v3.get("require_multi_source_route", True) and not any(
                int(route.get("source_count", len(route.get("sources", []))) or 0) >= 2
                for route in selected_routes
            ):
                return "discovery_v3 requires a route supported by multiple generator sources"
            min_novelty = float(discovery_v3.get("min_route_novelty", 0.0) or 0.0)
            if not any(float(route.get("novelty_score", 0.0) or 0.0) >= min_novelty for route in selected_routes):
                return f"discovery_v3 selected route novelty is below {min_novelty:g}"
            if discovery_v3.get("require_independent_validation", True) and not any(
                route.get("validation_status") == "passed" for route in selected_routes
            ):
                return "discovery_v3 requires an independently validated selected route"
        relevant = bool(
            not state.routes
            and any(not state.evidence[evidence_id].get("target_ids") for evidence_id in evidence_ids)
        )
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
        if self.allow_execution_tools:
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
        return _normalize_decision(decision)
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
    return _normalize_decision(parsed)


def _normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Normalize common model shorthands without weakening semantic validation."""
    normalized = dict(decision)
    action = normalized.get("action")
    if normalized.get("action_type") == "tool" and isinstance(action, str):
        normalized["action"] = {"tool": action, "parameters": {}}
        action = normalized["action"]
    if normalized.get("action_type") == "tool" and isinstance(action, dict):
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        if action.get("tool") == "run_python":
            normalized["action_type"] = "python"
            normalized["action"] = {"code": str(parameters.get("code", ""))}
        elif action.get("tool") == "run_command":
            normalized["action_type"] = "command"
            normalized["action"] = {"command": str(parameters.get("command", ""))}
    estimated_cost = normalized.get("estimated_cost")
    if isinstance(estimated_cost, (int, float)) and not isinstance(estimated_cost, bool):
        normalized["estimated_cost"] = {
            "time_minutes": estimated_cost,
            "compute": "unspecified",
        }
    return normalized


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

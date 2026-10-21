"""Shared evidence assimilation and completion checks, independent of any LLM loop."""

from __future__ import annotations

import json
from typing import Any
from retroagent.research_feedback import EvidenceAssimilator, action_fingerprint
from retroagent.research_state import ResearchLedger, next_entity_id


class ResearchResultRecorder:
    def __init__(self, ledger: ResearchLedger):
        self.ledger = ledger
        self.assimilator = EvidenceAssimilator()

    def assimilate(
        self,
        result: dict[str, Any],
        action: dict[str, Any],
        decision: dict[str, Any],
        cycle: int,
    ) -> None:
        data = _result_data(result)
        if isinstance(data, dict) and data.get("error") and result.get("returncode", 0) == 0:
            result["returncode"] = -1
            result["exception_info"] = str(data["error"])
        evidence_targets = {str(item) for item in decision.get("target_ids", [])}
        failure = data.get("failure") if isinstance(data, dict) else None
        if (isinstance(failure, dict) and failure.get("reaction_ids")
                and result.get("returncode", 0) == 0 and result.get("status") not in {"failed", "timed_out"}):
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

        if isinstance(data, dict) and result.get("returncode", 0) == 0 and result.get("status") not in {"failed", "timed_out"}:
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
                result.update({
                    "status": "failed",
                    "returncode": -1,
                    "output": json.dumps({"error": reason}),
                    "exception_info": reason,
                })
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
                result.update({
                    "status": "failed",
                    "returncode": -1,
                    "output": json.dumps({"error": "malformed state_event"}),
                    "exception_info": "malformed state_event",
                })
                data = {"error": "malformed state_event"}
                collections = {"reactions": [], "routes": []}
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
                "confidence": None,
                "confidence_basis": "Not calibrated; execution success is not chemical confidence",
                "execution_metadata": result.get("runtime", {}),
                "assessment_status": data.get("status", "observed") if isinstance(data, dict) else "observed",
                "target_ids": sorted(evidence_targets),
                "result": data,
            },
            actor="assimilator",
            cycle=cycle,
        )
        tool_name = str(action.get("tool", "")) if decision["action_type"] == "tool" else ""
        if isinstance(data, dict) and result.get("returncode", 0) == 0 and result.get("status") not in {"failed", "timed_out"}:
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

    def completion_error(self, decision: dict[str, Any]) -> str:
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

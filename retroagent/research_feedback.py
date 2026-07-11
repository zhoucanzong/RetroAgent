"""Convert computational feedback into research-state events."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from retroagent.research_state import ResearchEvent, ResearchLedger, next_entity_id


def action_fingerprint(action: dict[str, Any]) -> str:
    canonical = json.dumps(action, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


class EvidenceAssimilator:
    """Record negative evidence and propagate its impact to route beliefs."""

    def record_failure(
        self,
        ledger: ResearchLedger,
        *,
        reaction_ids: list[str],
        reason: str,
        category: str,
        repairable: bool,
        action: dict[str, Any] | None = None,
        cycle: int = 0,
        source: str = "computational-feedback",
    ) -> list[ResearchEvent]:
        state = ledger.replay()
        reaction_id_set = {str(item) for item in reaction_ids}
        affected_routes = [
            route for route in state.routes.values()
            if reaction_id_set.intersection(str(item) for item in route.get("reaction_ids", []))
        ]
        target_ids = sorted(reaction_id_set | {str(route["id"]) for route in affected_routes})
        events: list[ResearchEvent] = []

        evidence_id = next_entity_id(state.evidence, "evidence")
        events.append(ledger.append(
            "evidence_added",
            {
                "id": evidence_id,
                "direction": "opposes",
                "summary": reason,
                "category": category,
                "source": source,
                "confidence": 1.0,
                "repairable": repairable,
                "target_ids": target_ids,
            },
            actor="assimilator",
            cycle=cycle,
        ))

        for reaction_id in sorted(reaction_id_set):
            events.append(ledger.append(
                "reaction_status_changed",
                {
                    "id": reaction_id,
                    "status": "challenged" if repairable else "failed",
                    "failure_category": category,
                    "failure_reason": reason,
                    "evidence_ids": [evidence_id],
                },
                actor="assimilator",
                cycle=cycle,
                causal_parent_id=events[0].id,
            ))

        for route in affected_routes:
            scores = dict(route.get("scores", {}))
            if repairable:
                scores["feasibility"] = _clamp(float(scores.get("feasibility", 0.5)) - 0.3)
                scores["evidence_strength"] = _clamp(float(scores.get("evidence_strength", 0.5)) - 0.2)
                scores["uncertainty"] = _clamp(float(scores.get("uncertainty", 0.5)) + 0.3)
                status = "challenged"
            else:
                scores["feasibility"] = 0.0
                scores["evidence_strength"] = _clamp(float(scores.get("evidence_strength", 0.5)) - 0.4)
                scores["uncertainty"] = 1.0
                status = "failed"
            events.append(ledger.append(
                "route_status_changed",
                {
                    "id": route["id"],
                    "status": status,
                    "scores": scores,
                    "failure_reason": reason,
                    "evidence_ids": sorted(set(route.get("evidence_ids", [])) | {evidence_id}),
                },
                actor="assimilator",
                cycle=cycle,
                causal_parent_id=events[0].id,
            ))

        if action:
            events.append(ledger.append(
                "action_failed",
                {
                    "fingerprint": action_fingerprint(action),
                    "action": action,
                    "reason": reason,
                    "evidence_id": evidence_id,
                    "target_ids": target_ids,
                },
                actor="assimilator",
                cycle=cycle,
                causal_parent_id=events[0].id,
            ))

        events.append(ledger.append(
            "replan_required",
            {"reason": reason, "evidence_id": evidence_id, "affected_route_ids": [r["id"] for r in affected_routes]},
            actor="assimilator",
            cycle=cycle,
            causal_parent_id=events[0].id,
        ))
        return events


def _clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)

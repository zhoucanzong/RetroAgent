"""Replayable state primitives for autonomous discovery runs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchEvent:
    id: str
    type: str
    timestamp: str
    cycle: int
    actor: str
    payload: dict[str, Any]
    causal_parent_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchEvent":
        required = {"id", "type", "timestamp", "cycle", "actor", "payload"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Research event missing fields: {', '.join(missing)}")
        if not isinstance(data["payload"], dict):
            raise ValueError("Research event payload must be an object")
        return cls(
            id=str(data["id"]),
            type=str(data["type"]),
            timestamp=str(data["timestamp"]),
            cycle=int(data["cycle"]),
            actor=str(data["actor"]),
            payload=dict(data["payload"]),
            causal_parent_id=str(data.get("causal_parent_id", "")),
        )


@dataclass
class ResearchState:
    goal: dict[str, Any] = field(default_factory=dict)
    hypotheses: dict[str, dict[str, Any]] = field(default_factory=dict)
    routes: dict[str, dict[str, Any]] = field(default_factory=dict)
    reactions: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence_event_index: dict[str, int] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    subgoals: list[dict[str, Any]] = field(default_factory=list)
    executions: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    failed_action_fingerprints: set[str] = field(default_factory=set)
    failed_action_evidence: dict[str, set[str]] = field(default_factory=dict)
    failed_action_event_index: dict[str, int] = field(default_factory=dict)
    failed_action_targets: dict[str, set[str]] = field(default_factory=dict)
    replan_reasons: list[str] = field(default_factory=list)
    conclusion: dict[str, Any] = field(default_factory=dict)
    estimated_minutes_used: float = 0.0
    event_count: int = 0

    def apply(self, event: ResearchEvent) -> None:
        payload = dict(event.payload)
        event_type = event.type
        if event_type == "goal_set":
            self.goal = payload
        elif event_type == "hypothesis_proposed":
            self.hypotheses[str(payload["id"])] = payload
        elif event_type == "hypothesis_status_changed":
            hypothesis = self.hypotheses.setdefault(str(payload["id"]), {"id": payload["id"]})
            hypothesis.update(payload)
        elif event_type == "route_registered":
            route = dict(payload)
            route.setdefault("status", "proposed")
            route.setdefault("scores", {})
            existing = self.routes.get(str(route["id"]), {})
            if existing.get("status") in {"failed", "superseded", "completed"}:
                route = {**route, **existing}
            self.routes[str(route["id"])] = route
            for reaction_id in route.get("reaction_ids", []):
                self.reactions.setdefault(str(reaction_id), {"id": str(reaction_id), "status": "active"})
        elif event_type == "route_status_changed":
            route = self.routes.setdefault(str(payload["id"]), {"id": payload["id"]})
            route.update(payload)
        elif event_type == "reaction_status_changed":
            reaction = self.reactions.setdefault(str(payload["id"]), {"id": payload["id"]})
            reaction.update(payload)
        elif event_type == "reaction_registered":
            reaction_id = str(payload["id"])
            existing = self.reactions.get(reaction_id, {})
            reaction = {**existing, **payload}
            if existing.get("status") in {"failed", "superseded", "completed"}:
                reaction["status"] = existing["status"]
                for key in ("failure_category", "failure_reason", "evidence_ids"):
                    if key in existing:
                        reaction[key] = existing[key]
            self.reactions[reaction_id] = reaction
        elif event_type == "evidence_added":
            evidence_id = str(payload["id"])
            if evidence_id in self.evidence:
                raise ValueError(f"Duplicate evidence id: {evidence_id}")
            self.evidence[evidence_id] = payload
            self.evidence_event_index[evidence_id] = self.event_count
        elif event_type == "decision_made":
            self.decisions.append(payload)
        elif event_type == "decision_rejected":
            rejection = {"cycle": event.cycle, "reason": str(payload.get("reason", ""))}
            action = payload.get("action")
            if action is None and isinstance(payload.get("decision"), dict):
                action = payload["decision"].get("action")
            if action is not None:
                rejection["action"] = action
            self.rejections.append(rejection)
        elif event_type == "subgoal_set":
            self.subgoals.append(payload)
        elif event_type in ("execution_started", "execution_finished"):
            record = {"event_type": event_type, **payload}
            self.executions.append(record)
        elif event_type in ("approval_requested", "approval_granted", "approval_denied"):
            self.approvals.append({"event_type": event_type, **payload})
        elif event_type == "action_failed":
            fingerprint = str(payload.get("fingerprint", ""))
            if fingerprint:
                self.failed_action_fingerprints.add(fingerprint)
                self.failed_action_event_index[fingerprint] = self.event_count
                evidence_id = str(payload.get("evidence_id", ""))
                if evidence_id:
                    self.failed_action_evidence.setdefault(fingerprint, set()).add(evidence_id)
                self.failed_action_targets.setdefault(fingerprint, set()).update(
                    str(item) for item in payload.get("target_ids", [])
                )
        elif event_type == "replan_required":
            self.replan_reasons.append(str(payload.get("reason", "new evidence")))
        elif event_type == "research_completed":
            self.conclusion = payload
        elif event_type == "budget_charged":
            self.estimated_minutes_used += float(payload.get("time_minutes", 0.0) or 0.0)
        self.event_count += 1

    @property
    def replan_required(self) -> bool:
        return bool(self.replan_reasons)

    def to_observation(self) -> dict[str, Any]:
        hypotheses = list(self.hypotheses.values())
        routes = list(self.routes.values())
        reactions = list(self.reactions.values())
        return {
            "goal": dict(self.goal),
            "hypotheses": hypotheses[-30:],
            "hypothesis_count": len(hypotheses),
            "routes": routes[-30:],
            "route_count": len(routes),
            "reactions": reactions[-100:],
            "reaction_count": len(reactions),
            "recent_evidence": [_compact_evidence(item) for item in list(self.evidence.values())[-20:]],
            "recent_decisions": self.decisions[-10:],
            "recent_rejections": self.rejections[-5:],
            "subgoals": list(self.subgoals[-10:]),
            "recent_approvals": list(self.approvals[-10:]),
            "failed_action_fingerprints": sorted(self.failed_action_fingerprints),
            "replan_required": self.replan_required,
            "replan_reasons": list(self.replan_reasons[-10:]),
            "event_count": self.event_count,
            "estimated_minutes_used": round(self.estimated_minutes_used, 3),
        }


def _compact_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in evidence.items() if key != "result"}
    if "result" in evidence:
        result = evidence["result"]
        compact["result_chars"] = len(json.dumps(result, ensure_ascii=True, sort_keys=True))
        compact["result_keys"] = sorted(result) if isinstance(result, dict) else []
    return compact


def next_entity_id(existing_ids: Any, prefix: str, *, width: int = 4) -> str:
    existing = {str(item) for item in existing_ids}
    sequence = 1
    while f"{prefix}_{sequence:0{width}d}" in existing:
        sequence += 1
    return f"{prefix}_{sequence:0{width}d}"


class ResearchLedger:
    """Append-only research event store with deterministic replay."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self._events: list[ResearchEvent] = []
        self._event_ids: set[str] = set()
        if self.path and self.path.exists():
            self._load()

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        cycle: int = 0,
        causal_parent_id: str = "",
    ) -> ResearchEvent:
        sequence = 1
        while f"event_{sequence:06d}" in self._event_ids:
            sequence += 1
        event = ResearchEvent(
            id=f"event_{sequence:06d}",
            type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            cycle=cycle,
            actor=actor,
            payload=dict(payload),
            causal_parent_id=causal_parent_id,
        )
        return self.append_event(event)

    @property
    def events(self) -> tuple[ResearchEvent, ...]:
        return tuple(_clone_event(event) for event in self._events)

    def append_event(self, event: ResearchEvent) -> ResearchEvent:
        if event.id in self._event_ids:
            raise ValueError(f"Duplicate event id: {event.id}")
        stored = _clone_event(event)
        self._events.append(stored)
        self._event_ids.add(stored.id)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stored.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        return _clone_event(stored)

    def replay(self) -> ResearchState:
        state = ResearchState()
        for index, event in enumerate(self._events):
            try:
                state.apply(event)
            except Exception as exc:
                raise ValueError(
                    f"Invalid research event at event index {index} ({event.id}); "
                    f"last valid index {index - 1}: {exc}"
                ) from exc
        return state

    def _load(self) -> None:
        assert self.path is not None
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = ResearchEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid research ledger at {self.path}:{line_number}: {exc}") from exc
            if event.id in self._event_ids:
                raise ValueError(f"Duplicate event id: {event.id}")
            self._events.append(_clone_event(event))
            self._event_ids.add(event.id)


def _clone_event(event: ResearchEvent) -> ResearchEvent:
    return ResearchEvent(
        id=event.id,
        type=event.type,
        timestamp=event.timestamp,
        cycle=event.cycle,
        actor=event.actor,
        payload=deepcopy(event.payload),
        causal_parent_id=event.causal_parent_id,
    )

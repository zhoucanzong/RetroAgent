"""Stable resource addressing and structured project queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from retroagent.research_state import ResearchEvent, ResearchState


RESOURCE_COLLECTIONS = {
    "molecule": "molecules",
    "hypothesis": "hypotheses",
    "route": "routes",
    "reaction": "reactions",
    "evidence": "evidence",
    "experiment": "experiments",
}
RESOURCE_ALIASES = {
    "mol": "molecule",
    "hyp": "hypothesis",
    "rxn": "reaction",
    "ev": "evidence",
    "exp": "experiment",
}


@dataclass(frozen=True)
class ResourceRef:
    kind: str
    identifier: str

    @classmethod
    def parse(cls, value: str) -> "ResourceRef":
        if ":" not in value:
            raise ValueError("Resource must use kind:id syntax")
        raw_kind, identifier = value.split(":", 1)
        kind = RESOURCE_ALIASES.get(raw_kind.strip().lower(), raw_kind.strip().lower())
        identifier = identifier.strip()
        valid = set(RESOURCE_COLLECTIONS) | {"event", "goal", "project"}
        if kind not in valid:
            raise ValueError(f"Unknown resource kind: {kind}")
        if not identifier and kind not in {"goal", "project"}:
            raise ValueError("Resource id cannot be empty")
        return cls(kind, identifier)

    def __str__(self) -> str:
        return f"{self.kind}:{self.identifier}"


def resolve_resource(
    ref: ResourceRef | str,
    state: ResearchState,
    events: Iterable[ResearchEvent] = (),
) -> dict[str, Any]:
    parsed = ResourceRef.parse(ref) if isinstance(ref, str) else ref
    if parsed.kind == "goal":
        if not state.goal:
            raise KeyError("Goal has not been set")
        return dict(state.goal)
    if parsed.kind == "event":
        for event in events:
            if event.id == parsed.identifier:
                return event.to_dict()
        raise KeyError(f"Unknown event: {parsed.identifier}")
    if parsed.kind == "project":
        raise KeyError("Project resources are resolved by the project manifest")
    collection = getattr(state, RESOURCE_COLLECTIONS[parsed.kind])
    try:
        return dict(collection[parsed.identifier])
    except KeyError as exc:
        raise KeyError(f"Unknown {parsed.kind}: {parsed.identifier}") from exc


def query_resources(
    kind: str,
    state: ResearchState,
    *,
    filters: Iterable[str] = (),
    contains: str = "",
) -> list[dict[str, Any]]:
    normalized = RESOURCE_ALIASES.get(kind.lower(), kind.lower())
    if normalized not in RESOURCE_COLLECTIONS:
        raise ValueError(f"Unknown resource kind: {kind}")
    parsed_filters = [_parse_filter(item) for item in filters]
    values = list(getattr(state, RESOURCE_COLLECTIONS[normalized]).values())
    results = []
    for value in values:
        if not all(_matches_filter(value, *condition) for condition in parsed_filters):
            continue
        if contains and contains.lower() not in _flatten_text(value).lower():
            continue
        results.append(dict(value))
    return results


def _parse_filter(value: str) -> tuple[str, str, str]:
    for operator in ("!=", ">=", "<=", "~=", "=", ">", "<"):
        if operator in value:
            key, expected = value.split(operator, 1)
            if not key.strip():
                break
            return key.strip(), operator, expected.strip()
    raise ValueError(f"Invalid filter '{value}'; use field=value, field!=value, or field~=text")


def _matches_filter(data: dict[str, Any], key: str, operator: str, expected: str) -> bool:
    actual: Any = data
    for part in key.split("."):
        if not isinstance(actual, dict) or part not in actual:
            return operator == "!="
        actual = actual[part]
    if operator == "~=":
        return expected.lower() in _flatten_text(actual).lower()
    if operator in {">", "<", ">=", "<="}:
        try:
            left, right = float(actual), float(expected)
        except (TypeError, ValueError):
            return False
        return {">": left > right, "<": left < right, ">=": left >= right, "<=": left <= right}[operator]
    equal = _coerce_text(actual) == expected
    return not equal if operator == "!=" else equal


def _coerce_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    return str(value)


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return _coerce_text(value)

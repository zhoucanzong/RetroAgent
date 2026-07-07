"""ResearchMemoryTool: local JSONL memory for discovery runs.

The memory is intentionally simple and inspectable. It stores proposed routes,
critic summaries, experiment plans, and outcomes so future discovery runs can
retrieve prior work without a database service.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retroagent.schemas import ResearchMemoryRecord


DEFAULT_MEMORY_PATH = Path("logs/research_memory.jsonl")


class ResearchMemoryTool:
    name = "research_memory"
    description = (
        "Record, search, or summarize local research memory for chemical discovery. "
        "Use after build_research_report to persist route hypotheses, critic issues, "
        "experiment plans, outcomes, and user/lab feedback. Uses append-only JSONL; "
        "no external service."
    )

    def execute(self, parameters: dict) -> str:
        operation = parameters.get("operation", "record")
        memory_path = Path(parameters.get("memory_path") or DEFAULT_MEMORY_PATH)
        if operation == "record":
            record = build_memory_record(parameters)
            append_memory_record(memory_path, record)
            return json.dumps({
                "status": "recorded",
                "memory_path": str(memory_path),
                "record": record.to_dict(),
            }, ensure_ascii=False)
        if operation == "search":
            records = search_memory_records(
                memory_path,
                query=parameters.get("query", ""),
                target_smiles=parameters.get("target_smiles", ""),
                limit=int(parameters.get("limit", 5) or 5),
            )
            return json.dumps({
                "status": "ok",
                "memory_path": str(memory_path),
                "count": len(records),
                "records": [record.to_dict() for record in records],
            }, ensure_ascii=False)
        if operation == "summarize":
            records = search_memory_records(
                memory_path,
                query=parameters.get("query", ""),
                target_smiles=parameters.get("target_smiles", ""),
                limit=int(parameters.get("limit", 20) or 20),
            )
            return json.dumps({
                "status": "ok",
                "memory_path": str(memory_path),
                "summary": summarize_memory_records(records),
            }, ensure_ascii=False)
        return json.dumps({
            "error": f"Unknown operation: {operation}",
            "allowed_operations": ["record", "search", "summarize"],
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["record", "search", "summarize"],
                    "description": "Memory operation.",
                },
                "memory_path": {
                    "type": "string",
                    "description": "Optional JSONL path. Defaults to logs/research_memory.jsonl.",
                },
                "target_smiles": {"type": "string", "description": "Target molecule SMILES."},
                "target_name": {"type": "string", "description": "Optional target name."},
                "outcome": {
                    "type": "string",
                    "description": "Outcome label such as proposed, validated, failed, rejected, or needs-validation.",
                },
                "hypotheses": {
                    "type": "array",
                    "description": "Route hypotheses or build_research_report hypotheses.",
                    "items": {"type": "object"},
                },
                "critic_report": {"type": "object", "description": "Output from chemical_critic."},
                "experiment_plan": {"type": "object", "description": "Output from design_experiments."},
                "route_graph": {"type": "object", "description": "Serialized RouteGraph, if available."},
                "tags": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "string", "description": "User/lab notes or interpretation."},
                "query": {"type": "string", "description": "Search text."},
                "limit": {"type": "integer", "description": "Maximum records to return."},
            },
            "required": ["operation"],
        }


def build_memory_record(parameters: dict[str, Any]) -> ResearchMemoryRecord:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    critic_report = parameters.get("critic_report", {}) or {}
    experiment_plan = parameters.get("experiment_plan", {}) or {}
    route_graph = parameters.get("route_graph", {}) or {}
    hypotheses = parameters.get("hypotheses", []) or []
    record_id = parameters.get("id") or _stable_record_id(
        parameters.get("target_smiles", ""),
        parameters.get("outcome", "proposed"),
        hypotheses,
        now,
    )
    return ResearchMemoryRecord(
        id=record_id,
        target_smiles=parameters.get("target_smiles", ""),
        target_name=parameters.get("target_name", ""),
        outcome=parameters.get("outcome", "proposed"),
        hypotheses=hypotheses,
        critic_summary=_critic_summary(critic_report),
        experiment_plan=_experiment_plan_summary(experiment_plan),
        route_graph_summary=_route_graph_summary(route_graph),
        tags=list(parameters.get("tags", []) or []),
        notes=parameters.get("notes", ""),
        timestamp=now,
    )


def append_memory_record(path: Path, record: ResearchMemoryRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def load_memory_records(path: Path) -> list[ResearchMemoryRecord]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(ResearchMemoryRecord(
            id=data.get("id", ""),
            target_smiles=data.get("target_smiles", ""),
            target_name=data.get("target_name", ""),
            outcome=data.get("outcome", "proposed"),
            hypotheses=list(data.get("hypotheses", [])),
            critic_summary=dict(data.get("critic_summary", {})),
            experiment_plan=dict(data.get("experiment_plan", {})),
            route_graph_summary=dict(data.get("route_graph_summary", {})),
            tags=list(data.get("tags", [])),
            notes=data.get("notes", ""),
            timestamp=data.get("timestamp", ""),
        ))
    return records


def search_memory_records(
    path: Path,
    *,
    query: str = "",
    target_smiles: str = "",
    limit: int = 5,
) -> list[ResearchMemoryRecord]:
    records = load_memory_records(path)
    query_lower = query.lower().strip()
    target_lower = target_smiles.lower().strip()
    filtered = []
    for record in reversed(records):
        haystack = json.dumps(record.to_dict(), ensure_ascii=False).lower()
        if target_lower and target_lower != record.target_smiles.lower():
            continue
        if query_lower and query_lower not in haystack:
            continue
        filtered.append(record)
        if len(filtered) >= limit:
            break
    return filtered


def summarize_memory_records(records: list[ResearchMemoryRecord]) -> dict[str, Any]:
    outcome_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}
    targets = []
    for record in records:
        outcome_counts[record.outcome] = outcome_counts.get(record.outcome, 0) + 1
        targets.append(record.target_name or record.target_smiles)
        for category, count in record.critic_summary.get("issue_category_counts", {}).items():
            issue_counts[category] = issue_counts.get(category, 0) + int(count)
    return {
        "record_count": len(records),
        "targets": targets[:10],
        "outcome_counts": outcome_counts,
        "issue_category_counts": issue_counts,
        "latest_record_id": records[0].id if records else "",
    }


def _critic_summary(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {}) or {}
    return {
        "issue_count": report.get("issue_count", len(report.get("issues", []))),
        "critical_count": report.get("critical_count", 0),
        "warning_count": report.get("warning_count", 0),
        "info_count": report.get("info_count", 0),
        "needs_revision": report.get("needs_revision", False),
        "passed_checks": list(report.get("passed_checks", [])),
        "issue_category_counts": dict(metadata.get("issue_category_counts", {})),
    }


def _experiment_plan_summary(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": plan.get("objective", ""),
        "step_count": len(plan.get("steps", [])),
        "control_count": len(plan.get("controls", [])),
        "success_criteria": list(plan.get("success_criteria", [])),
        "failure_diagnostics": list(plan.get("failure_diagnostics", [])),
        "linked_issue_categories": list(plan.get("linked_issue_categories", [])),
        "ready_for_report": plan.get("ready_for_report", False),
    }


def _route_graph_summary(route_graph: dict[str, Any]) -> dict[str, Any]:
    stats = route_graph.get("stats", {}) or {}
    return {
        "target_id": stats.get("target_id", route_graph.get("target_id", "")),
        "molecule_count": stats.get("molecule_count", 0),
        "reaction_count": stats.get("reaction_count", 0),
        "unsolved_molecule_count": stats.get("unsolved_molecule_count", 0),
    }


def _stable_record_id(target_smiles: str, outcome: str, hypotheses: list[dict[str, Any]], timestamp: str) -> str:
    payload = json.dumps({
        "target_smiles": target_smiles,
        "outcome": outcome,
        "hypotheses": hypotheses,
        "timestamp": timestamp,
    }, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"mem_{digest}"

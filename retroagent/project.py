"""Persistent project workspace and deterministic research operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from retroagent.research_state import ResearchEvent, ResearchLedger, ResearchState, next_entity_id


PROJECT_SCHEMA_VERSION = "1.0"
PROJECT_DIR_NAME = ".retroagent"
MANIFEST_NAME = "project.json"
LEDGER_NAME = "ledger.jsonl"


class ProjectError(ValueError):
    """Raised when a project workspace or operation is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectManifest:
    schema_version: str
    name: str
    created_at: str
    updated_at: str
    status: str = "initialized"
    permission_mode: str = "standard"
    permission_scopes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["permission_scopes"] = sorted(self.permission_scopes or [])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectManifest":
        required = {"schema_version", "name", "created_at", "updated_at"}
        missing = sorted(required - set(data))
        if missing:
            raise ProjectError(f"Project manifest missing fields: {', '.join(missing)}")
        if data["schema_version"] != PROJECT_SCHEMA_VERSION:
            raise ProjectError(
                f"Unsupported project schema {data['schema_version']}; expected {PROJECT_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=str(data["schema_version"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            status=str(data.get("status", "initialized")),
            permission_mode=str(data.get("permission_mode", "standard")),
            permission_scopes=[str(item) for item in data.get("permission_scopes", [])],
        )


class ResearchProject:
    """A durable `.retroagent` workspace rooted in a user-owned directory."""

    def __init__(self, root: Path | str):
        root_path = Path(root).expanduser().resolve()
        self.root = root_path.parent if root_path.name == PROJECT_DIR_NAME else root_path
        self.control_dir = root_path if root_path.name == PROJECT_DIR_NAME else root_path / PROJECT_DIR_NAME
        self.manifest_path = self.control_dir / MANIFEST_NAME
        self.ledger_path = self.control_dir / LEDGER_NAME
        self.artifacts_dir = self.control_dir / "artifacts"
        self.cache_dir = self.control_dir / "cache"
        self.reports_dir = self.control_dir / "reports"
        self.views_dir = self.control_dir / "views"
        self.checkpoints_dir = self.control_dir / "checkpoints"
        self.runs_dir = self.control_dir / "runs"

    @classmethod
    def discover(cls, start: Path | str = ".") -> "ResearchProject":
        current = Path(start).expanduser().resolve()
        if current.is_file():
            current = current.parent
        if current.name == PROJECT_DIR_NAME and (current / MANIFEST_NAME).exists():
            return cls(current)
        for candidate in (current, *current.parents):
            if (candidate / PROJECT_DIR_NAME / MANIFEST_NAME).exists():
                return cls(candidate)
        raise ProjectError(f"No RetroAgent project found from {current}; run 'retroagent project init'")

    @classmethod
    def initialize(
        cls,
        root: Path | str,
        *,
        name: str = "",
        permission_mode: str = "standard",
        permission_scopes: Iterable[str] = (),
        force: bool = False,
    ) -> "ResearchProject":
        project = cls(root)
        if project.manifest_path.exists() and not force:
            raise ProjectError(f"RetroAgent project already exists at {project.control_dir}")
        if permission_mode not in {"restricted", "standard", "full"}:
            raise ProjectError("permission_mode must be restricted, standard, or full")
        scopes = sorted({str(scope).strip() for scope in permission_scopes if str(scope).strip()})
        if permission_mode == "full" and not scopes:
            raise ProjectError("full permission mode requires explicit permission scopes")
        project.control_dir.mkdir(parents=True, exist_ok=True)
        for directory in (
            project.artifacts_dir,
            project.cache_dir,
            project.reports_dir,
            project.views_dir,
            project.checkpoints_dir,
            project.runs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        manifest = ProjectManifest(
            schema_version=PROJECT_SCHEMA_VERSION,
            name=name.strip() or project.root.name,
            created_at=now,
            updated_at=now,
            permission_mode=permission_mode,
            permission_scopes=scopes,
        )
        project._write_json_atomic(project.manifest_path, manifest.to_dict())
        if not project.ledger_path.exists():
            project._write_text_atomic(project.ledger_path, "")
        return project

    def require(self) -> "ResearchProject":
        if not self.manifest_path.exists():
            raise ProjectError(f"Not a RetroAgent project: {self.root}")
        self.load_manifest()
        return self

    def load_manifest(self) -> ProjectManifest:
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectError(f"Missing project manifest: {self.manifest_path}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Invalid project manifest JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ProjectError("Project manifest must be a JSON object")
        return ProjectManifest.from_dict(data)

    def save_manifest(self, manifest: ProjectManifest) -> None:
        manifest.updated_at = utc_now()
        self._write_json_atomic(self.manifest_path, manifest.to_dict())

    def set_status(self, status: str) -> ProjectManifest:
        manifest = self.load_manifest()
        manifest.status = status
        self.save_manifest(manifest)
        return manifest

    def ledger(self) -> ResearchLedger:
        self.require()
        return ResearchLedger(self.ledger_path)

    def status(self) -> dict[str, Any]:
        manifest = self.load_manifest()
        ledger = self.ledger()
        state = ledger.replay()
        pending = unresolved_approval(ledger.events)
        return {
            "project": manifest.to_dict(),
            "root": str(self.root),
            "control_dir": str(self.control_dir),
            "ledger": str(self.ledger_path),
            "event_count": len(ledger.events),
            "goal": state.goal,
            "counts": {
                "molecules": len(state.molecules),
                "hypotheses": len(state.hypotheses),
                "routes": len(state.routes),
                "reactions": len(state.reactions),
                "evidence": len(state.evidence),
                "experiments": len(state.experiments),
            },
            "pending_approval": pending.to_dict() if pending else None,
            "conclusion": state.conclusion,
        }

    def replace_events(self, events: Iterable[ResearchEvent]) -> None:
        scratch = ResearchLedger()
        rows = []
        for event in events:
            scratch.append_event(event)
            rows.append(json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True))
        scratch.replay()
        self._write_text_atomic(self.ledger_path, "\n".join(rows) + ("\n" if rows else ""))

    def create_checkpoint(self, label: str = "") -> dict[str, Any]:
        ledger = self.ledger()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        slug = _slug(label) if label else "checkpoint"
        checkpoint_id = f"{timestamp}-{slug}"
        ledger_path = self.checkpoints_dir / f"{checkpoint_id}.jsonl"
        metadata_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        raw = self.ledger_path.read_bytes()
        self._write_bytes_atomic(ledger_path, raw)
        metadata = {
            "id": checkpoint_id,
            "label": label,
            "created_at": utc_now(),
            "event_count": len(ledger.events),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "ledger": ledger_path.name,
        }
        self._write_json_atomic(metadata_path, metadata)
        return metadata

    def list_checkpoints(self) -> list[dict[str, Any]]:
        checkpoints = []
        for path in sorted(self.checkpoints_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("id"):
                checkpoints.append(data)
        return checkpoints

    def checkpoint_ledger(self, checkpoint_id: str) -> ResearchLedger:
        path = self.checkpoints_dir / f"{checkpoint_id}.jsonl"
        if not path.exists():
            raise ProjectError(f"Unknown checkpoint: {checkpoint_id}")
        return ResearchLedger(path)

    def restore_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
        checkpoint = self.checkpoint_ledger(checkpoint_id)
        backup = self.create_checkpoint("before-restore")
        self.replace_events(checkpoint.events)
        self.set_status("restored")
        return {"restored": checkpoint_id, "backup": backup["id"], "event_count": len(checkpoint.events)}

    @staticmethod
    def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
        ResearchProject._write_text_atomic(
            path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )

    @staticmethod
    def _write_text_atomic(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ResearchProject._write_bytes_atomic(path, text.encode("utf-8"))

    @staticmethod
    def _write_bytes_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


@dataclass(frozen=True)
class PendingApproval:
    request_event_id: str
    scope: str
    action: dict[str, Any]
    decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unresolved_approval(events: Iterable[ResearchEvent]) -> PendingApproval | None:
    requests: dict[str, ResearchEvent] = {}
    resolved: set[str] = set()
    for event in events:
        if event.type == "approval_requested":
            requests[event.id] = event
        elif event.type in {"approval_granted", "approval_denied"} and event.causal_parent_id:
            resolved.add(event.causal_parent_id)
    for event_id, event in reversed(list(requests.items())):
        if event_id in resolved:
            continue
        return PendingApproval(
            request_event_id=event_id,
            scope=str(event.payload.get("required_scope", "")),
            action=dict(event.payload.get("action", {})),
            decision=dict(event.payload.get("decision", {})),
        )
    return None


def state_to_dict(state: ResearchState) -> dict[str, Any]:
    return {
        "goal": dict(state.goal),
        "molecules": list(state.molecules.values()),
        "hypotheses": list(state.hypotheses.values()),
        "routes": list(state.routes.values()),
        "reactions": list(state.reactions.values()),
        "evidence": list(state.evidence.values()),
        "experiments": list(state.experiments.values()),
        "experiment_selections": list(state.experiment_selections),
        "hypothesis_posterior_updates": list(state.hypothesis_posterior_updates),
        "route_portfolios": list(state.route_portfolios),
        "route_validations": list(state.route_validations.values()),
        "decisions": list(state.decisions),
        "rejections": list(state.rejections),
        "subgoals": list(state.subgoals),
        "approvals": list(state.approvals),
        "replan_required": state.replan_required,
        "replan_reasons": list(state.replan_reasons),
        "conclusion": dict(state.conclusion),
        "stopped": state.stopped,
        "estimated_minutes_used": state.estimated_minutes_used,
        "event_count": state.event_count,
    }


BATCH_OPERATIONS: dict[str, tuple[str, str, str]] = {
    "set_goal": ("goal_set", "goal", ""),
    "register_molecule": ("molecule_registered", "molecule", "mol"),
    "propose_hypothesis": ("hypothesis_proposed", "hypothesis", "hyp"),
    "set_hypothesis_status": ("hypothesis_status_changed", "hypothesis", ""),
    "register_route": ("route_registered", "route", "route"),
    "set_route_status": ("route_status_changed", "route", ""),
    "register_reaction": ("reaction_registered", "reaction", "rxn"),
    "set_reaction_status": ("reaction_status_changed", "reaction", ""),
    "add_evidence": ("evidence_added", "evidence", "evidence"),
    "register_experiment": ("experiment_registered", "experiment", "exp"),
    "set_experiment_status": ("experiment_status_changed", "experiment", ""),
    "set_subgoal": ("subgoal_set", "subgoal", ""),
}


def apply_batch(
    project: ResearchProject,
    actions: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    actor: str = "cli",
) -> dict[str, Any]:
    if not isinstance(actions, list) or not actions:
        raise ProjectError("Batch input must be a non-empty JSON array")
    original = project.ledger()
    scratch = ResearchLedger()
    for event in original.events:
        scratch.append_event(event)
    created: list[dict[str, Any]] = []
    for index, item in enumerate(actions):
        if not isinstance(item, dict):
            raise ProjectError(f"Batch item {index} must be an object")
        operation = str(item.get("operation", ""))
        if operation not in BATCH_OPERATIONS:
            raise ProjectError(f"Batch item {index} has unsupported operation: {operation}")
        event_type, payload_key, id_prefix = BATCH_OPERATIONS[operation]
        payload = item.get("payload", item.get(payload_key, {}))
        if not isinstance(payload, dict):
            raise ProjectError(f"Batch item {index} payload must be an object")
        payload = dict(payload)
        if id_prefix and not payload.get("id"):
            state = scratch.replay()
            collection = _state_collection(state, payload_key)
            payload["id"] = next_entity_id(collection, id_prefix)
        _validate_batch_payload(operation, payload, index)
        event = scratch.append(event_type, payload, actor=actor, cycle=int(item.get("cycle", 0)))
        scratch.replay()
        created.append(event.to_dict())
    if not dry_run:
        project.replace_events(scratch.events)
        project.set_status("active")
    return {
        "dry_run": dry_run,
        "atomic": True,
        "created_events": created,
        "event_count_before": len(original.events),
        "event_count_after": len(scratch.events),
    }


def diff_states(left: ResearchState, right: ResearchState) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("molecules", "hypotheses", "routes", "reactions", "evidence", "experiments"):
        before = getattr(left, name)
        after = getattr(right, name)
        before_ids = set(before)
        after_ids = set(after)
        changed = sorted(key for key in before_ids & after_ids if before[key] != after[key])
        result[name] = {
            "added": sorted(after_ids - before_ids),
            "removed": sorted(before_ids - after_ids),
            "changed": changed,
        }
    result["goal_changed"] = left.goal != right.goal
    result["conclusion_changed"] = left.conclusion != right.conclusion
    result["event_count"] = {"left": left.event_count, "right": right.event_count}
    return result


def _state_collection(state: ResearchState, payload_key: str) -> dict[str, Any]:
    plural = {"hypothesis": "hypotheses"}.get(payload_key, f"{payload_key}s")
    collection = getattr(state, plural, None)
    return collection if isinstance(collection, dict) else {}


def _validate_batch_payload(operation: str, payload: dict[str, Any], index: int) -> None:
    if operation == "set_goal" and not payload.get("task"):
        raise ProjectError(f"Batch item {index} goal requires task")
    if operation != "set_goal" and operation != "set_subgoal" and not payload.get("id"):
        raise ProjectError(f"Batch item {index} requires id")
    if operation == "propose_hypothesis" and not payload.get("claim"):
        raise ProjectError(f"Batch item {index} hypothesis requires claim")
    if operation == "register_route" and not isinstance(payload.get("reaction_ids", []), list):
        raise ProjectError(f"Batch item {index} route reaction_ids must be an array")
    if operation == "add_evidence" and not payload.get("summary"):
        raise ProjectError(f"Batch item {index} evidence requires summary")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug[:48] or "checkpoint"

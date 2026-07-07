"""Shared chemical data schemas for RetroAgent.

These dataclasses are intentionally lightweight. Tools still return JSON, but
the helpers here keep core records consistent across propose/evaluate, route
graphs, evidence tracking, and future discovery-mode workflows.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EvidenceLevel(str, Enum):
    KNOWN = "known"
    ANALOG_SUPPORTED = "analog-supported"
    TOOL_SUPPORTED = "tool-supported"
    SPECULATIVE = "speculative"


class RiskSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    TOOL_SUPPORTED = "tool-supported"
    NEEDS_VALIDATION = "needs-validation"
    REJECTED = "rejected"


class CriticScope(str, Enum):
    STRUCTURE = "structure"
    EVIDENCE = "evidence"
    FEASIBILITY = "feasibility"
    SAFETY = "safety"
    EXPERIMENT = "experiment"


@dataclass
class Evidence:
    level: EvidenceLevel
    source: str
    summary: str
    url: str = ""
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level"] = self.level.value
        return data


@dataclass
class RiskFlag:
    severity: RiskSeverity
    category: str
    message: str
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        return data


@dataclass
class AuditIssue:
    severity: RiskSeverity
    category: str
    message: str
    suggested_fix: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        return data


@dataclass
class ExperimentStep:
    objective: str
    procedure: str
    expected_observation: str = ""
    decision_point: str = ""
    safety_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentPlan:
    target_smiles: str
    objective: str
    steps: list[ExperimentStep] = field(default_factory=list)
    controls: list[ExperimentStep] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    failure_diagnostics: list[str] = field(default_factory=list)
    linked_issue_categories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "objective": self.objective,
            "steps": [_to_dict(step) for step in self.steps],
            "controls": [_to_dict(step) for step in self.controls],
            "success_criteria": list(self.success_criteria),
            "failure_diagnostics": list(self.failure_diagnostics),
            "linked_issue_categories": list(self.linked_issue_categories),
            "metadata": dict(self.metadata),
        }


@dataclass
class ResearchMemoryRecord:
    id: str
    target_smiles: str
    target_name: str = ""
    outcome: str = "proposed"
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    critic_summary: dict[str, Any] = field(default_factory=dict)
    experiment_plan: dict[str, Any] = field(default_factory=dict)
    route_graph_summary: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MoleculeNode:
    id: str
    smiles: str
    role: str = "intermediate"
    canonical_smiles: str = ""
    stock_status: str = "unknown"
    evidence: list[Evidence] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        data["risk_flags"] = [_to_dict(item) for item in self.risk_flags]
        return data


@dataclass
class ReactionRecord:
    id: str
    product_smiles: str
    precursor_smiles_list: list[str]
    template_index: int | None = None
    classification: str = ""
    library_occurrence: int | None = None
    evidence: list[Evidence] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_legacy_alias: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        data["risk_flags"] = [_to_dict(item) for item in self.risk_flags]
        if include_legacy_alias:
            data["precursors"] = list(self.precursor_smiles_list)
        return data


@dataclass
class Route:
    id: str
    target_smiles: str
    reaction_ids: list[str] = field(default_factory=list)
    score: float | None = None
    evidence: list[Evidence] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        data["risk_flags"] = [_to_dict(item) for item in self.risk_flags]
        return data


@dataclass
class ChemicalHypothesis:
    id: str
    claim: str
    target_smiles: str = ""
    route_summary: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    evidence: list[Evidence] = field(default_factory=list)
    risk_flags: list[RiskFlag] = field(default_factory=list)
    audit_issues: list[AuditIssue] = field(default_factory=list)
    next_experiments: list[ExperimentStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["evidence"] = [_to_dict(item) for item in self.evidence]
        data["risk_flags"] = [_to_dict(item) for item in self.risk_flags]
        data["audit_issues"] = [_to_dict(item) for item in self.audit_issues]
        data["next_experiments"] = [_to_dict(item) for item in self.next_experiments]
        return data


@dataclass
class CriticReport:
    target_smiles: str
    issues: list[AuditIssue] = field(default_factory=list)
    passed_checks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "issues": [_to_dict(item) for item in self.issues],
            "passed_checks": list(self.passed_checks),
            "metadata": dict(self.metadata),
            "issue_count": len(self.issues),
            "critical_count": sum(1 for issue in self.issues if issue.severity == RiskSeverity.CRITICAL),
            "warning_count": sum(1 for issue in self.issues if issue.severity == RiskSeverity.WARNING),
            "info_count": sum(1 for issue in self.issues if issue.severity == RiskSeverity.INFO),
        }


def _to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    return {"value": item}


def _coerce_evidence_level(value: Any) -> EvidenceLevel:
    if isinstance(value, EvidenceLevel):
        return value
    try:
        return EvidenceLevel(str(value))
    except ValueError:
        return EvidenceLevel.SPECULATIVE


def _coerce_risk_severity(value: Any) -> RiskSeverity:
    if isinstance(value, RiskSeverity):
        return value
    try:
        return RiskSeverity(str(value))
    except ValueError:
        return RiskSeverity.WARNING


def _coerce_hypothesis_status(value: Any) -> HypothesisStatus:
    if isinstance(value, HypothesisStatus):
        return value
    try:
        return HypothesisStatus(str(value))
    except ValueError:
        return HypothesisStatus.PROPOSED


def normalize_evidence(item: Evidence | dict[str, Any]) -> Evidence:
    if isinstance(item, Evidence):
        return item
    return Evidence(
        level=_coerce_evidence_level(item.get("level", EvidenceLevel.SPECULATIVE.value)),
        source=item.get("source", ""),
        summary=item.get("summary", ""),
        url=item.get("url", ""),
        score=item.get("score"),
        metadata=item.get("metadata", {}),
    )


def normalize_risk_flag(item: RiskFlag | dict[str, Any]) -> RiskFlag:
    if isinstance(item, RiskFlag):
        return item
    return RiskFlag(
        severity=_coerce_risk_severity(item.get("severity", RiskSeverity.WARNING.value)),
        category=item.get("category", ""),
        message=item.get("message", ""),
        evidence=[normalize_evidence(e) for e in item.get("evidence", [])],
    )


def normalize_audit_issue(item: AuditIssue | dict[str, Any]) -> AuditIssue:
    if isinstance(item, AuditIssue):
        return item
    return AuditIssue(
        severity=_coerce_risk_severity(item.get("severity", RiskSeverity.WARNING.value)),
        category=item.get("category", ""),
        message=item.get("message", ""),
        suggested_fix=item.get("suggested_fix", ""),
        evidence=[normalize_evidence(e) for e in item.get("evidence", [])],
    )


def normalize_experiment_step(item: ExperimentStep | dict[str, Any]) -> ExperimentStep:
    if isinstance(item, ExperimentStep):
        return item
    return ExperimentStep(
        objective=item.get("objective", ""),
        procedure=item.get("procedure", ""),
        expected_observation=item.get("expected_observation", ""),
        decision_point=item.get("decision_point", ""),
        safety_notes=list(item.get("safety_notes", [])),
    )


def normalize_chemical_hypothesis(
    item: ChemicalHypothesis | dict[str, Any],
    *,
    target_smiles: str = "",
    index: int = 1,
) -> ChemicalHypothesis:
    if isinstance(item, ChemicalHypothesis):
        return item
    return ChemicalHypothesis(
        id=item.get("id") or f"hypothesis_{index}",
        claim=item.get("claim", ""),
        target_smiles=item.get("target_smiles") or target_smiles,
        route_summary=item.get("route_summary", ""),
        status=_coerce_hypothesis_status(item.get("status", HypothesisStatus.PROPOSED.value)),
        evidence=[normalize_evidence(e) for e in item.get("evidence", [])],
        risk_flags=[normalize_risk_flag(r) for r in item.get("risk_flags", [])],
        audit_issues=[normalize_audit_issue(a) for a in item.get("audit_issues", [])],
        next_experiments=[
            normalize_experiment_step(step) for step in item.get("next_experiments", [])
        ],
        metadata=item.get("metadata", {}),
    )


def build_research_report(
    *,
    target_smiles: str,
    hypotheses: list[ChemicalHypothesis | dict[str, Any]],
    title: str = "",
) -> dict[str, Any]:
    normalized = [
        normalize_chemical_hypothesis(item, target_smiles=target_smiles, index=i)
        for i, item in enumerate(hypotheses, start=1)
    ]
    evidence_counts = {level.value: 0 for level in EvidenceLevel}
    risk_counts = {severity.value: 0 for severity in RiskSeverity}
    missing: list[str] = []
    for hypothesis in normalized:
        if not hypothesis.claim:
            missing.append(f"{hypothesis.id}: claim")
        if not hypothesis.evidence:
            missing.append(f"{hypothesis.id}: evidence")
        if not hypothesis.risk_flags and not hypothesis.audit_issues:
            missing.append(f"{hypothesis.id}: risk review")
        if not hypothesis.next_experiments:
            missing.append(f"{hypothesis.id}: next_experiments")
        for evidence in hypothesis.evidence:
            evidence_counts[evidence.level.value] += 1
        for risk in hypothesis.risk_flags:
            risk_counts[risk.severity.value] += 1
        for issue in hypothesis.audit_issues:
            risk_counts[issue.severity.value] += 1

    return {
        "title": title,
        "target_smiles": target_smiles,
        "hypothesis_count": len(normalized),
        "hypotheses": [hypothesis.to_dict() for hypothesis in normalized],
        "evidence_level_counts": evidence_counts,
        "risk_severity_counts": risk_counts,
        "next_experiment_count": sum(len(h.next_experiments) for h in normalized),
        "missing_review_fields": missing,
        "ready_for_review": not missing and bool(normalized),
    }


def build_reaction_record(
    *,
    product_smiles: str,
    precursor_smiles_list: list[str],
    reaction_id: str = "",
    template_index: int | None = None,
    classification: str = "",
    library_occurrence: int | None = None,
    evidence: list[Evidence] | None = None,
    risk_flags: list[RiskFlag] | None = None,
    metadata: dict[str, Any] | None = None,
    include_legacy_alias: bool = True,
) -> dict[str, Any]:
    """Build the canonical reaction JSON record used by tools."""
    record = ReactionRecord(
        id=reaction_id,
        product_smiles=product_smiles,
        precursor_smiles_list=list(precursor_smiles_list),
        template_index=template_index,
        classification=classification,
        library_occurrence=library_occurrence,
        evidence=evidence or [],
        risk_flags=risk_flags or [],
        metadata=metadata or {},
    )
    return record.to_dict(include_legacy_alias=include_legacy_alias)


def normalize_reaction_record(rxn: dict, *, target: str = "", index: int = 1) -> dict[str, Any]:
    """Normalize old and new reaction payloads into the canonical schema."""
    precursor_list = (
        rxn.get("precursor_smiles_list")
        or rxn.get("precursors")
        or rxn.get("reactants")
        or []
    )
    return build_reaction_record(
        reaction_id=rxn.get("id") or f"rxn_{index}",
        product_smiles=rxn.get("product_smiles") or rxn.get("product") or target,
        precursor_smiles_list=precursor_list,
        template_index=rxn.get("template_index"),
        classification=rxn.get("classification", ""),
        library_occurrence=rxn.get("library_occurrence"),
        evidence=rxn.get("evidence", []),
        risk_flags=rxn.get("risk_flags", []),
        metadata=rxn.get("metadata", {}),
    )

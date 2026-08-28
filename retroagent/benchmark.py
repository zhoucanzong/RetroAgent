"""Benchmark helpers for RetroAgent discovery-mode evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem


@dataclass
class DiscoveryBlindCase:
    id: str
    published_date: str
    source_title: str
    source_url: str
    task: str
    experiments: dict[str, dict[str, Any]]
    expected_outcome: str
    outcome_claim_groups: list[list[str]]
    mechanism_claim_groups: list[list[str]]
    contradiction_phrases: list[str]
    min_experiments: int
    success_criteria: list[str]
    scoring_profile: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryBlindCase":
        case = cls(
            id=str(data["id"]),
            published_date=str(data["published_date"]),
            source_title=str(data["source_title"]),
            source_url=str(data["source_url"]),
            task=str(data["task"]),
            experiments={str(key): dict(value) for key, value in data["experiments"].items()},
            expected_outcome=str(data["expected_outcome"]),
            outcome_claim_groups=[list(group) for group in data["outcome_claim_groups"]],
            mechanism_claim_groups=[list(group) for group in data["mechanism_claim_groups"]],
            contradiction_phrases=list(data.get("contradiction_phrases", [])),
            min_experiments=int(data.get("min_experiments", 2)),
            success_criteria=list(data["success_criteria"]),
            scoring_profile=str(data.get("scoring_profile", "photoreduction_post_set")),
        )
        if not case.id or not case.task:
            raise ValueError("blind case id and task are required")
        if len(case.experiments) < case.min_experiments:
            raise ValueError(f"{case.id}: insufficient registered experiments")
        if not case.outcome_claim_groups or not case.mechanism_claim_groups:
            raise ValueError(f"{case.id}: outcome and mechanism claim groups are required")
        if not case.success_criteria:
            raise ValueError(f"{case.id}: success criteria are required")
        if case.scoring_profile not in {"photoreduction_post_set", "aziridinium_migration"}:
            raise ValueError(f"{case.id}: unsupported scoring profile {case.scoring_profile!r}")
        return case

    def task_prompt(self) -> str:
        measurements = "; ".join(
            f"{experiment_id}: {experiment.get('question', '')}"
            for experiment_id, experiment in self.experiments.items()
        )
        return (
            "STRICT BLIND TEST. Do not use web_search, fetch_url, literature search, internet "
            "access, or prior knowledge of a paper. Use the supplied facts and local tools only.\n"
            f"{self.task}\n"
            f"The registered blind_study_oracle can reveal these withheld measurements: {measurements}.\n"
            f"Actively request at least {self.min_experiments} measurements before concluding. "
            "The oracle never reveals the published product assignment or conclusion.\n"
            "Before stopping, audit every conclusion dimension explicitly requested in the task "
            "against recorded evidence. If an available measurement addresses a dimension that "
            "would otherwise rely on an unsupported inference, request it rather than guessing.\n"
            "When one measurement reports several observations relevant to different task "
            "dimensions, carry each needed observation into the conclusion.\n"
            "Use explicit Uncertainty: and Falsification: labels in the final answer.\n"
            "Return one decisive evidence-backed answer that states what each requested measurement "
            "establishes, with uncertainty and falsification conditions."
        )


def load_blind_cases(path: Path) -> list[DiscoveryBlindCase]:
    cases = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(DiscoveryBlindCase.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid blind benchmark case at {path}:{line_no}: {exc}") from exc
    return cases


def score_blind_conclusion(
    conclusion: str,
    case: DiscoveryBlindCase,
    *,
    queried_experiment_ids: list[str],
) -> dict[str, Any]:
    if case.scoring_profile == "aziridinium_migration":
        return _score_aziridinium_conclusion(
            conclusion,
            case,
            queried_experiment_ids=queried_experiment_ids,
        )

    normalized = " ".join((conclusion or "").lower().split())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
        if sentence.strip()
    ]

    def groups_pass(groups: list[list[str]]) -> tuple[bool, list[bool]]:
        matches = [any(term.lower() in normalized for term in group) for group in groups]
        return all(matches), matches

    _, outcome_matches = groups_pass(case.outcome_claim_groups)
    h2_positive = any(re.search(
        r"(?:major (?:chemical )?(?:outcome|product)\s*(?:is|:)?\s*h2|"
        r"\bh2\b.{0,30}(?:the )?major (?:outcome|product)|"
        r"(?:select|choose|support|favor)\w*\s+h2\b)",
        sentence,
    ) for sentence in sentences)
    h2_negative = any(re.search(
        r"\bh2\b.{0,30}(?:lacks? support|is (?:ruled out|rejected|incorrect|false|not supported)|"
        r"was (?:ruled out|rejected)|is not the major)",
        sentence,
    ) for sentence in sentences)
    h1_selected = any(re.search(
        r"(?:major (?:chemical )?(?:outcome|product)\s*(?:is|:)?\s*h1|"
        r"(?:select|choose|support|favor)\w*\s+h1\b|"
        r"\bh1\b.{0,30}(?:the )?major (?:outcome|product)|\bh1\b.{0,20}dominates)",
        sentence,
    ) for sentence in sentences)
    outcome_pass = h2_positive and not h2_negative and not h1_selected

    scission_terms = ("beta-scission", "β-scission", "ring opening", "ring-opening", "fragment")
    bet_pattern = r"(?:\bbet\b|back electron transfer)"
    scission_pattern = r"(?:beta-scission|β-scission|ring opening|ring-opening|fragment(?:s|ed|ing|ation)?)"
    has_forward_regime = any(
        "both substrates" in sentence
        and any(term in sentence for term in (
            "diffusion", "indiscriminate", "non-selective", "nonselective", "reduces both", "rapid",
        ))
        for sentence in sentences
    )
    has_scission = any(any(term in sentence for term in scission_terms) for sentence in sentences)
    correct_scission_comparison = any(
        re.search(
            rf"{scission_pattern}.{{0,100}}(?:faster than|outcompet(?:es|ing)|outrun(?:s|ning)?|before).{{0,80}}{bet_pattern}",
            sentence,
        )
        for sentence in sentences
    )
    inverse_scission_comparison = any(
        re.search(
            rf"{bet_pattern}.{{0,100}}(?:faster than|outcompetes|outruns|happens before|before).{{0,80}}{scission_pattern}",
            sentence,
        )
        or re.search(rf"{scission_pattern}.{{0,80}}slower than.{{0,50}}{bet_pattern}", sentence)
        for sentence in sentences
    )
    has_styrene_radical = any(
        "styrene radical anion" in sentence or "styrene radical-anion" in sentence
        or "styrene•−" in sentence
        for sentence in sentences
    )
    styrene_fate_sentences = [
        sentence for sentence in sentences
        if (
            "styrene radical anion" in sentence
            or "styrene radical-anion" in sentence
            or "styrene•−" in sentence
        ) and re.search(bet_pattern, sentence)
    ]
    styrene_recycling_negated = any(re.search(
        r"(?:not|never|fails? to|does not|doesn't).{0,40}(?:recycl|recover|reoxid)|"
        r"(?:recycl|recover|reoxid).{0,30}(?:not|never|fails)",
        sentence,
    ) for sentence in styrene_fate_sentences)
    has_correct_styrene_fate = any(
        (
            any(term in sentence for term in ("recycl", "recover", "reoxid"))
            or re.search(rf"{bet_pattern}.{{0,80}}faster than proton", sentence)
        )
        for sentence in styrene_fate_sentences
    ) and not styrene_recycling_negated
    post_set_sentences = [
        sentence for sentence in sentences if "post-set" in sentence or "post set" in sentence
    ]
    post_set_control_negated = any(re.search(
        r"(?:does not|do not|doesn't|fails? to|not).{0,30}(?:control|determin|govern|dominat)|"
        r"(?:control|determin|govern|dominat).{0,30}(?:does not|do not|doesn't|not)",
        sentence,
    ) for sentence in post_set_sentences)
    has_post_set_control = any(
        any(term in sentence for term in ("branch", "kinetic", "competition"))
        and any(term in sentence for term in (
            "selectiv", "product-determining", "product determining", "determin", "control", "dominat",
        ))
        for sentence in post_set_sentences
    ) and not post_set_control_negated

    if len(case.mechanism_claim_groups) >= 6:
        mechanism_matches = [
            has_forward_regime,
            has_scission,
            correct_scission_comparison and not inverse_scission_comparison,
            has_styrene_radical,
            has_correct_styrene_fate,
            has_post_set_control,
        ]
        mechanism_pass = all(mechanism_matches)
    else:
        mechanism_pass, mechanism_matches = groups_pass(case.mechanism_claim_groups)
    contradictions = [
        phrase for phrase in case.contradiction_phrases if phrase.lower() in normalized
    ]
    if h2_negative:
        contradictions.append("H2 explicitly rejected")
    if h1_selected:
        contradictions.append("H1 declared major")
    if inverse_scission_comparison:
        contradictions.append("BET incorrectly outruns beta-scission")
    if styrene_recycling_negated:
        contradictions.append("styrene radical anion explicitly not recycled")
    if post_set_control_negated:
        contradictions.append("post-SET control explicitly negated")
    queried = sorted({item for item in queried_experiment_ids if item in case.experiments})
    measurement_pass = len(queried) >= case.min_experiments
    contradiction_pass = not contradictions
    return {
        "expected_outcome": case.expected_outcome,
        "outcome_pass": outcome_pass,
        "outcome_group_matches": outcome_matches,
        "mechanism_pass": mechanism_pass,
        "mechanism_group_matches": mechanism_matches,
        "measurement_pass": measurement_pass,
        "queried_experiment_ids": queried,
        "queried_experiment_count": len(queried),
        "contradiction_pass": contradiction_pass,
        "contradictions": contradictions,
        "blind_pass": outcome_pass and mechanism_pass and measurement_pass and contradiction_pass,
    }


def _score_aziridinium_conclusion(
    conclusion: str,
    case: DiscoveryBlindCase,
    *,
    queried_experiment_ids: list[str],
) -> dict[str, Any]:
    normalized = " ".join((conclusion or "").lower().split())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
        if sentence.strip()
    ]
    outcome_matches = [
        any(term.lower() in normalized for term in group)
        for group in case.outcome_claim_groups
    ]

    h2_positive = any(re.search(
        r"(?:\bh2\b.{0,35}(?:support|select|favor|mechanism)|"
        r"(?:support|select|choose|favor)\w*.{0,15}\bh2\b|"
        r"(?:mechanism|hypothesis)\s*(?:is|:)\s*h2|"
        r"(?:stepwise\s+)?mechanism\s*\(\s*h2\s*\)|"
        r"(?:proceed|occur)\w*\s+via\s+h2)",
        sentence,
    ) for sentence in sentences)
    h2_negative = any(re.search(
        r"\bh2\b.{0,35}(?:reject|ruled out|incorrect|false|not supported|lacks? support)",
        sentence,
    ) for sentence in sentences)
    competing_selected = any(
        re.search(
            r"(?:\bh[13]\b.{0,20}(?:is|as)\s+(?:the )?(?:supported|selected|favored|correct|dominant)|"
            r"\bh[13]\b.{0,20}mechanism.{0,15}(?:is )?supported|"
            r"(?:support|select|choose|favor)\w*.{0,15}\bh[13]\b|"
            r"(?:mechanism|hypothesis)\s*(?:is|:)\s*h[13])",
            sentence,
        )
        and not re.search(
            r"(?:falsif|rule[sd]? out|reject|not support).{0,70}\bh[13]\b|"
            r"\bh[13]\b.{0,70}(?:falsif|rule[sd]? out|reject|not supported)",
            sentence,
        )
        for sentence in sentences
    )
    outcome_pass = h2_positive and not h2_negative and not competing_selected

    aziridinium_pattern = r"aziridinium"
    migration_pattern = r"(?:1,2[- ]migration|1,2-migration|migration)"
    stepwise_sentences = [
        sentence for sentence in sentences
        if re.search(aziridinium_pattern, sentence) and re.search(migration_pattern, sentence)
    ]
    correct_step_order = any(
        (
            re.search(
                rf"(?:rapid|fast|quick)\w*.{{0,180}}(?:form|generat|accumulat)\w*.{{0,35}}"
                rf"{aziridinium_pattern}",
                sentence,
            )
            and re.search(
                rf"{aziridinium_pattern}.{{0,180}}(?:before|preced|followed by).{{0,70}}"
                rf"(?:slow|slower)?\w*.{{0,25}}{migration_pattern}",
                sentence,
            )
        )
        or re.search(
            rf"{aziridinium_pattern}.{{0,55}}(?:form|accumulat)\w*.{{0,80}}"
            rf"(?:before|preced|followed by).{{0,45}}(?:slow|slower)?\w*.{{0,20}}{migration_pattern}",
            sentence,
        )
        or re.search(
            rf"{aziridinium_pattern}.{{0,80}}(?:which )?(?:(?:then|subsequently) )?.{{0,20}}"
            rf"(?:undergo\w*.{{0,20}})?(?:slow|slower)\w*.{{0,30}}{migration_pattern}",
            sentence,
        )
        for sentence in stepwise_sentences
    )
    if not correct_step_order:
        transient_precedes_product = any(
            re.search(r"(?:transient|intermediate).{0,45}accumulat\w*.{0,45}before.{0,30}product", sentence)
            for sentence in sentences
        )
        transient_linked_to_aziridinium = any(
            "aziridinium" in sentence and re.search(r"(?:transient|ir band|spectrum|spectral)", sentence)
            for sentence in sentences
        )
        migration_from_aziridinium = any(
            "aziridinium" in sentence
            and "backside" in sentence
            and re.search(migration_pattern, sentence)
            for sentence in sentences
        )
        correct_step_order = (
            transient_precedes_product
            and transient_linked_to_aziridinium
            and migration_from_aziridinium
        )
    reversed_step_order = any(
        re.search(
            rf"{migration_pattern}.{{0,70}}(?:before|preced)\w*.{{0,45}}{aziridinium_pattern}",
            sentence,
        )
        or re.search(
            rf"{aziridinium_pattern}.{{0,50}}(?:after|subsequent to).{{0,45}}{migration_pattern}",
            sentence,
        )
        for sentence in stepwise_sentences
    )

    transient_sentences = [
        sentence for sentence in sentences
        if re.search(r"(?:transient|ir band|infrared|spectrum|spectral|cm-?1)", sentence)
        and "aziridinium" in sentence
    ]
    transient_assignment_negated = any(
        re.search(
            r"(?:not|does not|doesn't|fails? to).{0,45}(?:assign|identify|match|support)|"
            r"(?:transient|band).{0,45}(?:is not|isn't).{0,20}aziridinium",
            sentence,
        )
        and not any(term in sentence for term in (
            "falsif", "would challenge", "would contradict", "would refute", "would rule out",
        ))
        for sentence in transient_sentences
    )
    has_transient_assignment = any(
        re.search(r"(?:assign|identify|match|align|consistent|support)\w*", sentence)
        for sentence in transient_sentences
    ) and not transient_assignment_negated

    has_backside_anti = any(
        "backside" in sentence and "anti" in sentence and re.search(migration_pattern, sentence)
        for sentence in sentences
    )
    wrong_stereochemical_direction = any(
        (
            ("frontside" in sentence and re.search(r"(?:explain|give|yield|produce)\w*.{0,30}anti", sentence))
            or (re.search(r"\bsyn\b", sentence) and re.search(r"(?:product|selectiv|addition|migration)", sentence))
        )
        and not any(term in sentence for term in (
            "falsif", "would challenge", "would contradict", "would refute", "would rule out",
        ))
        for sentence in sentences
    )

    dpph_sentences = [sentence for sentence in sentences if "dpph" in sentence]
    has_low_effective_dpph = any(
        re.search(
            r"(?:low|lower|limited).{0,25}(?:effective )?concentration|"
            r"sparingly soluble|poorly soluble|sparing solubility",
            sentence,
        )
        for sentence in dpph_sentences
    )
    dpph_before_amine = any(
        re.search(
            r"(?:dpph.{0,45}(?:consum|deplet|disappear)\w*.{0,55}(?:before|prior to).{0,35}(?:free )?amine|"
            r"(?:consum|deplet)\w*.{0,25}dpph.{0,55}(?:before|prior to).{0,35}(?:free )?amine)",
            sentence,
        )
        for sentence in dpph_sentences
    )
    has_overreaction_suppression = any(
        re.search(
            r"(?:prevent|avoid|limit|suppress|minimi)\w*.{0,45}(?:overreaction|side reaction|further reaction)|"
            r"suppression of.{0,35}(?:overreaction|side reaction).{0,25}(?:explain|arise|result)",
            sentence,
        )
        for sentence in sentences
    )
    dpph_overlap_claimed = any(
        re.search(
            r"dpph.{0,35}(?:remain|persist|present).{0,45}(?:with|after|while).{0,35}(?:free )?amine|"
            r"dpph.{0,45}(?:not|never|isn't|is not).{0,20}(?:consum|deplet)",
            sentence,
        )
        for sentence in dpph_sentences
    )
    suppression_negated = any(re.search(
        r"(?:overreaction|side reaction).{0,25}(?:not|isn't|is not).{0,20}(?:prevent|avoid|limit|suppress)|"
        r"(?:not|does not|doesn't|fails? to).{0,35}(?:prevent|avoid|limit|suppress).{0,35}(?:overreaction|side reaction)",
        sentence,
    ) for sentence in sentences)

    has_uncertainty = any(
        term in normalized for term in ("uncertainty", "uncertain", "confidence", "limitation")
    )
    has_falsifier = any(
        term in normalized for term in ("falsif", "would contradict", "would refute", "would rule out")
    )
    mechanism_matches = [
        correct_step_order and not reversed_step_order,
        has_transient_assignment,
        has_backside_anti and not wrong_stereochemical_direction,
        dpph_before_amine and not dpph_overlap_claimed,
        has_low_effective_dpph and has_overreaction_suppression and not suppression_negated,
        has_uncertainty,
        has_falsifier,
    ]
    mechanism_pass = all(mechanism_matches)

    contradictions = [
        phrase for phrase in case.contradiction_phrases if phrase.lower() in normalized
    ]
    if h2_negative:
        contradictions.append("H2 explicitly rejected")
    if competing_selected:
        contradictions.append("H1 or H3 selected")
    if reversed_step_order:
        contradictions.append("migration incorrectly precedes aziridinium formation")
    if transient_assignment_negated:
        contradictions.append("aziridinium transient assignment explicitly negated")
    if wrong_stereochemical_direction:
        contradictions.append("frontside or syn stereochemical direction asserted")
    if dpph_overlap_claimed:
        contradictions.append("DPPH incorrectly persists with free amine")
    if suppression_negated:
        contradictions.append("overreaction suppression explicitly negated")

    queried = sorted({item for item in queried_experiment_ids if item in case.experiments})
    measurement_pass = len(queried) >= case.min_experiments
    contradiction_pass = not contradictions
    return {
        "expected_outcome": case.expected_outcome,
        "outcome_pass": outcome_pass,
        "outcome_group_matches": outcome_matches,
        "mechanism_pass": mechanism_pass,
        "mechanism_group_matches": mechanism_matches,
        "measurement_pass": measurement_pass,
        "queried_experiment_ids": queried,
        "queried_experiment_count": len(queried),
        "contradiction_pass": contradiction_pass,
        "contradictions": contradictions,
        "blind_pass": outcome_pass and mechanism_pass and measurement_pass and contradiction_pass,
    }


@dataclass
class DiscoveryFeedbackCase:
    id: str
    target_name: str
    target_smiles: str
    objective: str
    initial_routes: list[dict[str, Any]]
    feedback: dict[str, Any]
    success_criteria: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryFeedbackCase":
        case = cls(
            id=str(data["id"]),
            target_name=str(data["target_name"]),
            target_smiles=str(data["target_smiles"]),
            objective=str(data["objective"]),
            initial_routes=list(data["initial_routes"]),
            feedback=dict(data["feedback"]),
            success_criteria=list(data["success_criteria"]),
        )
        if Chem.MolFromSmiles(case.target_smiles) is None:
            raise ValueError(f"{case.id}: invalid target_smiles")
        if len(case.initial_routes) < 2:
            raise ValueError(f"{case.id}: at least two initial routes are required")
        route_ids = {str(route.get("id", "")) for route in case.initial_routes}
        if "" in route_ids or len(route_ids) != len(case.initial_routes):
            raise ValueError(f"{case.id}: initial route ids must be unique and non-empty")
        if "route_a" not in route_ids:
            raise ValueError(f"{case.id}: route_a is required as the preferred route")
        route_a = next(route for route in case.initial_routes if route.get("id") == "route_a")
        route_a_feasibility = float((route_a.get("scores", {}) or {}).get("feasibility", 0.0))
        if any(
            float((route.get("scores", {}) or {}).get("feasibility", 0.0)) > route_a_feasibility
            for route in case.initial_routes if route.get("id") != "route_a"
        ):
            raise ValueError(f"{case.id}: route_a must be the highest-feasibility initial route")
        if not case.feedback.get("reaction_ids"):
            raise ValueError(f"{case.id}: feedback reaction_ids are required")
        route_a_reactions = {str(item) for item in route_a.get("reaction_ids", [])}
        feedback_reactions = {str(item) for item in case.feedback["reaction_ids"]}
        if not feedback_reactions.issubset(route_a_reactions):
            raise ValueError(f"{case.id}: feedback reactions must belong to route_a")
        return case

    def task_prompt(self) -> str:
        return (
            f"Target: {self.target_smiles}\n"
            f"Objective: {self.objective}\n"
            "Use the registered feedback_oracle to obtain the withheld computational result. "
            "Then update route beliefs and choose a materially different route if required."
        )


@dataclass
class DiscoveryBenchmarkCase:
    id: str
    task_type: str
    target_name: str
    target_smiles: str
    objective: str
    desired_route_count: int
    required_evidence_levels: list[str]
    success_criteria: list[str]
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryBenchmarkCase":
        case = cls(
            id=data["id"],
            task_type=data["task_type"],
            target_name=data["target_name"],
            target_smiles=data["target_smiles"],
            objective=data["objective"],
            desired_route_count=int(data["desired_route_count"]),
            required_evidence_levels=list(data["required_evidence_levels"]),
            success_criteria=list(data["success_criteria"]),
            notes=data.get("notes", ""),
        )
        validate_discovery_case(case)
        return case

    def task_prompt(self) -> str:
        criteria = "; ".join(self.success_criteria)
        evidence = ", ".join(self.required_evidence_levels)
        return (
            f"{self.target_smiles}\n\n"
            f"Target name: {self.target_name}\n"
            f"Discovery objective: {self.objective}\n"
            f"Desired route hypotheses: at least {self.desired_route_count}\n"
            f"Required evidence labels to consider: {evidence}\n"
            f"Success criteria: {criteria}\n"
            f"Notes: {self.notes}"
        )


@dataclass
class DiscoveryBenchmarkResult:
    case_id: str
    target_name: str
    exit_status: str
    submission: str = ""
    trajectory_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "target_name": self.target_name,
            "exit_status": self.exit_status,
            "submission": self.submission,
            "trajectory_path": self.trajectory_path,
            "metrics": self.metrics,
            "error": self.error,
        }


def load_discovery_cases(path: Path) -> list[DiscoveryBenchmarkCase]:
    cases = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            cases.append(DiscoveryBenchmarkCase.from_dict(data))
        except Exception as exc:
            raise ValueError(f"Invalid discovery benchmark case at {path}:{line_no}: {exc}") from exc
    return cases


def load_feedback_cases(path: Path) -> list[DiscoveryFeedbackCase]:
    cases = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(DiscoveryFeedbackCase.from_dict(json.loads(line)))
        except Exception as exc:
            raise ValueError(f"Invalid feedback benchmark case at {path}:{line_no}: {exc}") from exc
    return cases


def validate_discovery_case(case: DiscoveryBenchmarkCase) -> None:
    if case.task_type != "discovery":
        raise ValueError(f"{case.id}: task_type must be discovery")
    if not case.id:
        raise ValueError("case id is required")
    if case.desired_route_count < 2:
        raise ValueError(f"{case.id}: desired_route_count must be at least 2")
    if not case.required_evidence_levels:
        raise ValueError(f"{case.id}: required_evidence_levels is empty")
    if not case.success_criteria:
        raise ValueError(f"{case.id}: success_criteria is empty")
    if Chem.MolFromSmiles(case.target_smiles) is None:
        raise ValueError(f"{case.id}: invalid target_smiles")


def score_discovery_submission(submission: str, case: DiscoveryBenchmarkCase) -> dict[str, Any]:
    """Heuristic scoring for benchmark summaries.

    This is not a chemistry oracle. It checks whether the agent produced the
    evidence-first portfolio shape that discovery mode requires; chemical
    validity still comes from tool traces, review, and human inspection.
    """
    text = submission or ""
    lower = text.lower()
    route_mentions = _count_route_mentions(text)
    evidence_present = [
        level for level in case.required_evidence_levels
        if level.lower() in lower
    ]
    risk_terms = [
        "risk", "风险", "warning", "critical", "selectivity", "chemoselectivity",
        "stereo", "chirality", "chiral", "safety", "unstable", "不稳定", "选择性",
    ]
    next_experiment_terms = [
        "next experiment", "next validation", "validate", "validation", "test",
        "实验", "验证", "control experiment", "对照",
    ]
    metrics = {
        "route_mentions": route_mentions,
        "desired_route_count": case.desired_route_count,
        "has_route_portfolio": route_mentions >= case.desired_route_count,
        "evidence_labels_present": evidence_present,
        "evidence_label_coverage": round(
            len(evidence_present) / len(case.required_evidence_levels), 3
        ),
        "mentions_risk": any(term in lower for term in risk_terms),
        "mentions_next_experiment": any(term in lower for term in next_experiment_terms),
        "submission_chars": len(text),
    }
    metrics["passes_shape_check"] = (
        metrics["has_route_portfolio"]
        and metrics["evidence_label_coverage"] >= 0.5
        and metrics["mentions_risk"]
        and metrics["mentions_next_experiment"]
    )
    return metrics


def analyze_trajectory(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract benchmark metrics from a RetroPlanner trajectory."""
    tool_calls = []
    proposal_reactions = []
    scored_route_count = 0
    solved_route_count = 0
    best_route_score = None
    evaluated_routes = 0
    stock_hits = 0
    literature_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    reviewer_messages = 0
    design_auditor_messages = 0
    warnings = 0
    structured_report_count = 0
    complete_structured_reports = 0
    critic_report_count = 0
    critic_warning_count = 0
    critic_critical_count = 0
    critic_route_graph_count = 0
    critic_unresolved_molecule_count = 0
    experiment_plan_count = 0
    experiment_step_count = 0
    experiment_control_count = 0
    memory_search_count = 0
    memory_record_count = 0

    for message in messages:
        content = message.get("content", "") or ""
        extra = message.get("extra", {}) or {}
        prompt_tokens += int(extra.get("prompt_tokens", 0) or 0)
        completion_tokens += int(extra.get("completion_tokens", 0) or 0)

        if "<role:reviewer>" in content:
            reviewer_messages += 1
        if "<role:design_auditor>" in content:
            design_auditor_messages += 1
        if "<system_warning>" in content or "<system_backtrack>" in content:
            warnings += 1

        tool = _extract_tag(content, "tool")
        output = _extract_tag(content, "output")
        if not tool:
            continue
        tool_calls.append(tool)
        if not output:
            continue
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            continue
        if tool == "propose":
            proposal_reactions.extend(data.get("reactions", []))
        elif tool == "score_routes":
            scored_route_count += len(data.get("routes", []))
            summary = data.get("summary", {}) or {}
            solved_route_count += int(summary.get("solved_route_count", 0) or 0)
            score = summary.get("best_score")
            if isinstance(score, (int, float)):
                best_route_score = score if best_route_score is None else max(best_route_score, score)
        elif tool == "evaluate":
            evaluated_routes += len(data.get("scores", {}))
        elif tool == "check_stock":
            stock_hits += len(data.get("in_stock", {}))
        elif tool == "search_literature":
            literature_hits += len(data.get("known_routes", []))
        elif tool == "web_search":
            literature_hits += len(data.get("results", []))
        elif tool == "build_research_report":
            structured_report_count += 1
            if data.get("ready_for_review"):
                complete_structured_reports += 1
        elif tool == "chemical_critic":
            critic_report_count += 1
            critic_warning_count += int(data.get("warning_count", 0) or 0)
            critic_critical_count += int(data.get("critical_count", 0) or 0)
            route_graph_metrics = (data.get("metadata", {}) or {}).get("route_graph", {}) or {}
            if route_graph_metrics.get("provided"):
                critic_route_graph_count += 1
                critic_unresolved_molecule_count += int(
                    route_graph_metrics.get("unsolved_molecule_count", 0) or 0
                )
        elif tool == "design_experiments":
            experiment_plan_count += 1
            experiment_step_count += len(data.get("steps", []))
            experiment_control_count += len(data.get("controls", []))
        elif tool == "research_memory":
            status = data.get("status")
            if status == "recorded":
                memory_record_count += 1
            else:
                memory_search_count += 1

    reaction_classes = sorted({
        str(reaction.get("classification", ""))
        for reaction in proposal_reactions
        if reaction.get("classification")
    })
    template_indices = sorted({
        int(reaction["template_index"])
        for reaction in proposal_reactions
        if reaction.get("template_index") is not None
    })
    precursor_sets = {
        ".".join(sorted(reaction.get("precursor_smiles_list") or reaction.get("precursors") or []))
        for reaction in proposal_reactions
    }
    precursor_sets.discard("")
    evidence_tools_used = any(
        tool in tool_calls for tool in ("search_literature", "web_search", "fetch_url")
    )
    risk_tools_used = any(
        tool in tool_calls for tool in (
            "check_patent",
            "functional_groups",
            "molecule_properties",
            "map_reaction",
            "chemical_critic",
            "design_experiments",
            "build_research_report",
            "research_memory",
        )
    )
    validation_tools_used = any(
        tool in tool_calls
        for tool in ("evaluate", "check_stock", "recommend_conditions", "map_reaction", "score_routes")
    )

    return {
        "tool_call_count": len(tool_calls),
        "unique_tools": sorted(set(tool_calls)),
        "tool_sequence": tool_calls,
        "proposal_reaction_count": len(proposal_reactions),
        "reaction_class_count": len(reaction_classes),
        "reaction_classes": reaction_classes[:20],
        "template_count": len(template_indices),
        "precursor_set_count": len(precursor_sets),
        "scored_route_count": scored_route_count,
        "solved_route_count": solved_route_count,
        "best_route_score": best_route_score,
        "evaluated_route_count": evaluated_routes,
        "stock_hit_count": stock_hits,
        "literature_hit_count": literature_hits,
        "structured_report_count": structured_report_count,
        "complete_structured_report_count": complete_structured_reports,
        "critic_report_count": critic_report_count,
        "critic_warning_count": critic_warning_count,
        "critic_critical_count": critic_critical_count,
        "critic_route_graph_count": critic_route_graph_count,
        "critic_unresolved_molecule_count": critic_unresolved_molecule_count,
        "experiment_plan_count": experiment_plan_count,
        "experiment_step_count": experiment_step_count,
        "experiment_control_count": experiment_control_count,
        "memory_search_count": memory_search_count,
        "memory_record_count": memory_record_count,
        "used_evidence_tools": evidence_tools_used,
        "used_risk_tools": risk_tools_used,
        "used_validation_tools": validation_tools_used,
        "reviewer_messages": reviewer_messages,
        "design_auditor_messages": design_auditor_messages,
        "warning_messages": warnings,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def analyze_trajectory_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"trajectory_missing": True}
    rows = json.loads(path.read_text())
    if rows and isinstance(rows[0], dict) and "type" in rows[0] and "payload" in rows[0]:
        return analyze_research_events(rows)
    return analyze_trajectory(rows)


def analyze_research_events(events: list[Any]) -> dict[str, Any]:
    rows = [_event_dict(event) for event in events]
    decisions = [row for row in rows if row.get("type") == "decision_made"]
    executions = [row for row in rows if row.get("type") == "execution_started"]
    tool_sequence = []
    for row in executions:
        action = row.get("payload", {}).get("action", {})
        tool_sequence.append(str(action.get("tool") or action.get("command") or "python"))
    proposal_reactions = []
    scored_route_count = 0
    solved_route_count = 0
    best_route_score = None
    evaluated_route_count = 0
    stock_hit_count = 0
    literature_hit_count = 0
    structured_report_count = 0
    critic_report_count = 0
    experiment_plan_count = 0
    memory_record_count = 0
    memory_search_count = 0
    for row in rows:
        if row.get("type") != "execution_finished":
            continue
        payload = row.get("payload", {})
        action = payload.get("action", {})
        tool = str(action.get("tool", ""))
        data = _event_result_data(payload.get("result", {}))
        if tool == "propose":
            proposal_reactions.extend(data.get("reactions", []))
        elif tool == "score_routes":
            scored_route_count += len(data.get("routes", []))
            summary = data.get("summary", {}) or {}
            solved_route_count += int(summary.get("solved_route_count", 0) or 0)
            score = summary.get("best_score")
            if isinstance(score, (int, float)):
                best_route_score = score if best_route_score is None else max(best_route_score, score)
        elif tool == "evaluate":
            evaluated_route_count += len(data.get("scores", {}))
        elif tool == "check_stock":
            stock_hit_count += len(data.get("in_stock", {}))
        elif tool in {"search_literature", "web_search"}:
            literature_hit_count += len(data.get("known_routes", data.get("results", [])))
        elif tool == "build_research_report":
            structured_report_count += 1
        elif tool == "chemical_critic":
            critic_report_count += 1
        elif tool == "design_experiments":
            experiment_plan_count += 1
        elif tool == "research_memory":
            if data.get("status") == "recorded":
                memory_record_count += 1
            else:
                memory_search_count += 1
    reaction_classes = {
        str(item.get("classification", "")) for item in proposal_reactions if item.get("classification")
    }
    template_indices = {
        item.get("template_index") for item in proposal_reactions if item.get("template_index") is not None
    }
    precursor_sets = {
        tuple(sorted(item.get("precursor_smiles_list") or item.get("precursors") or []))
        for item in proposal_reactions
    }
    precursor_sets.discard(())
    evidence_tools = {"search_literature", "web_search", "fetch_url"}
    risk_tools = {
        "check_patent", "functional_groups", "molecule_properties", "map_reaction",
        "chemical_critic", "design_experiments", "build_research_report", "research_memory",
        "build_route_portfolio", "validate_route_portfolio",
    }
    validation_tools = {
        "evaluate", "check_stock", "recommend_conditions", "map_reaction", "score_routes",
        "select_experiment", "update_hypotheses", "validate_route_portfolio",
    }
    closed_loop = score_closed_loop_events(rows)
    return {
        "event_count": len(rows),
        "decision_count": len(decisions),
        "execution_count": len(executions),
        "tool_call_count": len(tool_sequence),
        "tool_sequence": tool_sequence,
        "unique_tools": sorted(set(tool_sequence)),
        "proposal_reaction_count": len(proposal_reactions),
        "reaction_class_count": len(reaction_classes),
        "template_count": len(template_indices),
        "precursor_set_count": len(precursor_sets),
        "scored_route_count": scored_route_count,
        "solved_route_count": solved_route_count,
        "best_route_score": best_route_score,
        "evaluated_route_count": evaluated_route_count,
        "stock_hit_count": stock_hit_count,
        "literature_hit_count": literature_hit_count,
        "structured_report_count": structured_report_count,
        "critic_report_count": critic_report_count,
        "experiment_plan_count": experiment_plan_count,
        "memory_record_count": memory_record_count,
        "memory_search_count": memory_search_count,
        "used_evidence_tools": bool(set(tool_sequence).intersection(evidence_tools)),
        "used_risk_tools": bool(set(tool_sequence).intersection(risk_tools)),
        "used_validation_tools": bool(set(tool_sequence).intersection(validation_tools)),
        "reviewer_messages": 0,
        "warning_messages": sum(1 for row in rows if row.get("type") == "decision_rejected"),
        **closed_loop,
    }


def score_closed_loop_events(events: list[Any]) -> dict[str, bool]:
    rows = [_event_dict(event) for event in events]
    opposed_evidence = {
        str(row.get("payload", {}).get("id")): {
            str(item) for item in row.get("payload", {}).get("target_ids", [])
        }
        for row in rows
        if row.get("type") == "evidence_added"
        and row.get("payload", {}).get("direction") == "opposes"
    }
    failed_reactions = {
        str(row.get("payload", {}).get("id"))
        for row in rows
        if row.get("type") == "reaction_status_changed"
        and row.get("payload", {}).get("status") == "failed"
    }
    failed_routes = {
        str(row.get("payload", {}).get("id"))
        for row in rows
        if row.get("type") == "route_status_changed"
        and row.get("payload", {}).get("status") == "failed"
    }
    replan_index = next(
        (index for index, row in enumerate(rows) if row.get("type") == "replan_required"),
        None,
    )
    affected_routes: set[str] = set()
    if replan_index is not None:
        affected_routes = {
            str(item) for item in rows[replan_index].get("payload", {}).get("affected_route_ids", [])
        }
    distinct_replan = False
    if replan_index is not None:
        for row in rows[replan_index + 1:]:
            if row.get("type") != "decision_made":
                continue
            targets = {str(item) for item in row.get("payload", {}).get("target_ids", [])}
            if targets and targets.isdisjoint(affected_routes):
                distinct_replan = True
                break

    all_evidence = {
        str(row.get("payload", {}).get("id"))
        for row in rows if row.get("type") == "evidence_added"
    }
    completions = [row for row in rows if row.get("type") == "research_completed"]
    final_target_sets = [
        {str(item) for item in row.get("payload", {}).get("target_ids", [])}
        for row in completions
    ]
    final_avoids_failed = any(
        bool(targets) and targets.isdisjoint(failed_routes)
        for targets in final_target_sets
    )
    evidence_cited = any(
        bool(payload_ids)
        and payload_ids.issubset(all_evidence)
        for payload_ids in (
            {str(item) for item in row.get("payload", {}).get("evidence_ids", [])}
            for row in completions
        )
    )
    failed_action_indices: dict[str, int] = {}
    failure_evidence_ids: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("type") != "action_failed":
            continue
        payload = row.get("payload", {})
        fingerprint = str(payload.get("fingerprint", ""))
        if fingerprint:
            failed_action_indices.setdefault(fingerprint, index)
        evidence_id = str(payload.get("evidence_id", ""))
        if evidence_id:
            failure_evidence_ids.add(evidence_id)
    failed_action_not_repeated = not any(
        row.get("type") == "execution_started"
        and str(row.get("payload", {}).get("fingerprint", "")) in failed_action_indices
        and index > failed_action_indices[str(row.get("payload", {}).get("fingerprint", ""))]
        for index, row in enumerate(rows)
    )
    failure_evidence_cited = any(
        bool({str(item) for item in row.get("payload", {}).get("evidence_ids", [])}.intersection(failure_evidence_ids))
        for row in completions
    )
    metrics = {
        "failure_recorded": any(
            bool(target_ids.intersection(failed_reactions))
            for target_ids in opposed_evidence.values()
        ),
        "failed_route_abandoned": bool(failed_routes) and final_avoids_failed,
        "distinct_replan": distinct_replan and final_avoids_failed,
        "evidence_cited": evidence_cited,
        "failed_action_not_repeated": failed_action_not_repeated,
        "failure_evidence_cited": failure_evidence_cited,
    }
    metrics["closed_loop_pass"] = all(metrics.values())
    return metrics


def score_discovery_v3_events(events: list[Any]) -> dict[str, Any]:
    """Score whether a trajectory demonstrates the active route-discovery loop."""
    rows = [_event_dict(event) for event in events]
    hypothesis_ids = {
        str(row.get("payload", {}).get("id"))
        for row in rows if row.get("type") == "hypothesis_proposed"
    }
    selections = [
        row.get("payload", {}) for row in rows if row.get("type") == "experiment_selected"
    ]
    posterior_updates = [
        row.get("payload", {})
        for row in rows if row.get("type") == "hypothesis_posterior_updated"
    ]
    portfolios = [
        row.get("payload", {}) for row in rows if row.get("type") == "route_portfolio_built"
    ]
    validations = [
        row.get("payload", {}) for row in rows if row.get("type") == "route_validated"
    ]
    completions = [row.get("payload", {}) for row in rows if row.get("type") == "research_completed"]
    evidence_ids = {
        str(row.get("payload", {}).get("id"))
        for row in rows if row.get("type") == "evidence_added"
    }
    latest_updates: dict[str, dict[str, Any]] = {}
    for update in posterior_updates:
        latest_updates[str(update.get("id", ""))] = update
    posterior_sum = sum(
        float(update.get("posterior_probability", 0.0) or 0.0)
        for update in latest_updates.values()
    )
    validated_routes = {
        str(report.get("id")) for report in validations if report.get("status") == "passed"
    }
    completion_grounded = any(
        bool({str(item) for item in completion.get("target_ids", [])}.intersection(validated_routes))
        and bool(completion.get("evidence_ids"))
        and {str(item) for item in completion.get("evidence_ids", [])}.issubset(evidence_ids)
        for completion in completions
    )
    metrics = {
        "competing_hypotheses": len(hypothesis_ids) >= 2,
        "information_gain_selection": any(
            int(item.get("rank", 0) or 0) == 1
            and float(item.get("expected_information_gain", 0.0) or 0.0) > 0
            for item in selections
        ),
        "posterior_updated": (
            len(latest_updates) >= 2
            and abs(posterior_sum - 1.0) <= 1e-6
            and any(float(item.get("entropy_reduction", 0.0) or 0.0) > 0 for item in latest_updates.values())
        ),
        "multi_source_portfolio": any(
            int(item.get("multi_source_route_count", 0) or 0) >= 1 for item in portfolios
        ),
        "independent_validation": bool(validated_routes),
        "validated_completion": completion_grounded,
    }
    metrics["discovery_v3_pass"] = all(metrics.values())
    return metrics


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    if hasattr(event, "to_dict"):
        return event.to_dict()
    raise TypeError(f"Unsupported research event: {type(event).__name__}")


def _event_result_data(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    output = result.get("output")
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return output if isinstance(output, dict) else {}


def score_discovery_rubric(metrics: dict[str, Any], case: DiscoveryBenchmarkCase) -> dict[str, Any]:
    """Convert shape and trajectory metrics into benchmark pass/fail rubrics."""
    trajectory = metrics.get("trajectory", {}) or {}
    route_diversity_pass = (
        metrics.get("has_route_portfolio", False)
        and trajectory.get("proposal_reaction_count", 0) >= max(2, case.desired_route_count - 1)
        and (
            trajectory.get("reaction_class_count", 0) >= 2
            or trajectory.get("precursor_set_count", 0) >= 2
            or trajectory.get("template_count", 0) >= 2
        )
    )
    evidence_grounding_pass = (
        metrics.get("evidence_label_coverage", 0) >= 0.5
        and (
            trajectory.get("used_evidence_tools", False)
            or trajectory.get("literature_hit_count", 0) > 0
        )
    )
    validation_tool_pass = (
        trajectory.get("used_validation_tools", False)
        and (
            trajectory.get("evaluated_route_count", 0) > 0
            or trajectory.get("stock_hit_count", 0) > 0
            or trajectory.get("scored_route_count", 0) > 0
        )
    )
    risk_review_pass = (
        metrics.get("mentions_risk", False)
        and (
            trajectory.get("used_risk_tools", False)
            or trajectory.get("reviewer_messages", 0) > 0
            or trajectory.get("warning_messages", 0) > 0
        )
    )
    next_experiment_pass = metrics.get("mentions_next_experiment", False)
    shape_pass = metrics.get("passes_shape_check", False)

    checks = {
        "shape_pass": shape_pass,
        "route_diversity_pass": route_diversity_pass,
        "evidence_grounding_pass": evidence_grounding_pass,
        "validation_tool_pass": validation_tool_pass,
        "risk_review_pass": risk_review_pass,
        "next_experiment_pass": next_experiment_pass,
    }
    passed = sum(1 for value in checks.values() if value)
    required = len(checks)
    return {
        **checks,
        "passed_checks": passed,
        "required_checks": required,
        "overall_pass": passed == required,
        "score": round(passed / required, 3),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def run_discovery_cases(
    cases: list[DiscoveryBenchmarkCase],
    *,
    output_dir: Path,
    runner: Callable[[DiscoveryBenchmarkCase, Path], dict[str, Any]],
    dry_run: bool = False,
) -> list[DiscoveryBenchmarkResult]:
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        trajectory_path = output_dir / f"{case.id}.trajectory.json"
        if dry_run:
            results.append(DiscoveryBenchmarkResult(
                case_id=case.id,
                target_name=case.target_name,
                exit_status="DryRun",
                trajectory_path=str(trajectory_path),
                metrics={
                    "task_prompt_chars": len(case.task_prompt()),
                    "desired_route_count": case.desired_route_count,
                    "required_evidence_levels": case.required_evidence_levels,
                },
            ))
            continue
        try:
            raw = runner(case, trajectory_path)
            submission = raw.get("submission", "")
            metrics = score_discovery_submission(submission, case)
            metrics["trajectory"] = analyze_trajectory_file(trajectory_path)
            metrics["rubric"] = score_discovery_rubric(metrics, case)
            results.append(DiscoveryBenchmarkResult(
                case_id=case.id,
                target_name=case.target_name,
                exit_status=raw.get("exit_status", "Unknown"),
                submission=submission,
                trajectory_path=str(trajectory_path),
                metrics=metrics,
            ))
        except Exception as exc:
            results.append(DiscoveryBenchmarkResult(
                case_id=case.id,
                target_name=case.target_name,
                exit_status=type(exc).__name__,
                trajectory_path=str(trajectory_path),
                error=str(exc),
            ))
    return results


def _count_route_mentions(text: str) -> int:
    patterns = [
        r"(?im)^\s*(?:route|路线)\s+[A-Z0-9一二三四五六七八九十]+[\.:：、)]?",
        r"(?im)^\s*#{1,4}\s*(?:route|路线)\s+",
    ]
    matches = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            matches.add(match.start())
    return len(matches)


def _extract_tag(content: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL)
    return match.group(1).strip() if match else ""

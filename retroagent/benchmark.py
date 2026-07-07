"""Benchmark helpers for RetroAgent discovery-mode evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem


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
        tool in tool_calls for tool in ("evaluate", "check_stock", "recommend_conditions", "map_reaction")
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
    return analyze_trajectory(json.loads(path.read_text()))


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

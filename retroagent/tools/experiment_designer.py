"""ExperimentDesignerTool: turn critic findings into testable next actions.

The tool creates structured experiment plans from hypotheses, reaction records,
and chemical_critic findings. It does not claim experiments have been run.
"""

import json
from typing import Any

from retroagent.schemas import ExperimentPlan, ExperimentStep, normalize_audit_issue


class ExperimentDesignerTool:
    name = "design_experiments"
    description = (
        "Convert route hypotheses and chemical_critic issues into structured "
        "next experiments with controls, expected observations, decision points, "
        "success criteria, and failure diagnostics. Use this after chemical_critic "
        "and before build_research_report in discovery mode. The tool proposes "
        "experiments only; it does not validate that experiments succeeded."
    )

    def execute(self, parameters: dict) -> str:
        plan = build_experiment_plan(
            target_smiles=parameters.get("target_smiles", ""),
            hypotheses=parameters.get("hypotheses", []),
            reactions=parameters.get("reactions", []),
            critic_report=parameters.get("critic_report", {}),
            route_graph=parameters.get("route_graph", {}),
        )
        return json.dumps(plan, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string", "description": "Target molecule SMILES."},
                "hypotheses": {
                    "type": "array",
                    "description": "Route hypotheses being tested.",
                    "items": {"type": "object"},
                },
                "reactions": {
                    "type": "array",
                    "description": "Reaction records from propose.",
                    "items": {"type": "object"},
                },
                "critic_report": {
                    "type": "object",
                    "description": "Output from chemical_critic.",
                },
                "route_graph": {
                    "type": "object",
                    "description": "Serialized RouteGraph, if available.",
                },
            },
            "required": ["target_smiles"],
        }


def build_experiment_plan(
    *,
    target_smiles: str,
    hypotheses: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    critic_report: dict[str, Any],
    route_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues = [normalize_audit_issue(item) for item in critic_report.get("issues", [])]
    categories = sorted({issue.category for issue in issues})
    steps: list[ExperimentStep] = []
    controls: list[ExperimentStep] = []
    success_criteria: list[str] = []
    diagnostics: list[str] = []

    if not issues:
        steps.append(_general_route_probe(target_smiles, reactions))
        controls.append(_blank_or_no_reagent_control())
        success_criteria.append("At least one route shows clean target formation signal under microscale conditions.")
        diagnostics.append("If no route shows target signal, revisit disconnections and evidence support.")
    else:
        for category in categories:
            category_issues = [issue for issue in issues if issue.category == category]
            step, control, criteria, diagnostic = _plan_for_category(
                category,
                target_smiles,
                reactions,
                category_issues,
                route_graph or {},
            )
            steps.append(step)
            if control:
                controls.append(control)
            success_criteria.append(criteria)
            diagnostics.append(diagnostic)

    route_graph_metrics = (critic_report.get("metadata", {}) or {}).get("route_graph", {}) or {}
    if route_graph_metrics.get("provided") and route_graph_metrics.get("unsolved_molecule_count", 0):
        steps.append(ExperimentStep(
            objective="Resolve route-graph precursor gaps",
            procedure=(
                "For each unresolved precursor node, run supplier/common-reagent lookup, "
                "then choose either stock purchase, one-step precursor synthesis, or recursive expansion."
            ),
            expected_observation="Every route-graph leaf is labeled as stock, common reagent, or recursively expandable.",
            decision_point="Reject routes whose key precursors remain unsupported after lookup and one-step expansion.",
            safety_notes=["Do not treat ZINC absence alone as proof that a common reagent is unavailable."],
        ))
        success_criteria.append("All unresolved route-graph leaf nodes receive a documented availability decision.")
        diagnostics.append("Persistent unresolved leaves indicate the route is not yet executable.")

    plan = ExperimentPlan(
        target_smiles=target_smiles,
        objective=_plan_objective(hypotheses, issues),
        steps=_dedupe_steps(steps),
        controls=_dedupe_steps(controls),
        success_criteria=_dedupe_strings(success_criteria),
        failure_diagnostics=_dedupe_strings(diagnostics),
        linked_issue_categories=categories,
        metadata={
            "hypothesis_count": len(hypotheses),
            "reaction_count": len(reactions),
            "critic_issue_count": len(issues),
            "route_graph_provided": bool(route_graph),
            "priority": _priority_from_issues(issues),
        },
    ).to_dict()
    plan["ready_for_report"] = bool(plan["steps"]) and bool(plan["success_criteria"])
    return plan


def _plan_for_category(
    category: str,
    target_smiles: str,
    reactions: list[dict[str, Any]],
    issues,
    route_graph: dict[str, Any],
) -> tuple[ExperimentStep, ExperimentStep | None, str, str]:
    reaction_label = _reaction_label(reactions)
    message = issues[0].message if issues else category
    match category:
        case "mechanism":
            return (
                ExperimentStep(
                    objective="Validate reaction mechanism and product connectivity",
                    procedure=(
                        f"For {reaction_label}, prepare or simulate the reaction SMILES, run atom mapping/forward "
                        "product validation, and compare predicted bond changes with the proposed disconnection."
                    ),
                    expected_observation=f"Mapped bond changes account for formation of {target_smiles}.",
                    decision_point="Advance only if mapping/forward validation supports the same connectivity.",
                ),
                ExperimentStep(
                    objective="Negative mapping control",
                    procedure="Map a deliberately mismatched precursor set to confirm the audit catches incoherent bond changes.",
                    expected_observation="Control mapping has low confidence or incorrect connectivity.",
                    decision_point="If the control looks equally plausible, do not use mapping as strong evidence.",
                ),
                "Mechanism validation supports the proposed product connectivity and bond changes.",
                f"If mechanism validation fails, revisit the disconnection flagged by critic: {message}",
            )
        case "stereochemistry":
            return (
                ExperimentStep(
                    objective="Test stereochemical control",
                    procedure=(
                        "Run a microscale version of the stereochemistry-setting or stereochemistry-retaining step, "
                        "then analyze crude/product by chiral HPLC, chiral SFC, or stereospecific NMR method."
                    ),
                    expected_observation="The desired stereochemical assignment is detected with acceptable selectivity.",
                    decision_point="Advance only if stereochemical identity and selectivity are explicitly measured.",
                ),
                ExperimentStep(
                    objective="Racemate or achiral-method control",
                    procedure="Analyze racemic/known standard or achiral method in parallel to verify the chiral assay resolves species.",
                    expected_observation="Control confirms the assay can distinguish stereochemical outcomes.",
                    decision_point="If assay cannot resolve stereoisomers, redesign analytical method before route ranking.",
                ),
                "Stereochemical outcome is specified and analytically verified.",
                "If selectivity is poor, add resolution, asymmetric conditions, or choose a route avoiding that stereocenter.",
            )
        case "conditions":
            return (
                ExperimentStep(
                    objective="Screen runnable reaction conditions",
                    procedure=(
                        "Run a small matrix varying catalyst/reagent, solvent, and temperature around literature-like "
                        "conditions; monitor conversion, target formation, and major byproducts."
                    ),
                    expected_observation="At least one condition gives target signal with controlled byproduct profile.",
                    decision_point="Advance the simplest condition meeting conversion and impurity thresholds.",
                ),
                ExperimentStep(
                    objective="No-catalyst or no-activator control",
                    procedure="Run the same substrate set without the key catalyst/activator.",
                    expected_observation="Control shows lower conversion or different product profile.",
                    decision_point="If control performs equally, revise the mechanistic condition assumption.",
                ),
                "A concrete catalyst/reagent, solvent, and temperature window is identified.",
                "If all conditions fail, seek analog literature or change activation strategy.",
            )
        case "route_graph" | "availability":
            unresolved = (route_graph.get("stats", {}) or {}).get("unsolved_molecule_count")
            return (
                ExperimentStep(
                    objective="Resolve precursor availability and recursive expansion",
                    procedure=(
                        "List all unresolved leaf precursors, check supplier/common-reagent status, and for unsupported "
                        "intermediates propose one recursive disconnection or one-step synthesis."
                    ),
                    expected_observation="Each leaf has stock/common-reagent/recursive-expansion status.",
                    decision_point="Do not prioritize routes with unresolved key precursors after recursive expansion.",
                ),
                None,
                "All route leaves have a documented availability or synthesis path.",
                f"If unresolved leaves remain ({unresolved or 'unknown'}), the route is not yet executable.",
            )
        case "diversity" | "convergence":
            return (
                ExperimentStep(
                    objective="Benchmark route diversity and convergence",
                    procedure=(
                        "Select at least two route families with different disconnections or starting materials and compare "
                        "step count, convergence, precursor availability, and first experimental validation step."
                    ),
                    expected_observation="Routes differ in disconnection logic or precursor set, not just template rank.",
                    decision_point="Keep both only if they test meaningfully different synthetic hypotheses.",
                ),
                None,
                "Portfolio contains at least two meaningfully distinct route hypotheses.",
                "If alternatives collapse to the same precursor logic, return to disconnect/propose for new branches.",
            )
        case "safety":
            return (
                ExperimentStep(
                    objective="De-risk hazardous or reactive precursor handling",
                    procedure=(
                        "Before synthetic validation, document reagent hazards, compatible quench, concentration, scale cap, "
                        "and safer surrogate or milder precursor option."
                    ),
                    expected_observation="A bounded microscale protocol and safer fallback are documented.",
                    decision_point="Do not run the step until safety notes and quench plan are explicit.",
                    safety_notes=["Use microscale validation first.", "Record quench and incompatibility assumptions."],
                ),
                ExperimentStep(
                    objective="Safer surrogate control",
                    procedure="Test a less hazardous analog or surrogate substrate when feasible.",
                    expected_observation="Surrogate reproduces the intended transformation trend.",
                    decision_point="Escalate only if surrogate supports the chemistry.",
                ),
                "Safety plan covers handling, scale, quench, and fallback reagent choice.",
                "If risk cannot be bounded at microscale, reject or redesign the route.",
            )
        case _:
            return (
                ExperimentStep(
                    objective=f"Address critic issue: {category}",
                    procedure=f"Design a focused test for critic finding: {message}",
                    expected_observation="The test resolves whether the issue blocks route prioritization.",
                    decision_point="Advance only if the issue is resolved or explicitly accepted as an experimental risk.",
                ),
                None,
                f"Critic issue category '{category}' has a documented resolution test.",
                f"If unresolved, carry '{category}' forward as an explicit risk in the final report.",
            )


def _general_route_probe(target_smiles: str, reactions: list[dict[str, Any]]) -> ExperimentStep:
    return ExperimentStep(
        objective="Run first-pass route validation",
        procedure=(
            f"Choose the highest-scoring or best-evidenced proposed reaction toward {target_smiles}, run a microscale "
            "validation under literature-like or recommended conditions, and monitor target/product formation."
        ),
        expected_observation="Target or expected intermediate appears with interpretable conversion and byproduct profile.",
        decision_point="Advance route if target signal and precursor availability are both plausible.",
    )


def _blank_or_no_reagent_control() -> ExperimentStep:
    return ExperimentStep(
        objective="Baseline control",
        procedure="Run a blank/no-key-reagent control under otherwise identical analytical conditions.",
        expected_observation="No target formation signal in the absence of the key reagent or activation mode.",
        decision_point="If target-like signal appears in the control, troubleshoot assay or contamination.",
    )


def _reaction_label(reactions: list[dict[str, Any]]) -> str:
    if not reactions:
        return "the proposed reaction"
    first = reactions[0]
    return first.get("id") or first.get("classification") or "the proposed reaction"


def _plan_objective(hypotheses: list[dict[str, Any]], issues) -> str:
    if issues:
        return "Resolve critic findings with targeted validation experiments."
    if len(hypotheses) > 1:
        return "Compare route hypotheses with first-pass validation experiments."
    return "Generate first-pass validation data for the route hypothesis."


def _priority_from_issues(issues) -> str:
    severities = {issue.severity.value for issue in issues}
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    if issues:
        return "info"
    return "routine"


def _dedupe_steps(steps: list[ExperimentStep]) -> list[ExperimentStep]:
    seen = set()
    unique = []
    for step in steps:
        key = (step.objective, step.procedure)
        if key not in seen:
            unique.append(step)
            seen.add(key)
    return unique


def _dedupe_strings(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique

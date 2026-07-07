"""ChemicalCriticTool: deterministic audit checks for discovery hypotheses.

This is not a chemistry oracle. It turns common research-agent failure modes
into explicit findings: weak evidence, missing validation, thin experiments,
low-scoring reactions, unresolved precursors, and unreviewed risks.
"""

import json
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from retroagent.schemas import (
    AuditIssue,
    CriticReport,
    Evidence,
    EvidenceLevel,
    RiskSeverity,
    normalize_chemical_hypothesis,
    normalize_reaction_record,
)
from retroagent.tools.functional_groups import summarize_broad_functional_groups


class ChemicalCriticTool:
    name = "chemical_critic"
    description = (
        "Audit route hypotheses before final discovery submission. Checks for "
            "missing evidence, speculative-only support, missing risk review, thin "
            "next experiments, low route scores, rare templates, and unresolved "
            "precursors. Also uses RDKit structure checks to flag template/product "
            "functional-group mismatches, unassigned stereochemistry, reactive "
            "precursors, low-confidence atom mapping, and incomplete condition "
            "reports. If a route_graph is provided, audits unresolved molecules, "
            "target coverage, branch diversity, convergence, and protection-loop "
            "signals. The tool reports deterministic findings only; it does not "
            "prove a route chemically valid."
    )

    def execute(self, parameters: dict) -> str:
        report = build_critic_report(
            target_smiles=parameters.get("target_smiles", ""),
            hypotheses=parameters.get("hypotheses", []),
            reactions=parameters.get("reactions", []),
            route_scores=parameters.get("route_scores", {}),
            stock_hits=parameters.get("stock_hits", {}),
            literature_hits=parameters.get("literature_hits", []),
            mapping_reports=parameters.get("mapping_reports", []),
            condition_reports=parameters.get("condition_reports", []),
            route_graph=parameters.get("route_graph", {}),
        )
        return json.dumps(report, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string", "description": "Target molecule SMILES."},
                "hypotheses": {
                    "type": "array",
                    "description": "Route hypotheses, usually from build_research_report.",
                    "items": {"type": "object"},
                },
                "reactions": {
                    "type": "array",
                    "description": "Reaction records from propose.",
                    "items": {"type": "object"},
                },
                "route_scores": {
                    "type": "object",
                    "description": "Scores from evaluate, keyed by reaction id.",
                },
                "stock_hits": {
                    "type": "object",
                    "description": "In-stock precursors from check_stock.",
                },
                "literature_hits": {
                    "type": "array",
                    "description": "Known or analog literature hits.",
                    "items": {"type": "object"},
                },
                "mapping_reports": {
                    "type": "array",
                    "description": "Atom-mapping reports from map_reaction.",
                    "items": {"type": "object"},
                },
                "condition_reports": {
                    "type": "array",
                    "description": "Condition suggestions from recommend_conditions.",
                    "items": {"type": "object"},
                },
                "route_graph": {
                    "type": "object",
                    "description": "Serialized AND-OR RouteGraph from blackboard.route_graph.to_dict().",
                },
            },
            "required": ["target_smiles"],
        }


def build_critic_report(
    *,
    target_smiles: str,
    hypotheses: list[dict[str, Any]],
    reactions: list[dict[str, Any]],
    route_scores: dict[str, Any],
    stock_hits: dict[str, Any],
    literature_hits: list[dict[str, Any]],
    mapping_reports: list[dict[str, Any]] | None = None,
    condition_reports: list[dict[str, Any]] | None = None,
    route_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_hypotheses = [
        normalize_chemical_hypothesis(item, target_smiles=target_smiles, index=i)
        for i, item in enumerate(hypotheses, start=1)
    ]
    normalized_reactions = [
        normalize_reaction_record(item, target=target_smiles, index=i)
        for i, item in enumerate(reactions, start=1)
    ]

    issues: list[AuditIssue] = []
    passed: list[str] = []

    if normalized_hypotheses:
        passed.append("hypothesis_portfolio_present")
    else:
        issues.append(_issue(
            RiskSeverity.CRITICAL,
            "structure",
            "No route hypotheses were provided for critic review.",
            "Provide at least one route hypothesis with evidence, risks, and a next experiment.",
        ))

    for hypothesis in normalized_hypotheses:
        _audit_hypothesis(hypothesis, issues, passed)

    if normalized_reactions:
        passed.append("reaction_records_present")
        _audit_reactions(normalized_reactions, route_scores, stock_hits, issues, passed)
        _audit_chemistry_features(normalized_reactions, issues, passed)
    else:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "validation",
            "No structured reaction records were provided.",
            "Run propose and include candidate reaction records before relying on the route portfolio.",
        ))

    if route_scores:
        passed.append("route_scores_present")
    else:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "validation",
            "No route feasibility scores were provided.",
            "Run evaluate on proposed reactions and include the score table.",
        ))

    if stock_hits:
        passed.append("stock_signal_present")
    elif normalized_reactions:
        issues.append(_issue(
            RiskSeverity.INFO,
            "availability",
            "No precursor stock hits were provided.",
            "Run check_stock, then separately treat common reagents absent from ZINC as supplier-check items.",
        ))

    if literature_hits:
        passed.append("literature_signal_present")
    else:
        issues.append(_issue(
            RiskSeverity.INFO,
            "evidence",
            "No literature hits were provided to the critic.",
            "Use search_literature or web_search to distinguish known routes from analog-supported hypotheses.",
        ))

    _audit_mapping_reports(mapping_reports or [], issues, passed)
    _audit_condition_reports(condition_reports or [], issues, passed)
    route_graph_metrics = _audit_route_graph(route_graph or {}, issues, passed)

    category_counts: dict[str, int] = {}
    for issue in issues:
        category_counts[issue.category] = category_counts.get(issue.category, 0) + 1

    report = CriticReport(
        target_smiles=target_smiles,
        issues=issues,
        passed_checks=sorted(set(passed)),
        metadata={
            "hypothesis_count": len(normalized_hypotheses),
            "reaction_count": len(normalized_reactions),
            "score_count": len(route_scores),
            "stock_hit_count": len(stock_hits),
            "literature_hit_count": len(literature_hits),
            "mapping_report_count": len(mapping_reports or []),
            "condition_report_count": len(condition_reports or []),
            "route_graph": route_graph_metrics,
            "issue_category_counts": category_counts,
        },
    ).to_dict()
    report["needs_revision"] = report["critical_count"] > 0 or report["warning_count"] > 0
    return report


def _audit_hypothesis(hypothesis, issues: list[AuditIssue], passed: list[str]) -> None:
    prefix = f"{hypothesis.id}: "
    if hypothesis.claim.strip():
        passed.append("claims_present")
    else:
        issues.append(_issue(
            RiskSeverity.CRITICAL,
            "structure",
            f"{prefix}missing route claim.",
            "State the route hypothesis as a concrete, testable claim.",
        ))

    if not hypothesis.evidence:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "evidence",
            f"{prefix}no evidence records attached.",
            "Attach literature, tool, analog, or explicitly speculative evidence.",
        ))
    elif all(e.level == EvidenceLevel.SPECULATIVE for e in hypothesis.evidence):
        issues.append(_issue(
            RiskSeverity.WARNING,
            "evidence",
            f"{prefix}support is speculative-only.",
            "Add analog/tool/literature support or mark this route as a low-confidence experiment.",
        ))
    else:
        passed.append("non_speculative_evidence_present")

    if hypothesis.risk_flags or hypothesis.audit_issues:
        passed.append("risk_review_present")
    else:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "risk",
            f"{prefix}no risk review attached.",
            "Add stereochemistry, selectivity, condition, safety, IP, or availability risks.",
        ))

    if not hypothesis.next_experiments:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "experiment",
            f"{prefix}no next experiment provided.",
            "Add a small-scale experiment, expected observation, and decision point.",
        ))
    else:
        passed.append("next_experiment_present")
        for i, step in enumerate(hypothesis.next_experiments, start=1):
            text = f"{step.objective} {step.procedure}".strip()
            if len(text) < 40 or not step.decision_point:
                issues.append(_issue(
                    RiskSeverity.INFO,
                    "experiment",
                    f"{prefix}experiment {i} is underspecified.",
                    "Include concrete scale/condition screen, expected observation, and go/no-go decision.",
                ))


def _audit_reactions(
    reactions: list[dict[str, Any]],
    route_scores: dict[str, Any],
    stock_hits: dict[str, Any],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    stock_smiles = set(stock_hits)
    unresolved_precursors = set()

    for reaction in reactions:
        reaction_id = reaction.get("id") or "unknown_reaction"
        score = _score_for_reaction(route_scores, reaction_id)
        if score is not None and score < 0.35:
            issues.append(_issue(
                RiskSeverity.WARNING,
                "feasibility",
                f"{reaction_id}: low route score ({score:.3f}).",
                "Backtrack, seek analog evidence, or design a validation experiment before prioritizing this route.",
            ))
        elif score is not None:
            passed.append("non_low_route_score_present")

        occurrence = reaction.get("library_occurrence")
        if occurrence is not None and occurrence < 5:
            issues.append(_issue(
                RiskSeverity.INFO,
                "template_bias",
                f"{reaction_id}: template library occurrence is very low ({occurrence}).",
                "Treat the template as weak support unless literature or forward validation backs it.",
            ))

        for precursor in reaction.get("precursor_smiles_list", []):
            if precursor and stock_smiles and precursor not in stock_smiles:
                unresolved_precursors.add(precursor)

    if unresolved_precursors:
        preview = ", ".join(sorted(unresolved_precursors)[:5])
        issues.append(_issue(
            RiskSeverity.INFO,
            "availability",
            f"{len(unresolved_precursors)} proposed precursor(s) are not in provided stock hits: {preview}",
            "Expand unresolved precursors recursively or document supplier/common-reagent assumptions.",
        ))
    elif stock_smiles:
        passed.append("precursor_stock_checked")


def _audit_chemistry_features(
    reactions: list[dict[str, Any]],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    for reaction in reactions:
        reaction_id = reaction.get("id") or "unknown_reaction"
        product = reaction.get("product_smiles", "")
        precursors = reaction.get("precursor_smiles_list", [])
        classification = str(reaction.get("classification", "")).lower()
        product_facts = _molecule_facts(product)
        precursor_facts = [_molecule_facts(smiles) for smiles in precursors]

        if product_facts.get("valid"):
            passed.append("product_structure_parseable")
            _audit_template_product_match(reaction_id, classification, product_facts, issues, passed)
            _audit_stereochemistry(reaction_id, product_facts, issues, passed)
            _audit_polyfunctional_selectivity(reaction_id, classification, product_facts, issues)
        else:
            issues.append(_issue(
                RiskSeverity.CRITICAL,
                "structure",
                f"{reaction_id}: product SMILES could not be parsed.",
                "Fix the reaction record before using it as route evidence.",
            ))

        for precursor, facts in zip(precursors, precursor_facts, strict=False):
            if not facts.get("valid"):
                issues.append(_issue(
                    RiskSeverity.WARNING,
                    "structure",
                    f"{reaction_id}: precursor SMILES could not be parsed: {precursor}",
                    "Fix or replace the precursor before recursive route expansion.",
                ))
                continue
            _audit_precursor_reactivity(reaction_id, precursor, facts, issues)


def _audit_template_product_match(
    reaction_id: str,
    classification: str,
    product_facts: dict[str, Any],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    present = set(product_facts.get("broad_groups", []))
    expected = {
        "amide": ("amide",),
        "n-acylation": ("amide",),
        "ester": ("ester",),
        "hydrolysis": ("carboxylic_acid", "alcohol", "phenol"),
        "suzuki": ("arene", "halide"),
        "reductive amination": ("amine",),
    }
    for pattern, groups in expected.items():
        if pattern in classification:
            if any(group in present for group in groups):
                passed.append("classification_product_group_match")
            else:
                issues.append(_issue(
                    RiskSeverity.WARNING,
                    "mechanism",
                    f"{reaction_id}: classification '{classification}' does not match obvious product functional groups.",
                    "Do not trust this template blindly; run map_reaction, inspect the disconnection, or use fallback proposals.",
                ))
            return


def _audit_stereochemistry(
    reaction_id: str,
    product_facts: dict[str, Any],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    assigned = product_facts.get("assigned_stereocenters", 0)
    unassigned = product_facts.get("unassigned_stereocenters", 0)
    if unassigned:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "stereochemistry",
            f"{reaction_id}: product has {unassigned} unassigned stereocenter(s).",
            "Specify stereochemistry or design a stereochemical control/resolution experiment.",
        ))
    elif assigned:
        passed.append("stereochemistry_specified")


def _audit_polyfunctional_selectivity(
    reaction_id: str,
    classification: str,
    product_facts: dict[str, Any],
    issues: list[AuditIssue],
) -> None:
    present = set(product_facts.get("broad_groups", []))
    if len(present.intersection({"carboxylic_acid", "ester", "amide", "alcohol", "phenol", "amine"})) >= 3:
        issues.append(_issue(
            RiskSeverity.INFO,
            "chemoselectivity",
            f"{reaction_id}: product is polyfunctional and may need chemoselectivity/protection analysis.",
            "Use functional_groups and recommend_conditions to document compatibility and protection strategy.",
        ))
    if "hydrolysis" in classification and {"ester", "amide"}.issubset(present):
        issues.append(_issue(
            RiskSeverity.WARNING,
            "chemoselectivity",
            f"{reaction_id}: hydrolysis proposal contains both ester and amide motifs.",
            "Check whether conditions selectively transform the intended functional group.",
        ))


def _audit_precursor_reactivity(
    reaction_id: str,
    precursor: str,
    facts: dict[str, Any],
    issues: list[AuditIssue],
) -> None:
    alerts = set(facts.get("reactivity_alerts", []))
    if alerts:
        issues.append(_issue(
            RiskSeverity.INFO,
            "safety",
            f"{reaction_id}: precursor has reactive alert(s) {sorted(alerts)}: {precursor}",
            "Document handling, compatibility, and whether a milder precursor is available.",
        ))
    if facts.get("formal_charge", 0) != 0:
        issues.append(_issue(
            RiskSeverity.INFO,
            "formulation",
            f"{reaction_id}: precursor has nonzero formal charge: {precursor}",
            "Record counterion/salt form assumptions before treating stock status as definitive.",
        ))


def _audit_mapping_reports(
    mapping_reports: list[dict[str, Any]],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    for index, report in enumerate(mapping_reports, start=1):
        label = report.get("reaction_id") or f"mapping_{index}"
        if report.get("error"):
            issues.append(_issue(
                RiskSeverity.WARNING,
                "mechanism",
                f"{label}: atom mapping failed with error: {report.get('error')}",
                "Inspect reaction SMILES and avoid using this mapping as validation evidence.",
            ))
            continue
        confidence = report.get("confidence")
        if isinstance(confidence, (int, float)):
            if confidence < 0.5:
                issues.append(_issue(
                    RiskSeverity.WARNING,
                    "mechanism",
                    f"{label}: atom mapping confidence is low ({confidence:.3f}).",
                    "Treat mechanism support as weak and inspect bond changes manually.",
                ))
            else:
                passed.append("atom_mapping_confidence_present")


def _audit_condition_reports(
    condition_reports: list[dict[str, Any]],
    issues: list[AuditIssue],
    passed: list[str],
) -> None:
    for index, report in enumerate(condition_reports, start=1):
        label = report.get("reaction_id") or f"conditions_{index}"
        conditions = report.get("conditions", report)
        missing = [
            key for key in ("catalyst", "solvent", "temperature")
            if str(conditions.get(key, "")).lower() in {"", "to be determined", "unknown", "n/a"}
        ]
        if missing:
            issues.append(_issue(
                RiskSeverity.INFO,
                "conditions",
                f"{label}: condition report is incomplete ({', '.join(missing)}).",
                "Add literature or condition-screen support before treating this as a runnable experiment.",
            ))
        else:
            passed.append("conditions_specified")


def _audit_route_graph(
    route_graph: dict[str, Any],
    issues: list[AuditIssue],
    passed: list[str],
) -> dict[str, Any]:
    if not route_graph:
        return {"provided": False}

    molecules = route_graph.get("molecules", {}) or {}
    reactions = route_graph.get("reactions", {}) or {}
    product_to_reactions = route_graph.get("product_to_reactions", {}) or {}
    reaction_to_precursors = route_graph.get("reaction_to_precursors", {}) or {}
    target_id = route_graph.get("target_id", "")
    stats = route_graph.get("stats", {}) or {}
    unsolved_ids = _route_graph_unsolved_molecules(molecules, product_to_reactions)
    class_count = len({
        str(reaction.get("classification", "")).strip().lower()
        for reaction in reactions.values()
        if str(reaction.get("classification", "")).strip()
    })
    precursor_set_count = len({
        ".".join(sorted(reaction.get("precursor_smiles_list", [])))
        for reaction in reactions.values()
        if reaction.get("precursor_smiles_list")
    })
    max_precursor_count = max(
        (len(ids) for ids in reaction_to_precursors.values()),
        default=0,
    )
    target_reaction_count = len(product_to_reactions.get(target_id, [])) if target_id else 0
    protection_like = _count_protection_like_reactions(reactions)

    if target_id and target_reaction_count:
        passed.append("route_graph_target_has_reaction")
    else:
        issues.append(_issue(
            RiskSeverity.WARNING,
            "route_graph",
            "Route graph has no reaction attached to the target molecule.",
            "Use propose on the target or repair product_smiles so the target OR node has candidate AND reactions.",
        ))

    if unsolved_ids:
        preview = ", ".join(_molecule_label(molecules, mol_id) for mol_id in unsolved_ids[:5])
        severity = RiskSeverity.WARNING if len(unsolved_ids) > 3 else RiskSeverity.INFO
        issues.append(_issue(
            severity,
            "route_graph",
            f"Route graph still has {len(unsolved_ids)} unresolved molecule node(s): {preview}",
            "Recursively expand unresolved intermediates or justify supplier/common-reagent assumptions.",
        ))
    else:
        passed.append("route_graph_precursors_resolved")

    if len(reactions) >= 3 and class_count < 2 and precursor_set_count < 2:
        issues.append(_issue(
            RiskSeverity.INFO,
            "diversity",
            "Route graph has multiple reactions but little class or precursor-set diversity.",
            "Explore alternative disconnections, starting materials, late-stage functionalization, or protection logic.",
        ))
    elif class_count >= 2 or precursor_set_count >= 2:
        passed.append("route_graph_route_diversity_present")

    if len(reactions) >= 2 and max_precursor_count <= 1:
        issues.append(_issue(
            RiskSeverity.INFO,
            "convergence",
            "Route graph appears mostly linear with no multi-precursor convergent step.",
            "For discovery tasks, compare whether a convergent route could reduce step count or improve modularity.",
        ))
    elif max_precursor_count > 1:
        passed.append("route_graph_convergent_step_present")

    if protection_like >= 2:
        issues.append(_issue(
            RiskSeverity.INFO,
            "protecting_groups",
            f"Route graph contains {protection_like} protection/deprotection-like reaction classifications.",
            "Check whether protecting-group choreography is necessary or whether a more direct chemoselective route exists.",
        ))

    return {
        "provided": True,
        "target_reaction_count": target_reaction_count,
        "molecule_count": len(molecules) or stats.get("molecule_count", 0),
        "reaction_count": len(reactions) or stats.get("reaction_count", 0),
        "unsolved_molecule_count": len(unsolved_ids),
        "reaction_class_count": class_count,
        "precursor_set_count": precursor_set_count,
        "max_precursor_count": max_precursor_count,
        "protection_like_reaction_count": protection_like,
    }


def _route_graph_unsolved_molecules(
    molecules: dict[str, Any],
    product_to_reactions: dict[str, list[str]],
) -> list[str]:
    unsolved = []
    for mol_id, molecule in molecules.items():
        has_route = bool(product_to_reactions.get(mol_id))
        stock_status = molecule.get("stock_status", "unknown")
        in_stock = stock_status not in ("unknown", "", "not_in_stock")
        if not has_route and not in_stock:
            unsolved.append(mol_id)
    return unsolved


def _molecule_label(molecules: dict[str, Any], mol_id: str) -> str:
    molecule = molecules.get(mol_id, {})
    return molecule.get("canonical_smiles") or molecule.get("smiles") or mol_id


def _count_protection_like_reactions(reactions: dict[str, Any]) -> int:
    patterns = ("protection", "deprotection", "protect", "deprotect", "boc", "cbz", "silyl")
    count = 0
    for reaction in reactions.values():
        classification = str(reaction.get("classification", "")).lower()
        if any(pattern in classification for pattern in patterns):
            count += 1
    return count


def _molecule_facts(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return {"valid": False, "smiles": smiles}
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    broad = summarize_broad_functional_groups(mol)
    alerts = []
    alert_smarts = {
        "acyl_halide": "C(=O)[F,Cl,Br,I]",
        "aldehyde": "[CX3H1](=O)",
        "isocyanate": "N=C=O",
        "peroxide": "OO",
        "thiol": "[SX2H]",
        "nitro": "[N+](=O)[O-]",
    }
    for name, smarts in alert_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            alerts.append(name)
    return {
        "valid": True,
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "broad_groups": broad["present"],
        "reactivity_alerts": alerts,
        "assigned_stereocenters": sum(1 for _, label in chiral_centers if label != "?"),
        "unassigned_stereocenters": sum(1 for _, label in chiral_centers if label == "?"),
        "formal_charge": Chem.GetFormalCharge(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "h_bond_donors": Lipinski.NumHDonors(mol),
        "h_bond_acceptors": Lipinski.NumHAcceptors(mol),
    }


def _score_for_reaction(route_scores: dict[str, Any], reaction_id: str) -> float | None:
    raw = route_scores.get(reaction_id)
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        for key in ("total", "score", "feasibility", "probability"):
            if key in raw and isinstance(raw[key], (int, float)):
                return float(raw[key])
    return None


def _issue(
    severity: RiskSeverity,
    category: str,
    message: str,
    suggested_fix: str,
) -> AuditIssue:
    return AuditIssue(
        severity=severity,
        category=category,
        message=message,
        suggested_fix=suggested_fix,
        evidence=[
            Evidence(
                level=EvidenceLevel.TOOL_SUPPORTED,
                source="chemical_critic",
                summary="Deterministic completeness and consistency check.",
            )
        ],
    )

"""Multi-source route portfolio construction and independent validation gates."""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdChemReactions


DEFAULT_ROUTE_SOURCES = ("template", "template_free", "mechanism", "analogy")


class RoutePortfolioTool:
    name = "build_route_portfolio"
    description = (
        "Merge route candidates from template, template-free, mechanism, and analogy generators. "
        "Deduplicates chemically equivalent routes, records source support and score disagreement, "
        "and estimates transformation novelty/extrapolation distance against known reaction references."
    )

    def execute(self, parameters: dict) -> str:
        result = build_route_portfolio(
            target_smiles=str(parameters.get("target_smiles", "")),
            candidates=parameters.get("candidates", []),
            known_reactions=parameters.get("known_reactions", []),
            expected_sources=parameters.get("expected_sources", list(DEFAULT_ROUTE_SOURCES)),
        )
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "description": "Routes tagged with source, score, and structured reactions.",
                    "items": {"type": "object"},
                },
                "known_reactions": {
                    "type": "array",
                    "description": "Known reaction SMILES used only as a novelty reference set.",
                    "items": {"type": "string"},
                },
                "expected_sources": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["target_smiles", "candidates"],
        }


class IndependentRouteValidatorTool:
    name = "validate_route_portfolio"
    description = (
        "Independently validate route candidates without reusing generator scores. Checks molecular "
        "structure, target coverage, source diversity, atom-mapping evidence, forward predictions, "
        "condition completeness, and precursor stock evidence. Missing required evidence yields "
        "needs-evidence; contradictory evidence yields failed."
    )

    def execute(self, parameters: dict) -> str:
        result = validate_route_portfolio(
            target_smiles=str(parameters.get("target_smiles", "")),
            routes=parameters.get("routes", []),
            mapping_reports=parameters.get("mapping_reports", []),
            forward_predictions=parameters.get("forward_predictions", []),
            condition_reports=parameters.get("condition_reports", []),
            stock_hits=parameters.get("stock_hits", {}),
            required_checks=parameters.get("required_checks"),
            strict=bool(parameters.get("strict", False)),
        )
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string"},
                "routes": {"type": "array", "items": {"type": "object"}},
                "mapping_reports": {"type": "array", "items": {"type": "object"}},
                "forward_predictions": {"type": "array", "items": {"type": "object"}},
                "condition_reports": {"type": "array", "items": {"type": "object"}},
                "stock_hits": {"type": "object"},
                "required_checks": {"type": "array", "items": {"type": "string"}},
                "strict": {"type": "boolean"},
            },
            "required": ["target_smiles", "routes"],
        }


def build_route_portfolio(
    *,
    target_smiles: str,
    candidates: list[dict[str, Any]],
    known_reactions: list[str] | None = None,
    expected_sources: list[str] | tuple[str, ...] = DEFAULT_ROUTE_SOURCES,
) -> dict[str, Any]:
    target = _canonical_smiles(target_smiles)
    if not target:
        raise ValueError("target_smiles must be a valid molecule")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty array")
    source_universe = {str(item).strip() for item in expected_sources if str(item).strip()}
    if not source_universe:
        raise ValueError("expected_sources must contain at least one source")
    reference_fingerprints = [
        fingerprint for fingerprint in (_reaction_fingerprint(item) for item in known_reactions or [])
        if fingerprint is not None
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(candidates):
        normalized = _normalize_candidate(candidate, target, index)
        grouped.setdefault(normalized["signature"], []).append(normalized)

    routes = []
    for members in grouped.values():
        members.sort(key=lambda item: (item["id"], item["source"]))
        sources = sorted({item["source"] for item in members})
        scores = [item["score"] for item in members]
        mean_score = sum(scores) / len(scores)
        disagreement = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        source_coverage = min(1.0, len(set(sources) & source_universe) / len(source_universe))
        reactions = members[0]["reactions"]
        reaction_fingerprints = [
            fingerprint for fingerprint in (
                _reaction_fingerprint(reaction["reaction_smiles"]) for reaction in reactions
            ) if fingerprint is not None
        ]
        if reaction_fingerprints and reference_fingerprints:
            similarities = [
                max(DataStructs.TanimotoSimilarity(candidate_fp, reference_fp) for reference_fp in reference_fingerprints)
                for candidate_fp in reaction_fingerprints
            ]
            novelty = 1.0 - sum(similarities) / len(similarities)
            novelty_basis = "reaction_fingerprint_reference_distance"
        else:
            novelty = 0.5
            novelty_basis = "insufficient_reference_data"
        extrapolation = novelty
        uncertainty = _clamp(
            0.45 * extrapolation + 0.35 * min(1.0, disagreement) + 0.20 * (1.0 - source_coverage)
        )
        portfolio_score = _clamp(
            0.50 * mean_score
            + 0.20 * source_coverage
            + 0.15 * novelty
            + 0.15 * (1.0 - min(1.0, disagreement))
        )
        representative = members[0]
        routes.append({
            "id": representative["id"],
            "target_smiles": target,
            "reaction_ids": [reaction["id"] for reaction in reactions],
            "reactions": reactions,
            "candidate_ids": sorted({item["id"] for item in members}),
            "sources": sources,
            "source_count": len(sources),
            "source_coverage": round(source_coverage, 6),
            "source_scores": {item["source"]: round(item["score"], 6) for item in members},
            "model_disagreement": round(disagreement, 6),
            "novelty_score": round(novelty, 6),
            "novelty_basis": novelty_basis,
            "extrapolation_distance": round(extrapolation, 6),
            "scores": {
                "feasibility": round(mean_score, 6),
                "portfolio": round(portfolio_score, 6),
                "novelty": round(novelty, 6),
                "uncertainty": round(uncertainty, 6),
                "evidence_strength": round(source_coverage, 6),
            },
            "status": "proposed",
            "metadata": {
                "signature": representative["signature"],
                "generator_count": len(members),
            },
        })
    routes.sort(key=lambda item: (-item["scores"]["portfolio"], -item["novelty_score"], item["id"]))
    return {
        "routes": routes,
        "route_portfolio": {
            "candidate_count": len(candidates),
            "unique_route_count": len(routes),
            "sources_observed": sorted({source for route in routes for source in route["sources"]}),
            "expected_sources": sorted(source_universe),
            "multi_source_route_count": sum(route["source_count"] >= 2 for route in routes),
            "max_novelty_score": max(route["novelty_score"] for route in routes),
            "max_model_disagreement": max(route["model_disagreement"] for route in routes),
        },
    }


def validate_route_portfolio(
    *,
    target_smiles: str,
    routes: list[dict[str, Any]],
    mapping_reports: list[dict[str, Any]] | None = None,
    forward_predictions: list[dict[str, Any]] | None = None,
    condition_reports: list[dict[str, Any]] | None = None,
    stock_hits: dict[str, Any] | None = None,
    required_checks: list[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    target = _canonical_smiles(target_smiles)
    if not target:
        raise ValueError("target_smiles must be a valid molecule")
    if not isinstance(routes, list) or not routes:
        raise ValueError("routes must be a non-empty array")
    all_checks = {
        "structure", "target_coverage", "source_diversity", "atom_mapping",
        "forward_prediction", "conditions", "stock",
    }
    if required_checks is None:
        required = {"structure", "target_coverage"}
        if strict:
            required = set(all_checks)
    else:
        required = {str(item) for item in required_checks}
        unknown = sorted(required - all_checks)
        if unknown:
            raise ValueError(f"Unknown required checks: {', '.join(unknown)}")
    mappings = _index_reports(mapping_reports or [])
    forwards = _index_reports(forward_predictions or [])
    conditions = _index_reports(condition_reports or [])
    normalized_stock = _normalize_stock(stock_hits or {})

    reports = []
    validated_routes = []
    for index, route in enumerate(routes):
        route_id = str(route.get("id") or f"route_{index + 1}")
        reactions = route.get("reactions", [])
        checks = []
        checks.append(_check("structure", _structure_check(reactions), "Every reaction has valid product and precursor molecules."))
        final_product = _final_product(route, reactions)
        checks.append(_check(
            "target_coverage",
            "pass" if final_product == target else "fail",
            "Final route product matches the requested target." if final_product == target else "Final route product does not match target.",
        ))
        source_count = int(route.get("source_count", len(route.get("sources", []))) or 0)
        checks.append(_check(
            "source_diversity", "pass" if source_count >= 2 else "missing",
            f"Route has support from {source_count} independent generator source(s).",
        ))
        checks.append(_check(
            "atom_mapping", _mapping_check(reactions, mappings),
            "Every reaction has atom-mapping evidence with confidence >= 0.70.",
        ))
        checks.append(_check(
            "forward_prediction", _forward_check(reactions, forwards),
            "Every reaction has a matching forward prediction with confidence >= 0.50.",
        ))
        checks.append(_check(
            "conditions", _condition_check(reactions, conditions),
            "Every reaction has solvent, temperature, and reagent or catalyst conditions.",
        ))
        leaf_precursors = _leaf_precursors(reactions)
        checks.append(_check(
            "stock", _stock_check(leaf_precursors, normalized_stock),
            "Every route leaf has affirmative stock or availability evidence.",
        ))
        for check in checks:
            check["required"] = check["name"] in required
        failed = [check for check in checks if check["required"] and check["status"] == "fail"]
        missing = [check for check in checks if check["required"] and check["status"] == "missing"]
        if failed:
            status = "failed"
        elif missing:
            status = "needs-evidence"
        else:
            status = "passed"
        required_rows = [check for check in checks if check["required"]]
        confidence = (
            sum(check["status"] == "pass" for check in required_rows) / len(required_rows)
            if required_rows else 1.0
        )
        report = {
            "id": f"validation_{route_id}",
            "route_id": route_id,
            "status": status,
            "confidence": round(confidence, 6),
            "checks": checks,
            "issues": [check["message"] for check in checks if check["status"] != "pass"],
            "required_checks": sorted(required),
            "validator": "independent_route_validator_v1",
        }
        reports.append(report)
        validated_routes.append({
            **route,
            "id": route_id,
            "validation_status": status,
            "validation_confidence": round(confidence, 6),
            "validation_report_id": report["id"],
        })
    return {
        "validation_reports": reports,
        "routes": validated_routes,
        "summary": {
            "route_count": len(reports),
            "passed_count": sum(item["status"] == "passed" for item in reports),
            "failed_count": sum(item["status"] == "failed" for item in reports),
            "needs_evidence_count": sum(item["status"] == "needs-evidence" for item in reports),
            "strict": strict,
        },
    }


def _normalize_candidate(candidate: Any, target: str, index: int) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        raise ValueError(f"Candidate {index} must be an object")
    source = str(candidate.get("source", "")).strip()
    if not source:
        raise ValueError(f"Candidate {index} requires source")
    candidate_id = str(candidate.get("id") or f"route_candidate_{index + 1}")
    score = _probability(candidate.get("score", candidate.get("feasibility", 0.5)), f"candidate.{candidate_id}.score")
    raw_reactions = candidate.get("reactions", [])
    if not isinstance(raw_reactions, list) or not raw_reactions:
        raise ValueError(f"Candidate {candidate_id} requires reactions")
    reactions = []
    for reaction_index, reaction in enumerate(raw_reactions):
        if not isinstance(reaction, dict):
            raise ValueError(f"Candidate {candidate_id} reaction {reaction_index} must be an object")
        reaction_id = str(reaction.get("id") or f"{candidate_id}_rxn_{reaction_index + 1}")
        reaction_smiles = str(reaction.get("reaction_smiles", ""))
        precursors = reaction.get("precursor_smiles_list", reaction.get("precursors", []))
        product = str(reaction.get("product_smiles", reaction.get("product", "")))
        if reaction_smiles and ">>" in reaction_smiles:
            left, right = reaction_smiles.split(">>", 1)
            if not precursors:
                precursors = [item for item in left.split(".") if item]
            if not product:
                product = right
        if not isinstance(precursors, list) or not precursors:
            raise ValueError(f"Reaction {reaction_id} requires precursor_smiles_list")
        canonical_precursors = [_canonical_smiles(item) for item in precursors]
        canonical_product = _canonical_smiles(product)
        if not canonical_product or any(not item for item in canonical_precursors):
            raise ValueError(f"Reaction {reaction_id} contains invalid SMILES")
        canonical_reaction = f"{'.'.join(sorted(canonical_precursors))}>>{canonical_product}"
        reactions.append({
            **reaction,
            "id": reaction_id,
            "product_smiles": canonical_product,
            "precursor_smiles_list": canonical_precursors,
            "reaction_smiles": canonical_reaction,
        })
    if reactions[-1]["product_smiles"] != target:
        raise ValueError(f"Candidate {candidate_id} final product does not match target_smiles")
    signature = "||".join(sorted(reaction["reaction_smiles"] for reaction in reactions))
    return {
        "id": candidate_id,
        "source": source,
        "score": score,
        "reactions": reactions,
        "signature": signature,
    }


def _reaction_fingerprint(reaction_smiles: str):
    try:
        reaction = rdChemReactions.ReactionFromSmarts(str(reaction_smiles), useSmiles=True)
        if reaction is None:
            return None
        return rdChemReactions.CreateDifferenceFingerprintForReaction(reaction)
    except Exception:
        return None


def _canonical_smiles(smiles: Any) -> str:
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return ""
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else ""


def _index_reports(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        key = str(report.get("reaction_id") or report.get("id") or "")
        if key:
            indexed[key] = report
    return indexed


def _normalize_stock(stock_hits: dict[str, Any]) -> dict[str, bool]:
    result = {}
    for key, value in stock_hits.items():
        canonical = _canonical_smiles(key)
        if canonical:
            if isinstance(value, dict):
                result[canonical] = bool(value.get("in_stock", value.get("available", False)))
            else:
                result[canonical] = bool(value)
    return result


def _structure_check(reactions: Any) -> str:
    if not isinstance(reactions, list) or not reactions:
        return "missing"
    for reaction in reactions:
        if not isinstance(reaction, dict):
            return "fail"
        if not _canonical_smiles(reaction.get("product_smiles", "")):
            return "fail"
        precursors = reaction.get("precursor_smiles_list", [])
        if not isinstance(precursors, list) or not precursors:
            return "fail"
        if any(not _canonical_smiles(item) for item in precursors):
            return "fail"
    return "pass"


def _final_product(route: dict[str, Any], reactions: Any) -> str:
    if isinstance(reactions, list) and reactions:
        return _canonical_smiles(reactions[-1].get("product_smiles", ""))
    return _canonical_smiles(route.get("target_smiles", ""))


def _mapping_check(reactions: list[dict[str, Any]], mappings: dict[str, dict[str, Any]]) -> str:
    if not reactions:
        return "missing"
    for reaction in reactions:
        report = mappings.get(str(reaction.get("id", "")))
        if report is None:
            return "missing"
        confidence = float(report.get("confidence", 0) or 0)
        if confidence < 0.70 or not str(report.get("mapped_reaction", "")):
            return "fail"
    return "pass"


def _forward_check(reactions: list[dict[str, Any]], forwards: dict[str, dict[str, Any]]) -> str:
    if not reactions:
        return "missing"
    for reaction in reactions:
        report = forwards.get(str(reaction.get("id", "")))
        if report is None:
            return "missing"
        expected = _canonical_smiles(reaction.get("product_smiles", ""))
        predicted = _canonical_smiles(report.get("predicted_product", ""))
        if predicted != expected or float(report.get("confidence", 0) or 0) < 0.50:
            return "fail"
    return "pass"


def _condition_check(reactions: list[dict[str, Any]], conditions: dict[str, dict[str, Any]]) -> str:
    if not reactions:
        return "missing"
    for reaction in reactions:
        report = conditions.get(str(reaction.get("id", "")))
        if report is None:
            return "missing"
        if not report.get("solvent") or report.get("temperature") in {None, ""}:
            return "fail"
        if not (report.get("reagent") or report.get("reagents") or report.get("catalyst")):
            return "fail"
    return "pass"


def _leaf_precursors(reactions: list[dict[str, Any]]) -> set[str]:
    products = {_canonical_smiles(reaction.get("product_smiles", "")) for reaction in reactions}
    return {
        canonical
        for reaction in reactions
        for precursor in reaction.get("precursor_smiles_list", [])
        if (canonical := _canonical_smiles(precursor)) and canonical not in products
    }


def _stock_check(leaves: set[str], stock: dict[str, bool]) -> str:
    if not leaves:
        return "missing"
    if any(leaf not in stock for leaf in leaves):
        return "missing"
    return "pass" if all(stock[leaf] for leaf in leaves) else "fail"


def _check(name: str, status: str, message: str) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message}


def _probability(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0 or number > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))

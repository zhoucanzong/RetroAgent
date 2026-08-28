"""Information-gain experiment selection and evidence-driven belief updates."""

from __future__ import annotations

import json
import math
from typing import Any


class ExperimentSelectorTool:
    name = "select_experiment"
    description = (
        "Rank candidate experiments by expected information gain over competing hypotheses, "
        "then penalize time, material, compute, and risk. Use before requesting an expensive "
        "measurement. Returns the complete ranking and one selected experiment; it does not "
        "execute the experiment or observe its outcome."
    )

    def execute(self, parameters: dict) -> str:
        result = rank_experiments(
            hypotheses=parameters.get("hypotheses", []),
            experiments=parameters.get("experiments", []),
            cost_weight=float(parameters.get("cost_weight", 0.25)),
            risk_weight=float(parameters.get("risk_weight", 0.5)),
        )
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "description": "Competing hypotheses with id and optional posterior_probability.",
                    "items": {"type": "object"},
                },
                "experiments": {
                    "type": "array",
                    "description": (
                        "Candidate experiments. Each needs id and outcomes; every outcome contains "
                        "likelihoods P(outcome|hypothesis)."
                    ),
                    "items": {"type": "object"},
                },
                "cost_weight": {"type": "number", "minimum": 0},
                "risk_weight": {"type": "number", "minimum": 0},
            },
            "required": ["hypotheses", "experiments"],
        }


class BeliefUpdaterTool:
    name = "update_hypotheses"
    description = (
        "Update competing hypothesis probabilities from one observed outcome using Bayes' rule. "
        "The input likelihoods are P(observation|hypothesis). Returns normalized posterior updates, "
        "entropy reduction, and likelihood ratios suitable for ledger assimilation."
    )

    def execute(self, parameters: dict) -> str:
        result = update_hypothesis_probabilities(
            hypotheses=parameters.get("hypotheses", []),
            likelihoods=parameters.get("likelihoods", {}),
            experiment_id=str(parameters.get("experiment_id", "")),
            observation=str(parameters.get("observation", "")),
            evidence_id=str(parameters.get("evidence_id", "")),
        )
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "description": "Hypotheses with id and posterior_probability or prior_probability.",
                    "items": {"type": "object"},
                },
                "likelihoods": {
                    "type": "object",
                    "description": "Map hypothesis id to P(observation|hypothesis).",
                    "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "experiment_id": {"type": "string"},
                "observation": {"type": "string"},
                "evidence_id": {"type": "string"},
            },
            "required": ["hypotheses", "likelihoods", "experiment_id", "observation"],
        }


def rank_experiments(
    *,
    hypotheses: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    cost_weight: float = 0.25,
    risk_weight: float = 0.5,
) -> dict[str, Any]:
    hypothesis_ids, priors = _hypothesis_distribution(hypotheses)
    if len(hypothesis_ids) < 2:
        raise ValueError("Experiment selection requires at least two competing hypotheses")
    if not experiments:
        raise ValueError("Experiment selection requires at least one candidate experiment")
    if cost_weight < 0 or risk_weight < 0:
        raise ValueError("cost_weight and risk_weight must be non-negative")

    prior_entropy = _entropy(priors)
    ranking = []
    seen_ids: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict) or not str(experiment.get("id", "")).strip():
            raise ValueError("Every experiment requires a non-empty id")
        experiment_id = str(experiment["id"])
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)
        outcomes = _normalize_outcomes(experiment.get("outcomes"), hypothesis_ids)
        expected_entropy = 0.0
        outcome_predictions = []
        for outcome_id, likelihoods in outcomes:
            probability = sum(priors[index] * likelihoods[index] for index in range(len(priors)))
            if probability <= 0:
                posterior = list(priors)
            else:
                posterior = [
                    priors[index] * likelihoods[index] / probability
                    for index in range(len(priors))
                ]
            posterior_entropy = _entropy(posterior)
            expected_entropy += probability * posterior_entropy
            outcome_predictions.append({
                "outcome": outcome_id,
                "probability": round(probability, 8),
                "posterior": {
                    hypothesis_ids[index]: round(posterior[index], 8)
                    for index in range(len(hypothesis_ids))
                },
                "posterior_entropy": round(posterior_entropy, 8),
            })
        information_gain = max(0.0, prior_entropy - expected_entropy)
        cost = _cost_score(experiment.get("cost", {}))
        risk = _bounded_probability(experiment.get("risk_score", 0.0), "risk_score")
        utility = information_gain / (1.0 + cost_weight * cost + risk_weight * risk)
        ranking.append({
            "id": experiment_id,
            "question": str(experiment.get("question", "")),
            "expected_information_gain": round(information_gain, 8),
            "expected_posterior_entropy": round(expected_entropy, 8),
            "cost_score": round(cost, 8),
            "risk_score": round(risk, 8),
            "utility": round(utility, 8),
            "outcome_predictions": outcome_predictions,
            "experiment": dict(experiment),
        })
    ranking.sort(key=lambda item: (-item["utility"], -item["expected_information_gain"], item["id"]))
    for index, item in enumerate(ranking, 1):
        item["rank"] = index
    selected = dict(ranking[0])
    return {
        "prior_entropy": round(prior_entropy, 8),
        "hypothesis_priors": dict(zip(hypothesis_ids, (round(value, 8) for value in priors))),
        "ranking": ranking,
        "selected_experiment": selected,
        "selection_criterion": "expected_information_gain_adjusted_for_cost_and_risk",
    }


def update_hypothesis_probabilities(
    *,
    hypotheses: list[dict[str, Any]],
    likelihoods: dict[str, Any],
    experiment_id: str,
    observation: str,
    evidence_id: str = "",
) -> dict[str, Any]:
    hypothesis_ids, priors = _hypothesis_distribution(hypotheses)
    if not experiment_id.strip() or not observation.strip():
        raise ValueError("experiment_id and observation are required")
    if not isinstance(likelihoods, dict):
        raise ValueError("likelihoods must be an object")
    missing = [hypothesis_id for hypothesis_id in hypothesis_ids if hypothesis_id not in likelihoods]
    if missing:
        raise ValueError(f"Missing observation likelihoods for: {', '.join(missing)}")
    normalized_likelihoods = [
        _bounded_probability(likelihoods[hypothesis_id], f"likelihoods.{hypothesis_id}")
        for hypothesis_id in hypothesis_ids
    ]
    evidence_probability = sum(
        priors[index] * normalized_likelihoods[index]
        for index in range(len(priors))
    )
    if evidence_probability <= 0:
        raise ValueError("Observation has zero probability under every hypothesis")
    posteriors = [
        priors[index] * normalized_likelihoods[index] / evidence_probability
        for index in range(len(priors))
    ]
    prior_entropy = _entropy(priors)
    posterior_entropy = _entropy(posteriors)
    strongest = max(posteriors)
    updates = []
    for index, hypothesis_id in enumerate(hypothesis_ids):
        alternatives = [
            normalized_likelihoods[other]
            for other in range(len(hypothesis_ids))
            if other != index
        ]
        alternative_likelihood = sum(alternatives) / len(alternatives) if alternatives else 1.0
        likelihood_ratio = (
            normalized_likelihoods[index] / alternative_likelihood
            if alternative_likelihood > 0 else math.inf
        )
        updates.append({
            "id": hypothesis_id,
            "prior_probability": round(priors[index], 8),
            "observation_likelihood": round(normalized_likelihoods[index], 8),
            "posterior_probability": round(posteriors[index], 8),
            "likelihood_ratio_vs_alternatives": (
                "infinity" if math.isinf(likelihood_ratio) else round(likelihood_ratio, 8)
            ),
            "experiment_id": experiment_id,
            "observation": observation,
            "evidence_id": evidence_id,
        })
    return {
        "experiment_id": experiment_id,
        "observation": observation,
        "evidence_probability": round(evidence_probability, 8),
        "prior_entropy": round(prior_entropy, 8),
        "posterior_entropy": round(posterior_entropy, 8),
        "entropy_reduction": round(prior_entropy - posterior_entropy, 8),
        "max_posterior_probability": round(strongest, 8),
        "hypothesis_updates": updates,
    }


def _hypothesis_distribution(hypotheses: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("hypotheses must be a non-empty array")
    ids = []
    raw = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict) or not str(hypothesis.get("id", "")).strip():
            raise ValueError("Every hypothesis requires a non-empty id")
        hypothesis_id = str(hypothesis["id"])
        if hypothesis_id in ids:
            raise ValueError(f"Duplicate hypothesis id: {hypothesis_id}")
        ids.append(hypothesis_id)
        value = hypothesis.get("posterior_probability", hypothesis.get("prior_probability"))
        raw.append(None if value is None else _bounded_probability(value, f"probability.{hypothesis_id}"))
    if all(value is None for value in raw):
        return ids, [1.0 / len(ids)] * len(ids)
    if any(value is None for value in raw):
        raise ValueError("Hypothesis probabilities must be provided for all hypotheses or none")
    total = sum(float(value) for value in raw)
    if total <= 0:
        raise ValueError("Hypothesis probabilities must sum to a positive value")
    return ids, [float(value) / total for value in raw]


def _normalize_outcomes(raw: Any, hypothesis_ids: list[str]) -> list[tuple[str, list[float]]]:
    if isinstance(raw, dict):
        records = [{"id": key, "likelihoods": value} for key, value in raw.items()]
    elif isinstance(raw, list):
        records = raw
    else:
        raise ValueError("Experiment outcomes must be an object or array")
    if len(records) < 2:
        raise ValueError("Every experiment requires at least two possible outcomes")
    outcomes = []
    columns = {hypothesis_id: 0.0 for hypothesis_id in hypothesis_ids}
    for record in records:
        if not isinstance(record, dict) or not str(record.get("id", "")).strip():
            raise ValueError("Every outcome requires a non-empty id")
        likelihood_map = record.get("likelihoods")
        if not isinstance(likelihood_map, dict):
            raise ValueError(f"Outcome {record['id']} requires likelihoods")
        missing = [hypothesis_id for hypothesis_id in hypothesis_ids if hypothesis_id not in likelihood_map]
        if missing:
            raise ValueError(f"Outcome {record['id']} missing likelihoods for: {', '.join(missing)}")
        values = []
        for hypothesis_id in hypothesis_ids:
            value = _bounded_probability(likelihood_map[hypothesis_id], f"outcome.{record['id']}.{hypothesis_id}")
            values.append(value)
            columns[hypothesis_id] += value
        outcomes.append((str(record["id"]), values))
    invalid = [key for key, total in columns.items() if not math.isclose(total, 1.0, abs_tol=1e-6)]
    if invalid:
        raise ValueError(f"Outcome likelihoods must sum to 1 for each hypothesis: {', '.join(invalid)}")
    return outcomes


def _cost_score(raw: Any) -> float:
    if raw is None:
        return 0.0
    if not isinstance(raw, dict):
        raise ValueError("experiment cost must be an object")
    time_minutes = _nonnegative(raw.get("time_minutes", 0), "cost.time_minutes")
    material_cost = _nonnegative(raw.get("material_cost", 0), "cost.material_cost")
    compute_units = _nonnegative(raw.get("compute_units", 0), "cost.compute_units")
    return math.log1p(time_minutes / 60.0 + material_cost / 100.0 + compute_units)


def _entropy(probabilities: list[float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities if value > 0)


def _bounded_probability(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0 or number > 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def _nonnegative(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be non-negative")
    return number

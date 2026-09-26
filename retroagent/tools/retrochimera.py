"""RetroChimera adapter and model-agnostic rank-weighted disconnection fusion."""

from __future__ import annotations

import json
import hashlib
import math
import os
import statistics
import sys
import threading
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any, Callable

from rdkit import Chem


_OUTPUT_LOCK = threading.RLock()


@contextmanager
def _diagnostic_output():
    # Upstream template workers inherit fd 1, while tqdm uses Python sys.stdout.
    # Redirect both so model startup cannot corrupt a CLI/MCP JSON response.
    with _OUTPUT_LOCK:
        saved = None
        try:
            try:
                stdout_fd, stderr_fd = sys.stdout.fileno(), sys.stderr.fileno()
                sys.stdout.flush()
                saved = os.dup(stdout_fd)
                os.dup2(stderr_fd, stdout_fd)
            except (AttributeError, OSError, ValueError):
                pass
            with redirect_stdout(sys.stderr):
                yield
        finally:
            if saved is not None:
                os.dup2(saved, stdout_fd)
                os.close(saved)


class DisconnectionEnsembleTool:
    name = "ensemble_disconnections"
    description = (
        "Fuse ranked one-step retrosynthesis predictions from models, databases, or human input "
        "after canonicalizing reactant sets. Scores are summed from source-and-rank weights, so "
        "agreement between complementary sources is rewarded. Supports calibrated weight profiles "
        "and consensus-only filtering. Ensemble score is a ranking signal, not feasibility evidence."
    )

    def execute(self, parameters: dict) -> str:
        result = ensemble_disconnections(
            target_smiles=str(parameters.get("target_smiles", "")),
            sources=parameters.get("sources", []),
            rank_weights=parameters.get("rank_weights"),
            consensus_only=bool(parameters.get("consensus_only", False)),
            min_sources=int(parameters.get("min_sources", 1)),
            max_results=int(parameters.get("max_results", 10)),
            score_temperature=float(parameters.get("score_temperature", 8.0)),
        )
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string"},
                "sources": {
                    "type": "array",
                    "description": (
                        "Ranked prediction lists. Each item needs a unique name and predictions; "
                        "each prediction needs precursor_smiles_list, reactants, or precursors."
                    ),
                    "items": {"type": "object"},
                },
                "rank_weights": {
                    "type": "object",
                    "description": (
                        "Optional calibrated source-to-rank weight arrays. Arrays must be "
                        "non-negative, non-increasing, and convex."
                    ),
                },
                "consensus_only": {
                    "type": "boolean",
                    "description": "Keep only reactant sets proposed by every source.",
                },
                "min_sources": {"type": "integer", "minimum": 1},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
                "score_temperature": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["target_smiles", "sources"],
        }


class RetroChimeraPredictionTool:
    name = "retrochimera_predict"
    model_class = "RetroChimeraModel"
    component_classes: tuple[str, ...] = ()
    description = (
        "Run an optional Microsoft RetroChimera checkpoint for one-step retrosynthesis. "
        "Returns canonical reaction records with constituent-model ranks and probabilities. "
        "Requires the optional retrochimera package and a model directory; predictions must still "
        "be independently checked for feasibility, conditions, mapping, and availability."
    )

    def __init__(
        self,
        *,
        model_factory: Callable[..., Any] | None = None,
        molecule_factory: Callable[[str], Any] | None = None,
        model_cache: dict[tuple[Any, ...], Any] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._molecule_factory = molecule_factory
        self._models = model_cache if model_cache is not None else {}

    def execute(self, parameters: dict) -> str:
        with _diagnostic_output():
            return self._execute(parameters)

    def _execute(self, parameters: dict) -> str:
        target = _canonical_smiles(parameters.get("target_smiles", ""))
        if not target:
            raise ValueError("target_smiles must be a valid molecule")
        raw_model_dir = str(
            parameters.get("model_dir") or os.environ.get("RETROCHIMERA_MODEL_DIR", "")
        ).strip()
        model_dir = Path(raw_model_dir).expanduser()
        if not raw_model_dir or not model_dir.is_dir():
            raise ValueError(
                "RetroChimera model directory is required; pass model_dir or set "
                "RETROCHIMERA_MODEL_DIR"
            )
        model_dir = self._component_directory(model_dir).resolve()
        num_results = int(parameters.get("num_results", 5))
        if num_results < 1 or num_results > 10:
            raise ValueError("num_results must be between 1 and 10 before independent filtering")
        device = str(parameters.get("device", "cpu"))
        consensus_only = bool(parameters.get("consensus_only", False))
        parallel = bool(parameters.get("call_submodels_in_parallel", device.startswith("cuda")))
        submodel_kwargs = parameters.get("submodel_kwargs", {})
        if not isinstance(submodel_kwargs, dict) or any(
            not isinstance(value, dict) for value in submodel_kwargs.values()
        ):
            raise ValueError("submodel_kwargs must map checkpoint submodel names to option objects")
        if submodel_kwargs:
            model_names = set(json.loads((model_dir / "models.json").read_text()))
            if set(submodel_kwargs) - model_names:
                raise ValueError("submodel_kwargs contains names absent from checkpoint models.json")
        model_factory, molecule_factory = self._resolve_factories()
        if self.component_classes:
            if consensus_only or submodel_kwargs or "call_submodels_in_parallel" in parameters:
                raise ValueError("Single-model tools use model_kwargs; consensus and parallel-submodel options require retrochimera_predict")
            options = parameters.get("model_kwargs", {})
            if not isinstance(options, dict) or set(options) & {"model_dir", "device"}:
                raise ValueError("model_kwargs must be an object without model_dir or device")
            options = dict(options)
            if device == "cpu" and self.model_class == "RetroChimeraEditModel":
                options.setdefault("num_processes", 2)
        else:
            submodel_kwargs = {name: dict(value) for name, value in submodel_kwargs.items()}
            manifest = model_dir / "models.json"
            if device == "cpu" and manifest.exists():
                for name, entry in json.loads(manifest.read_text()).items():
                    if entry[0] in NeuralLocPredictionTool.component_classes:
                        submodel_kwargs.setdefault(name, {}).setdefault("num_processes", 2)
            options = {"consensus_only": consensus_only,
                       "call_submodels_in_parallel": parallel, **submodel_kwargs}
        key = (self.model_class, str(model_dir), device, json.dumps(options, sort_keys=True))
        if key not in self._models:
            self.close()
            self._models[key] = model_factory(
                model_dir=model_dir,
                device=device,
                **options,
            )
        context = nullcontext()
        if self._model_factory is None:
            import torch
            context = torch.no_grad()
        with context:
            predictions = self._models[key]([molecule_factory(target)], num_results=num_results)[0]
        reactions = []
        for index, prediction in enumerate(predictions, 1):
            precursors = sorted(
                _canonical_smiles(getattr(molecule, "smiles", str(molecule)))
                for molecule in prediction.reactants
            )
            if not precursors or any(not precursor for precursor in precursors):
                continue
            metadata = _json_value(dict(getattr(prediction, "metadata", {}) or {}))
            individual_ranks = metadata.get("individual_ranks", {}) if not self.component_classes else {model_dir.name: index - 1}
            supporting_sources = sorted(
                str(source) for source, rank in individual_ranks.items() if rank is not None
            )
            source_ids = ([str(model_dir)] if self.component_classes else
                          [str((model_dir / name).resolve()) for name in supporting_sources] or [str(model_dir)])
            reactions.append({
                "id": _reaction_id(self.name, target, precursors),
                "product_smiles": target,
                "precursor_smiles_list": precursors,
                "reaction_smiles": f"{'.'.join(precursors)}>>{target}",
                "rank": index,
                "score": float(metadata.get("score", 0.0) or 0.0),
                "probability": float(metadata.get("probability", 0.0) or 0.0),
                "source": self.name,
                "sources": supporting_sources or [self.name],
                "source_ids": source_ids,
                "rank_base": 1,
                "individual_rank_base": 0,
                "individual_ranks": individual_ranks,
                "source_metadata": metadata.get("individual_metadata", {}),
                "metadata": {
                    "adapter": "retrochimera>=1.3",
                    "consensus_only": consensus_only,
                    "raw": metadata,
                },
            })
        source_ids = sorted({source for reaction in reactions for source in reaction["source_ids"]})
        if not self.component_classes and (model_dir / "models.json").exists():
            source_ids = sorted(str((model_dir / name).resolve()) for name in json.loads((model_dir / "models.json").read_text()))
        elif self.component_classes:
            source_ids = [str(model_dir)]
        return json.dumps({
            "reactions": reactions,
            "count": len(reactions),
            "target_smiles": target,
            "consensus_only": consensus_only,
            "model_dir": str(model_dir.resolve()),
            "tool": self.name,
            "inference_options": {"device": device, "model_options": options},
            "ensemble_source": {"name": self.name, "source_ids": source_ids, "predictions": reactions},
            "warning": (
                "RetroChimera rankings are proposal signals and require independent chemical "
                "validation; lower-ranked and out-of-distribution predictions may hallucinate."
            ),
        }, ensure_ascii=False)

    def _component_directory(self, model_dir: Path) -> Path:
        manifest = model_dir / "models.json"
        if not self.component_classes or not manifest.exists():
            return model_dir
        matches = [name for name, entry in json.loads(manifest.read_text()).items()
                   if entry[0] in self.component_classes]
        if len(matches) != 1:
            raise ValueError(f"Expected one {self.model_class} component in models.json")
        component = (model_dir / matches[0]).resolve()
        if not component.is_relative_to(model_dir.resolve()) or not component.is_dir():
            raise ValueError("Invalid checkpoint component directory")
        return component

    def close(self) -> None:
        """Release template workers when a tool or research run is finished."""
        for model in self._models.values():
            for component in [model, *getattr(model, "_models", [])]:
                server = getattr(component, "_server", None)
                if server is not None:
                    server.close()
        self._models.clear()

    def _resolve_factories(self) -> tuple[Callable[..., Any], Callable[[str], Any]]:
        if self._model_factory is not None and self._molecule_factory is not None:
            return self._model_factory, self._molecule_factory
        try:
            import retrochimera
            from syntheseus import Molecule
        except ImportError as exc:
            raise RuntimeError(
                "RetroChimera integration is optional; install RetroAgent with "
                "the 'retrochimera' extra"
            ) from exc
        return self._model_factory or getattr(retrochimera, self.model_class), self._molecule_factory or Molecule

    def parameters_schema(self) -> dict:
        schema = {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string"},
                "model_dir": {
                    "type": "string",
                    "description": "Checkpoint directory; defaults to RETROCHIMERA_MODEL_DIR.",
                },
                "num_results": {"type": "integer", "minimum": 1, "maximum": 10},
                "device": {"type": "string"},
                "consensus_only": {"type": "boolean"},
                "call_submodels_in_parallel": {"type": "boolean"},
                "submodel_kwargs": {
                    "type": "object",
                    "description": "Options keyed by checkpoint models.json submodel names, e.g. num_processes, beam_size, augmentation_size.",
                    "additionalProperties": {"type": "object"},
                },
            },
            "required": ["target_smiles"],
        }
        if self.component_classes:
            for key in ("consensus_only", "call_submodels_in_parallel", "submodel_kwargs"):
                schema["properties"].pop(key)
            schema["properties"]["model_kwargs"] = {
                "type": "object", "description": "Upstream options for this submodel only. "
                "NeuralLoc: num_processes; Transformer: beam_size, augmentation_size.",
            }
            schema["properties"]["model_dir"]["description"] += " Accepts ensemble root or the component directory."
        return schema


class NeuralLocPredictionTool(RetroChimeraPredictionTool):
    name = "neuralloc_predict"
    model_class = "RetroChimeraEditModel"
    component_classes = ("TemplateLocalizationModel", "RetroChimeraEditModel")
    description = (
        "Call NeuralLoc alone for template-based retrosynthesis with reaction-site localization. "
        "Use when you want edit-based proposals without running the Transformer. Returns an "
        "ensemble_source for comparison or fusion; the Agent chooses what to validate next."
    )


class TransformerPredictionTool(RetroChimeraPredictionTool):
    name = "transformer_predict"
    model_class = "RetroChimeraDeNovoModel"
    component_classes = ("SmilesTransformerModel", "RetroChimeraDeNovoModel")
    description = (
        "Call the R-SMILES 2 Transformer alone to generate precursor SMILES de novo. "
        "Use for complementary proposals beyond template matching. This is a chemistry "
        "prediction model, not the Agent's LLM. Returns an ensemble_source for optional fusion."
    )


def ensemble_disconnections(
    *,
    target_smiles: str,
    sources: list[dict[str, Any]],
    rank_weights: dict[str, Any] | None = None,
    consensus_only: bool = False,
    min_sources: int = 1,
    max_results: int = 10,
    score_temperature: float = 8.0,
) -> dict[str, Any]:
    target = _canonical_smiles(target_smiles)
    if not target:
        raise ValueError("target_smiles must be a valid molecule")
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("At least two ranked prediction sources are required")
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be between 1 and 50")
    if not math.isfinite(score_temperature) or score_temperature <= 0:
        raise ValueError("score_temperature must be positive")
    names = [str(source.get("name", "")).strip() for source in sources if isinstance(source, dict)]
    if len(names) != len(sources) or any(not name for name in names):
        raise ValueError("Every source requires a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique")
    identities = {}
    seen_identities: set[str] = set()
    for source, name in zip(sources, names):
        predictions = source.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError(f"Source {name} predictions must be an array")
        inferred = set()
        for prediction in predictions:
            if isinstance(prediction, dict) and "source_ids" in prediction:
                inferred.update(_source_ids(prediction["source_ids"]))
        ids = _source_ids(source["source_ids"]) if "source_ids" in source else inferred or {name}
        if not inferred.issubset(ids):
            raise ValueError(f"Source {name} omits prediction provenance from source_ids")
        if ids & seen_identities:
            raise ValueError("Overlapping source_ids: do not fuse an ensemble with its own submodels or duplicate calls")
        seen_identities.update(ids)
        identities[name] = ids
    if min_sources < 1 or min_sources > len(sources):
        raise ValueError("min_sources must be between 1 and the number of sources")
    required_sources = len(sources) if consensus_only else min_sources
    weights = _weight_profile(names, rank_weights)

    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    source_prediction_counts: dict[str, int] = {}
    for source, source_name in zip(sources, names):
        predictions = source.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError(f"Source {source_name} predictions must be an array")
        source_prediction_counts[source_name] = len(predictions)
        best_for_source: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}
        for position, prediction in enumerate(predictions, 1):
            if not isinstance(prediction, dict):
                raise ValueError(f"Source {source_name} prediction {position} must be an object")
            if prediction.get("product_smiles") and _canonical_smiles(prediction["product_smiles"]) != target:
                raise ValueError("Prediction product does not match target_smiles")
            rank = int(prediction.get("rank", position))
            if rank < 1:
                raise ValueError(f"Source {source_name} ranks must be positive")
            reactants = _prediction_reactants(prediction)
            current = best_for_source.get(reactants)
            if current is None or rank < current[0]:
                best_for_source[reactants] = (rank, prediction)
        for reactants, (rank, prediction) in best_for_source.items():
            row = grouped.setdefault(reactants, {
                "individual_ranks": {},
                "individual_scores": {},
                "source_metadata": {},
            })
            row["individual_ranks"][source_name] = rank
            row["individual_scores"][source_name] = prediction.get("score")
            row["source_metadata"][source_name] = prediction.get("metadata", {})

    candidates = []
    for reactants, row in grouped.items():
        source_count = len(row["individual_ranks"])
        if source_count < required_sources:
            continue
        score = sum(
            _rank_weight(weights[source_name], rank)
            for source_name, rank in row["individual_ranks"].items()
        )
        ranks = list(row["individual_ranks"].values())
        candidates.append({
            "precursor_smiles_list": list(reactants),
            "reaction_smiles": f"{'.'.join(reactants)}>>{target}",
            "ensemble_score": score,
            "source_count": source_count,
            "sources": sorted(row["individual_ranks"]),
            "source_ids": sorted({identity for name in row["individual_ranks"] for identity in identities[name]}),
            "individual_ranks": row["individual_ranks"],
            "individual_scores": row["individual_scores"],
            "source_metadata": row["source_metadata"],
            "rank_disagreement": statistics.pstdev(ranks) if len(ranks) > 1 else 0.0,
            "unanimous": source_count == len(sources),
        })
    candidates.sort(key=lambda item: (
        -item["ensemble_score"],
        -item["source_count"],
        tuple(item["precursor_smiles_list"]),
    ))
    candidates = candidates[:max_results]
    probabilities = _score_probabilities(candidates, weights, score_temperature)
    reactions = []
    for index, (candidate, probability) in enumerate(zip(candidates, probabilities), 1):
        reactions.append({
            "id": _reaction_id("ensemble", target, candidate["precursor_smiles_list"]),
            "rank": index,
            "rank_base": 1,
            "individual_rank_base": 1,
            "product_smiles": target,
            **candidate,
            "score": round(candidate["ensemble_score"], 8),
            "ensemble_score": round(candidate["ensemble_score"], 8),
            "probability": round(probability, 8),
            "rank_disagreement": round(candidate["rank_disagreement"], 8),
            "source": "rank_weighted_ensemble",
            "metadata": {
                "ranking_method": "source_rank_weight_sum",
                "calibrated_weights": rank_weights is not None,
                "feasibility_validated": False,
            },
        })
    warnings = [
        "Ensemble scores rank agreement; they do not establish reaction feasibility."
    ]
    if rank_weights is None:
        warnings.append(
            "No calibrated rank_weights were supplied; reciprocal-rank fallback weights were used."
        )
    if max_results > 10:
        warnings.append(
            "More than 10 candidates were requested; apply stringent independent filtering."
        )
    return {
        "reactions": reactions,
        "count": len(reactions),
        "ensemble_source": {"name": "ensemble_disconnections", "source_ids": sorted(seen_identities), "predictions": reactions},
        "ensemble_summary": {
            "method": "retrochimera_inspired_rank_weighted_consensus",
            "source_names": names,
            "source_prediction_counts": source_prediction_counts,
            "unique_reactant_sets": len(grouped),
            "returned_count": len(reactions),
            "consensus_only": consensus_only,
            "minimum_sources": required_sources,
            "calibrated_weights": rank_weights is not None,
            "rank_weights": weights,
        },
        "warnings": warnings,
    }


def _weight_profile(names: list[str], profile: dict[str, Any] | None) -> dict[str, list[float]]:
    if profile is not None and not isinstance(profile, dict):
        raise ValueError("rank_weights must be an object")
    unknown = sorted(set(profile or {}) - set(names))
    if unknown:
        raise ValueError(f"rank_weights has unknown sources: {', '.join(unknown)}")
    if profile is not None:
        missing = sorted(set(names) - set(profile))
        if missing:
            raise ValueError(f"rank_weights is missing sources: {', '.join(missing)}")
    result = {}
    for name in names:
        raw = (profile or {}).get(name)
        if raw is None:
            result[name] = [1.0 / rank for rank in range(1, 11)]
            continue
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"rank_weights.{name} must be a non-empty array")
        values = [float(value) for value in raw]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"rank_weights.{name} must contain finite non-negative values")
        if values[0] <= 0 or any(left < right for left, right in zip(values, values[1:])):
            raise ValueError(f"rank_weights.{name} must be positive at rank 1 and non-increasing")
        if any(
            values[index] - 2 * values[index + 1] + values[index + 2] < -1e-12
            for index in range(len(values) - 2)
        ):
            raise ValueError(f"rank_weights.{name} must be convex")
        result[name] = values
    return result


def _rank_weight(weights: list[float], rank: int) -> float:
    if rank <= len(weights):
        return weights[rank - 1]
    return weights[-1] / (rank - len(weights) + 1)


def _score_probabilities(
    candidates: list[dict[str, Any]],
    weights: dict[str, list[float]],
    temperature: float,
) -> list[float]:
    if not candidates:
        return []
    maximum_score = sum(values[0] for values in weights.values())
    logits = [candidate["ensemble_score"] * temperature / maximum_score for candidate in candidates]
    offset = max(logits)
    exponentials = [math.exp(logit - offset) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _prediction_reactants(prediction: dict[str, Any]) -> tuple[str, ...]:
    raw = prediction.get(
        "precursor_smiles_list",
        prediction.get("reactants", prediction.get("precursors")),
    )
    if isinstance(raw, str):
        raw = [item for item in raw.split(".") if item]
    if not isinstance(raw, list) or not raw:
        raise ValueError("Every prediction requires a non-empty reactant set")
    canonical = tuple(sorted(_canonical_smiles(item) for item in raw))
    if any(not item for item in canonical):
        raise ValueError("Prediction contains invalid reactant SMILES")
    return canonical


def _canonical_smiles(smiles: Any) -> str:
    try:
        molecule = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return ""
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True)


def _source_ids(value: Any) -> set[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError("source_ids must be a non-empty array of identity strings")
    return set(value)


def _reaction_id(source: str, target: str, precursors: list[str]) -> str:
    identity = json.dumps([source, target, sorted(precursors)], separators=(",", ":"))
    return "rxn_" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)

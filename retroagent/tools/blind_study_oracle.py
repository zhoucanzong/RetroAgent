"""Withhold recent-study measurements until a blind benchmark agent requests them."""

from __future__ import annotations

import json
from typing import Any


class BlindStudyOracleTool:
    name = "blind_study_oracle"

    def __init__(self, experiments: dict[str, dict[str, Any]]):
        self.experiments = {str(key): dict(value) for key, value in experiments.items()}
        available = "; ".join(
            f"{experiment_id}: {experiment.get('question', '')}"
            for experiment_id, experiment in self.experiments.items()
        )
        self.description = (
            "Request one withheld measurement from a recent blind-study benchmark. "
            "The paper conclusion and product identity remain hidden. Available measurements: "
            f"{available}"
        )

    def execute(self, parameters: dict) -> str:
        experiment_id = str(parameters.get("experiment_id", ""))
        experiment = self.experiments.get(experiment_id)
        if experiment is None:
            return json.dumps({
                "error": f"Experiment is not registered: {experiment_id}",
                "available_experiment_ids": sorted(self.experiments),
            }, ensure_ascii=True, sort_keys=True)
        return json.dumps({
            "experiment_id": experiment_id,
            "question": str(experiment.get("question", "")),
            "result": experiment.get("result"),
        }, ensure_ascii=True, sort_keys=True)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "enum": sorted(self.experiments),
                    "description": "Identifier of the single measurement to reveal.",
                },
            },
            "required": ["experiment_id"],
        }

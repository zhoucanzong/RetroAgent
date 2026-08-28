import json
import math
import unittest

from retroagent.tools.active_learning import (
    BeliefUpdaterTool,
    ExperimentSelectorTool,
    rank_experiments,
    update_hypothesis_probabilities,
)


HYPOTHESES = [
    {"id": "H1", "posterior_probability": 0.5},
    {"id": "H2", "posterior_probability": 0.5},
]


class ActiveLearningTests(unittest.TestCase):
    def test_selector_prefers_discriminating_experiment_over_uninformative_one(self):
        result = rank_experiments(
            hypotheses=HYPOTHESES,
            experiments=[
                {
                    "id": "uninformative",
                    "outcomes": {
                        "positive": {"H1": 0.5, "H2": 0.5},
                        "negative": {"H1": 0.5, "H2": 0.5},
                    },
                    "cost": {"time_minutes": 1},
                },
                {
                    "id": "discriminating",
                    "outcomes": {
                        "positive": {"H1": 0.95, "H2": 0.05},
                        "negative": {"H1": 0.05, "H2": 0.95},
                    },
                    "cost": {"time_minutes": 30},
                },
            ],
        )

        self.assertEqual(result["selected_experiment"]["id"], "discriminating")
        self.assertGreater(result["ranking"][0]["expected_information_gain"], 0.7)
        self.assertEqual(result["ranking"][1]["expected_information_gain"], 0.0)
        self.assertEqual(result["prior_entropy"], 1.0)

    def test_selector_penalizes_risk_when_information_gain_is_equal(self):
        common_outcomes = {
            "positive": {"H1": 0.9, "H2": 0.1},
            "negative": {"H1": 0.1, "H2": 0.9},
        }
        result = rank_experiments(
            hypotheses=HYPOTHESES,
            experiments=[
                {"id": "high_risk", "outcomes": common_outcomes, "risk_score": 0.9},
                {"id": "low_risk", "outcomes": common_outcomes, "risk_score": 0.1},
            ],
        )

        self.assertEqual(result["selected_experiment"]["id"], "low_risk")

    def test_selector_rejects_non_normalized_outcome_model(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            rank_experiments(
                hypotheses=HYPOTHESES,
                experiments=[{
                    "id": "broken",
                    "outcomes": {
                        "a": {"H1": 0.8, "H2": 0.5},
                        "b": {"H1": 0.8, "H2": 0.5},
                    },
                }],
            )

    def test_bayesian_update_normalizes_posteriors_and_reduces_entropy(self):
        result = update_hypothesis_probabilities(
            hypotheses=HYPOTHESES,
            likelihoods={"H1": 0.1, "H2": 0.9},
            experiment_id="exp_1",
            observation="positive",
            evidence_id="evidence_1",
        )

        updates = {item["id"]: item for item in result["hypothesis_updates"]}
        self.assertAlmostEqual(updates["H1"]["posterior_probability"], 0.1)
        self.assertAlmostEqual(updates["H2"]["posterior_probability"], 0.9)
        self.assertAlmostEqual(sum(item["posterior_probability"] for item in updates.values()), 1.0)
        self.assertGreater(result["entropy_reduction"], 0)
        self.assertEqual(updates["H2"]["evidence_id"], "evidence_1")

    def test_bayesian_update_rejects_impossible_observation(self):
        with self.assertRaisesRegex(ValueError, "zero probability"):
            update_hypothesis_probabilities(
                hypotheses=HYPOTHESES,
                likelihoods={"H1": 0.0, "H2": 0.0},
                experiment_id="exp_1",
                observation="impossible",
            )

    def test_tool_contracts_return_json_and_schema(self):
        selector = ExperimentSelectorTool()
        updater = BeliefUpdaterTool()
        selected = json.loads(selector.execute({
            "hypotheses": HYPOTHESES,
            "experiments": [{
                "id": "exp",
                "outcomes": {
                    "yes": {"H1": 0.8, "H2": 0.2},
                    "no": {"H1": 0.2, "H2": 0.8},
                },
            }],
        }))
        updated = json.loads(updater.execute({
            "hypotheses": HYPOTHESES,
            "likelihoods": {"H1": 0.8, "H2": 0.2},
            "experiment_id": "exp",
            "observation": "yes",
        }))

        self.assertEqual(selected["selected_experiment"]["id"], "exp")
        self.assertTrue(math.isfinite(updated["posterior_entropy"]))
        self.assertIn("required", selector.parameters_schema())
        self.assertIn("required", updater.parameters_schema())


if __name__ == "__main__":
    unittest.main()

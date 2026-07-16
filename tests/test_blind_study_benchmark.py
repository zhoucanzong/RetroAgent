import unittest
from pathlib import Path

from retroagent.benchmark import load_blind_cases, score_blind_conclusion
from retroagent.tools.blind_study_oracle import BlindStudyOracleTool


PROJECT_DIR = Path(__file__).resolve().parents[1]
CASE_PATH = PROJECT_DIR / "benchmarks" / "blind_studies.jsonl"


class BlindStudyBenchmarkTests(unittest.TestCase):
    def test_recent_case_loads_without_exposing_answer_in_prompt(self):
        cases = load_blind_cases(CASE_PATH)

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.id, "photoreduction-selectivity-2026-001")
        self.assertEqual(case.published_date, "2026-07-15")
        self.assertGreaterEqual(len(case.experiments), 4)
        self.assertEqual(case.min_experiments, 3)
        self.assertEqual(case.expected_outcome, "H2")
        prompt = case.task_prompt()
        self.assertIn("blind_study_oracle", prompt)
        self.assertIn("what each requested measurement establishes", prompt)
        self.assertNotIn(case.source_url, prompt)
        self.assertNotIn(case.source_title, prompt)
        self.assertNotIn("expected_outcome", prompt)
        for criterion in case.success_criteria:
            self.assertNotIn(criterion, prompt)

    def test_oracle_lists_measurements_but_withholds_results_until_queried(self):
        case = load_blind_cases(CASE_PATH)[0]
        oracle = BlindStudyOracleTool(case.experiments)

        schema_text = str(oracle.parameters_schema())
        self.assertIn("transient_absorption", schema_text)
        self.assertNotIn("30000000000", schema_text)
        result = oracle.execute({"experiment_id": "transient_absorption"})
        self.assertIn("30000000000", result)
        self.assertIn("4000000000", result)
        missing = oracle.execute({"experiment_id": "not_registered"})
        self.assertIn("not registered", missing)

    def test_blind_score_requires_correct_outcome_and_mechanistic_chain(self):
        case = load_blind_cases(CASE_PATH)[0]
        correct = (
            "H2 is the major outcome: a [3+2] annulation. The super-potent photoreductant "
            "makes forward SET effectively diffusion-limited for both substrates. The ketyl "
            "undergoes beta-scission faster than back electron transfer, then adds to styrene; "
            "styrene radical anion is recycled by BET. Product selectivity is therefore controlled "
            "by post-SET kinetics rather than reduction-potential ordering."
        )
        wrong = (
            "H1 is the major outcome because styrene is easier to reduce. Styrene hydrogenation "
            "dominates and the cyclopropyl ketone remains unreacted."
        )

        passing = score_blind_conclusion(
            correct, case, queried_experiment_ids=[
                "transient_absorption", "post_set_barriers", "electrochemical_controls",
            ],
        )
        failing = score_blind_conclusion(
            wrong, case, queried_experiment_ids=["transient_absorption", "post_set_barriers"],
        )

        self.assertTrue(passing["blind_pass"])
        self.assertTrue(passing["outcome_pass"])
        self.assertTrue(passing["mechanism_pass"])
        self.assertFalse(failing["blind_pass"])
        self.assertFalse(failing["outcome_pass"])

    def test_blind_score_requires_active_measurement_selection(self):
        case = load_blind_cases(CASE_PATH)[0]
        conclusion = (
            "H2 is the major outcome and gives [3+2] annulation because diffusion-limited SET is followed by ketyl "
            "beta-scission faster than BET, while styrene radical anion is recycled by BET; "
            "post-SET kinetics override redox-potential ordering."
        )

        score = score_blind_conclusion(conclusion, case, queried_experiment_ids=[])

        self.assertTrue(score["outcome_pass"])
        self.assertFalse(score["measurement_pass"])
        self.assertFalse(score["blind_pass"])

    def test_blind_score_accepts_equivalent_post_set_mechanism_wording(self):
        case = load_blind_cases(CASE_PATH)[0]
        conclusion = (
            "The major outcome is H2: cyclopropyl ketone SET followed by rapid beta-scission "
            "and intermolecular [3+2] annulation with styrene. Both substrates capture the "
            "photogenerated reductant rapidly, but productive flux is determined by post-SET branching. The ketyl "
            "fragments faster than back electron transfer, while the styrene radical anion preferentially "
            "undergoes BET faster than protonation."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "transient_absorption", "post_set_barriers", "electrochemical_controls",
            ],
        )

        self.assertTrue(score["mechanism_pass"])
        self.assertTrue(score["blind_pass"])

    def test_blind_score_rejects_negated_outcome_and_reversed_causality(self):
        case = load_blind_cases(CASE_PATH)[0]
        adversarial_answers = [
            (
                "H2 is ruled out even though [3+2] annulation and beta-scission were considered. "
                "The major product is H1. Both substrates can undergo diffusion-limited SET, but BET "
                "is faster than beta-scission and therefore post-SET kinetics favor styrene reduction. "
                "The styrene radical anion is not recycled by BET."
            ),
            (
                "H2 lacks support; I choose H1 as the major outcome. Both substrates undergo "
                "diffusion-limited SET and beta-scission is possible, but BET happens before "
                "beta-scission. BET fails to recycle the styrene radical anion, and post-SET "
                "kinetics do not control selectivity. A [3+2] annulation is therefore rejected."
            ),
        ]

        for adversarial in adversarial_answers:
            with self.subTest(adversarial=adversarial):
                score = score_blind_conclusion(
                    adversarial,
                    case,
                    queried_experiment_ids=[
                        "transient_absorption", "post_set_barriers", "electrochemical_controls",
                    ],
                )

                self.assertFalse(score["outcome_pass"])
                self.assertFalse(score["mechanism_pass"])
                self.assertFalse(score["contradiction_pass"])
                self.assertFalse(score["blind_pass"])


if __name__ == "__main__":
    unittest.main()

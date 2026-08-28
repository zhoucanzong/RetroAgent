import unittest
from pathlib import Path

from retroagent.benchmark import load_blind_cases, score_blind_conclusion
from retroagent.tools.blind_study_oracle import BlindStudyOracleTool


PROJECT_DIR = Path(__file__).resolve().parents[1]
CASE_PATH = PROJECT_DIR / "benchmarks" / "blind_studies.jsonl"


class BlindStudyBenchmarkTests(unittest.TestCase):
    def test_recent_case_loads_without_exposing_answer_in_prompt(self):
        cases = load_blind_cases(CASE_PATH)

        self.assertEqual(len(cases), 2)
        case = cases[0]
        self.assertEqual(case.id, "photoreduction-selectivity-2026-001")
        self.assertEqual(case.published_date, "2026-07-15")
        self.assertGreaterEqual(len(case.experiments), 4)
        self.assertEqual(case.min_experiments, 3)
        self.assertEqual(case.expected_outcome, "H2")
        prompt = case.task_prompt()
        self.assertIn("blind_study_oracle", prompt)
        self.assertIn("what each requested measurement establishes", prompt)
        self.assertIn("audit every conclusion dimension", prompt)
        self.assertIn("carry each needed observation", prompt)
        self.assertIn("explicit Uncertainty:", prompt)
        self.assertNotIn(case.source_url, prompt)
        self.assertNotIn(case.source_title, prompt)
        self.assertNotIn("expected_outcome", prompt)
        for criterion in case.success_criteria:
            self.assertNotIn(criterion, prompt)

    def test_boronate_case_withholds_source_answer_and_measurement_results(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )

        self.assertEqual(case.published_date, "2026-07-13")
        self.assertEqual(case.scoring_profile, "aziridinium_migration")
        self.assertEqual(case.expected_outcome, "H2")
        self.assertEqual(case.min_experiments, 3)
        prompt = case.task_prompt()
        self.assertIn("reaction_progress_ir", prompt)
        self.assertNotIn(case.source_url, prompt)
        self.assertNotIn(case.source_title, prompt)
        self.assertNotIn("s41557-026-02206-6", prompt)
        self.assertNotIn("17.2", prompt)
        self.assertNotIn("19.6", prompt)
        self.assertNotIn("1,195", prompt)
        self.assertNotIn("expected_outcome", prompt)
        self.assertNotIn("aziridinium_migration", prompt)
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

    def test_boronate_oracle_withholds_transient_and_barriers_until_queried(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        oracle = BlindStudyOracleTool(case.experiments)

        schema_text = str(oracle.parameters_schema())
        self.assertIn("reaction_progress_ir", schema_text)
        self.assertIn("computed_energy_surface", schema_text)
        self.assertNotIn("1195", schema_text)
        self.assertNotIn("17.2", schema_text)
        self.assertNotIn("19.6", schema_text)
        result = oracle.execute({"experiment_id": "computed_energy_surface"})
        self.assertIn("17.2", result)
        self.assertIn("19.6", result)

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

    def test_boronate_final_run_keeps_real_omission_without_false_contradictions(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        conclusion = (
            "The evidence supports H2: amination forms an aziridinium intermediate whose transient "
            "IR band matches the calculated spectrum. This transient accumulates before product "
            "formation, falsifying the concerted H1 mechanism. Backside 1,2-migration from the "
            "aziridinium accounts for anti stereoselectivity. Product overreaction is suppressed "
            "because DPPH is consumed before free amine appears. H3 is ruled out. Uncertainty: "
            "barriers were not measured. Falsification: if constrained substrates gave syn products, "
            "the model would be contradicted."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "reaction_progress_ir", "spectral_assignment",
                "stereochemical_and_reactivity_controls",
            ],
        )

        self.assertTrue(score["outcome_pass"])
        self.assertEqual(score["mechanism_group_matches"][:4], [True] * 4)
        self.assertFalse(score["mechanism_group_matches"][4])
        self.assertTrue(score["mechanism_group_matches"][5])
        self.assertTrue(score["mechanism_group_matches"][6])
        self.assertTrue(score["contradiction_pass"])
        self.assertFalse(score["blind_pass"])

    def test_boronate_trial_three_remains_failed_only_for_missing_uncertainty(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        conclusion = (
            "The transformation proceeds via H2: rapid electrophilic amination forms a transient "
            "aziridinium intermediate, which undergoes slower backside 1,2-migration, giving anti "
            "stereoselectivity. The IR transient matches calculated aziridinium and accumulates "
            "before product. E/Z substrates give opposite products. Suppression of product "
            "overreaction is explained by DPPH being consumed before free amine forms, aided by "
            "its sparing solubility. H2 would be falsified if the transient did not match "
            "aziridinium or if stereochemical outcomes contradicted backside migration."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "reaction_progress_ir", "spectral_assignment",
                "stereochemical_and_reactivity_controls",
            ],
        )

        self.assertTrue(score["outcome_pass"])
        self.assertEqual(score["mechanism_group_matches"][:5], [True] * 5)
        self.assertFalse(score["mechanism_group_matches"][5])
        self.assertTrue(score["mechanism_group_matches"][6])
        self.assertTrue(score["contradiction_pass"])
        self.assertFalse(score["blind_pass"])

    def test_boronate_trial_two_falsifier_is_not_treated_as_actual_negation(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        conclusion = (
            "The mechanism is H2, stepwise via aziridinium. The transient IR band uniquely matches "
            "calculated aziridinium, identifying the intermediate, and the transient accumulates "
            "before product. E/Z substrates give opposite products, consistent with backside anti "
            "migration; DPPH is consumed before free amine appears, explaining compatibility with "
            "no overreaction. Electrophilic amination forms an aziridinium intermediate, which then "
            "undergoes slower backside 1,2-migration. Uncertainty remains. Falsification: evidence "
            "showing the transient is not aziridinium would challenge H2."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "reaction_progress_ir", "spectral_assignment",
                "stereochemical_and_reactivity_controls",
            ],
        )

        self.assertTrue(score["outcome_pass"])
        self.assertTrue(score["mechanism_group_matches"][0])
        self.assertTrue(score["mechanism_group_matches"][1])
        self.assertTrue(score["mechanism_group_matches"][2])
        self.assertTrue(score["mechanism_group_matches"][3])
        self.assertFalse(score["mechanism_group_matches"][4])
        self.assertTrue(score["contradiction_pass"])
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

    def test_boronate_score_requires_stepwise_temporal_and_self_limiting_mechanism(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        conclusion = (
            "H2 is the supported mechanism. The reaction-progress IR shows rapid accumulation "
            "of an aziridinium intermediate before a slower 1,2-migration; matching calculated "
            "and observed IR bands assign the transient to that aziridinium. Backside 1,2-migration "
            "opens it and explains the anti stereochemistry. DPPH has a low effective concentration "
            "and is consumed before free amine product accumulates, temporally separating the two "
            "and preventing product overreaction. Uncertainty remains in the computed barriers; "
            "failure to observe the transient by an independent rapid-scan method would falsify H2."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "reaction_progress_ir", "computed_energy_surface", "spectral_assignment",
            ],
        )

        self.assertTrue(score["outcome_pass"])
        self.assertTrue(score["mechanism_pass"])
        self.assertTrue(score["measurement_pass"])
        self.assertTrue(score["contradiction_pass"])
        self.assertTrue(score["blind_pass"])

    def test_boronate_trial_one_is_not_false_failed_but_still_lacks_control_evidence(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        conclusion = (
            "The evidence establishes that the reaction proceeds via a stepwise mechanism (H2): "
            "rapid asynchronous nitrogen transfer from DPPH to the alkenyl boronate complex forms "
            "an aziridinium intermediate whose observed transient IR band uniquely matches the "
            "calculated aziridinium, followed by slower backside 1,2-migration with inversion at "
            "the migration terminus, yielding the anti beta-amino boronic ester product. The "
            "suppression of product overreaction is explained by a protective intermediate combined "
            "with rapid intramolecular coordination or low residual aminating reagent. Uncertainty "
            "remains, and observing radical intermediates would falsify H2."
        )

        score = score_blind_conclusion(
            conclusion,
            case,
            queried_experiment_ids=[
                "reaction_progress_ir", "computed_energy_surface", "spectral_assignment",
            ],
        )

        self.assertTrue(score["outcome_pass"])
        self.assertTrue(score["mechanism_group_matches"][0])
        self.assertTrue(score["mechanism_group_matches"][1])
        self.assertTrue(score["mechanism_group_matches"][2])
        self.assertFalse(score["mechanism_group_matches"][3])
        self.assertFalse(score["mechanism_group_matches"][4])
        self.assertFalse(score["mechanism_pass"])
        self.assertTrue(score["contradiction_pass"])
        self.assertFalse(score["blind_pass"])

    def test_boronate_score_rejects_concerted_reversed_and_negated_mechanisms(self):
        case = next(
            item for item in load_blind_cases(CASE_PATH)
            if item.id == "boronate-amination-2026-002"
        )
        adversarial_answers = [
            (
                "H2 is rejected and H1 is the supported concerted mechanism. The migration happens "
                "before aziridinium formation, the transient is not an aziridinium, and frontside "
                "migration gives syn product. DPPH remains present with the free amine without "
                "being consumed, so overreaction is not suppressed."
            ),
            (
                "H2 is supported because an aziridinium may exist, but slow 1,2-migration occurs "
                "before rapid aziridinium formation. The observed band does not identify that "
                "intermediate. A frontside migration explains anti selectivity, while DPPH persists "
                "after free amine accumulates and therefore prevents overreaction."
            ),
        ]

        for adversarial in adversarial_answers:
            with self.subTest(adversarial=adversarial):
                score = score_blind_conclusion(
                    adversarial,
                    case,
                    queried_experiment_ids=[
                        "reaction_progress_ir", "computed_energy_surface", "spectral_assignment",
                    ],
                )

                self.assertFalse(score["mechanism_pass"])
                self.assertFalse(score["contradiction_pass"])
                self.assertFalse(score["blind_pass"])

    def test_blind_scoring_profiles_do_not_accept_the_other_case_mechanism(self):
        cases = {case.id: case for case in load_blind_cases(CASE_PATH)}
        photoreduction = cases["photoreduction-selectivity-2026-001"]
        boronate = cases["boronate-amination-2026-002"]
        photoreduction_answer = (
            "H2 is the major outcome: [3+2] annulation. Both substrates undergo rapid SET, "
            "beta-scission is faster than BET, styrene radical anion is recycled by BET, and "
            "post-SET kinetics control selectivity."
        )
        boronate_answer = (
            "H2 proceeds by rapid aziridinium formation before slower backside 1,2-migration, "
            "which gives anti selectivity. The transient IR matches aziridinium, and sparingly "
            "soluble DPPH is consumed before free amine accumulates, preventing overreaction."
        )

        wrong_for_boronate = score_blind_conclusion(
            photoreduction_answer,
            boronate,
            queried_experiment_ids=list(boronate.experiments)[:3],
        )
        wrong_for_photoreduction = score_blind_conclusion(
            boronate_answer,
            photoreduction,
            queried_experiment_ids=list(photoreduction.experiments)[:3],
        )

        self.assertFalse(wrong_for_boronate["mechanism_pass"])
        self.assertFalse(wrong_for_boronate["blind_pass"])
        self.assertFalse(wrong_for_photoreduction["mechanism_pass"])
        self.assertFalse(wrong_for_photoreduction["blind_pass"])


if __name__ == "__main__":
    unittest.main()

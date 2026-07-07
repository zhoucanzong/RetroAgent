import unittest

from retroagent.schemas import (
    AuditIssue,
    build_research_report,
    CriticReport,
    Evidence,
    EvidenceLevel,
    ExperimentPlan,
    ExperimentStep,
    normalize_chemical_hypothesis,
    ResearchMemoryRecord,
    RiskFlag,
    RiskSeverity,
    build_reaction_record,
    normalize_reaction_record,
)


class SchemaTests(unittest.TestCase):
    def test_evidence_and_risk_flags_serialize_to_json_ready_values(self):
        evidence = Evidence(
            level=EvidenceLevel.ANALOG_SUPPORTED,
            source="literature",
            summary="Similar substrate reported under related conditions.",
            url="https://example.test/paper",
            score=0.8,
        )
        risk = RiskFlag(
            severity=RiskSeverity.WARNING,
            category="chemoselectivity",
            message="Competing ester hydrolysis is possible.",
            evidence=[evidence],
        )

        self.assertEqual(evidence.to_dict()["level"], "analog-supported")
        self.assertEqual(risk.to_dict()["severity"], "warning")
        self.assertEqual(risk.to_dict()["evidence"][0]["level"], "analog-supported")

    def test_reaction_record_has_canonical_and_legacy_precursor_fields(self):
        reaction = build_reaction_record(
            reaction_id="rxn_a",
            product_smiles="CC=O",
            precursor_smiles_list=["CCO"],
            template_index=5,
            classification="Oxidation",
            library_occurrence=42,
        )

        self.assertEqual(reaction["id"], "rxn_a")
        self.assertEqual(reaction["precursor_smiles_list"], ["CCO"])
        self.assertEqual(reaction["precursors"], ["CCO"])
        self.assertEqual(reaction["template_index"], 5)

    def test_normalize_reaction_record_accepts_legacy_payload(self):
        normalized = normalize_reaction_record(
            {"precursors": ["CCO"], "product": "CC=O"},
            index=3,
        )

        self.assertEqual(normalized["id"], "rxn_3")
        self.assertEqual(normalized["product_smiles"], "CC=O")
        self.assertEqual(normalized["precursor_smiles_list"], ["CCO"])

    def test_chemical_hypothesis_normalization_defaults_unknown_labels(self):
        hypothesis = normalize_chemical_hypothesis(
            {
                "claim": "Late-stage oxidation could access the target.",
                "status": "not-a-status",
                "evidence": [{"level": "unrecognized", "source": "tool", "summary": "low score"}],
                "risk_flags": [{"severity": "severe", "category": "selectivity", "message": "unknown"}],
                "next_experiments": [{"objective": "probe oxidation", "procedure": "small-scale screen"}],
            },
            target_smiles="CC=O",
        )

        data = hypothesis.to_dict()

        self.assertEqual(data["id"], "hypothesis_1")
        self.assertEqual(data["status"], "proposed")
        self.assertEqual(data["target_smiles"], "CC=O")
        self.assertEqual(data["evidence"][0]["level"], "speculative")
        self.assertEqual(data["risk_flags"][0]["severity"], "warning")

    def test_build_research_report_checks_completeness(self):
        report = build_research_report(
            target_smiles="CC=O",
            hypotheses=[
                {
                    "id": "h1",
                    "claim": "Analog-supported route.",
                    "evidence": [{"level": "analog-supported", "source": "paper", "summary": "similar substrate"}],
                    "risk_flags": [{"severity": "warning", "category": "conditions", "message": "needs screen"}],
                    "next_experiments": [{"objective": "validate", "procedure": "run a microscale test"}],
                },
                {"id": "h2", "claim": "Incomplete route."},
            ],
        )

        self.assertEqual(report["hypothesis_count"], 2)
        self.assertEqual(report["evidence_level_counts"]["analog-supported"], 1)
        self.assertEqual(report["risk_severity_counts"]["warning"], 1)
        self.assertFalse(report["ready_for_review"])
        self.assertIn("h2: evidence", report["missing_review_fields"])

    def test_critic_report_serializes_issue_counts(self):
        report = CriticReport(
            target_smiles="CC=O",
            issues=[
                AuditIssue(
                    severity=RiskSeverity.WARNING,
                    category="evidence",
                    message="Speculative-only route.",
                )
            ],
            passed_checks=["hypothesis_portfolio_present"],
        ).to_dict()

        self.assertEqual(report["issue_count"], 1)
        self.assertEqual(report["warning_count"], 1)
        self.assertEqual(report["critical_count"], 0)
        self.assertEqual(report["issues"][0]["severity"], "warning")

    def test_experiment_plan_serializes_steps_and_controls(self):
        plan = ExperimentPlan(
            target_smiles="CC=O",
            objective="Validate route",
            steps=[
                ExperimentStep(
                    objective="Run microscale reaction",
                    procedure="Combine precursor and reagent under screened conditions.",
                    decision_point="Advance on target formation.",
                )
            ],
            controls=[
                ExperimentStep(
                    objective="No reagent control",
                    procedure="Run without key reagent.",
                )
            ],
            success_criteria=["target detected"],
            failure_diagnostics=["no conversion means revisit activation"],
            linked_issue_categories=["mechanism"],
        ).to_dict()

        self.assertEqual(plan["steps"][0]["objective"], "Run microscale reaction")
        self.assertEqual(plan["controls"][0]["objective"], "No reagent control")
        self.assertEqual(plan["linked_issue_categories"], ["mechanism"])

    def test_research_memory_record_serializes_summary_payloads(self):
        record = ResearchMemoryRecord(
            id="mem_1",
            target_smiles="CC=O",
            target_name="acetaldehyde",
            outcome="needs-validation",
            critic_summary={"warning_count": 1},
            experiment_plan={"step_count": 2},
            route_graph_summary={"unsolved_molecule_count": 1},
            tags=["oxidation"],
        ).to_dict()

        self.assertEqual(record["id"], "mem_1")
        self.assertEqual(record["critic_summary"]["warning_count"], 1)
        self.assertEqual(record["experiment_plan"]["step_count"], 2)
        self.assertEqual(record["tags"], ["oxidation"])


if __name__ == "__main__":
    unittest.main()

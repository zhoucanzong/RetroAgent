import json
import unittest

from retroagent.tools.research_report import ResearchReportTool


class ResearchReportToolTests(unittest.TestCase):
    def test_build_research_report_tool_returns_ready_report(self):
        tool = ResearchReportTool()

        result = json.loads(tool.execute({
            "title": "Discovery portfolio",
            "target_smiles": "CC=O",
            "hypotheses": [
                {
                    "id": "route_a",
                    "claim": "Analog-supported oxidation route.",
                    "route_summary": "Use an alcohol precursor and oxidize under mild conditions.",
                    "status": "tool-supported",
                    "evidence": [
                        {
                            "level": "analog-supported",
                            "source": "search_literature",
                            "summary": "Related primary alcohol oxidation precedent.",
                        }
                    ],
                    "risk_flags": [
                        {
                            "severity": "warning",
                            "category": "selectivity",
                            "message": "Overoxidation is possible.",
                        }
                    ],
                    "next_experiments": [
                        {
                            "objective": "Test oxidation selectivity.",
                            "procedure": "Run small-scale condition screen and monitor conversion.",
                        }
                    ],
                }
            ],
        }))

        self.assertTrue(result["ready_for_review"])
        self.assertEqual(result["hypothesis_count"], 1)
        self.assertEqual(result["evidence_level_counts"]["analog-supported"], 1)
        self.assertEqual(result["risk_severity_counts"]["warning"], 1)

    def test_build_research_report_tool_flags_missing_review_parts(self):
        tool = ResearchReportTool()

        result = json.loads(tool.execute({
            "target_smiles": "CC=O",
            "hypotheses": [{"id": "route_a", "claim": "Unreviewed route."}],
        }))

        self.assertFalse(result["ready_for_review"])
        self.assertIn("route_a: evidence", result["missing_review_fields"])
        self.assertIn("route_a: risk review", result["missing_review_fields"])
        self.assertIn("route_a: next_experiments", result["missing_review_fields"])

    def test_parameters_schema_requires_target_and_hypotheses(self):
        schema = ResearchReportTool().parameters_schema()

        self.assertEqual(schema["required"], ["target_smiles", "hypotheses"])
        self.assertIn("hypotheses", schema["properties"])


if __name__ == "__main__":
    unittest.main()

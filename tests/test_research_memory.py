import json
import tempfile
import unittest
from pathlib import Path

from retroagent.tools.research_memory import (
    ResearchMemoryTool,
    build_memory_record,
    load_memory_records,
    search_memory_records,
    summarize_memory_records,
)


class ResearchMemoryToolTests(unittest.TestCase):
    def test_build_memory_record_summarizes_critic_and_experiment_plan(self):
        record = build_memory_record({
            "target_smiles": "CC=O",
            "target_name": "acetaldehyde",
            "outcome": "needs-validation",
            "hypotheses": [{"id": "h1", "claim": "Oxidation route"}],
            "critic_report": {
                "issue_count": 2,
                "warning_count": 1,
                "metadata": {"issue_category_counts": {"mechanism": 1}},
                "passed_checks": ["reaction_records_present"],
            },
            "experiment_plan": {
                "objective": "Validate mechanism",
                "steps": [{"objective": "Map reaction"}],
                "controls": [{"objective": "negative control"}],
                "success_criteria": ["mapping supports product"],
                "linked_issue_categories": ["mechanism"],
                "ready_for_report": True,
            },
            "route_graph": {"stats": {"molecule_count": 2, "reaction_count": 1, "unsolved_molecule_count": 1}},
            "tags": ["oxidation"],
        })

        data = record.to_dict()

        self.assertEqual(data["target_name"], "acetaldehyde")
        self.assertEqual(data["critic_summary"]["warning_count"], 1)
        self.assertEqual(data["experiment_plan"]["step_count"], 1)
        self.assertEqual(data["route_graph_summary"]["unsolved_molecule_count"], 1)

    def test_tool_records_searches_and_summarizes_jsonl_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            tool = ResearchMemoryTool()

            recorded = json.loads(tool.execute({
                "operation": "record",
                "memory_path": str(path),
                "target_smiles": "CC=O",
                "target_name": "acetaldehyde",
                "outcome": "needs-validation",
                "hypotheses": [{"id": "h1", "claim": "Oxidation route"}],
                "critic_report": {
                    "issue_count": 1,
                    "metadata": {"issue_category_counts": {"conditions": 1}},
                },
                "experiment_plan": {"steps": [{"objective": "condition screen"}]},
                "tags": ["oxidation", "screen"],
            }))
            searched = json.loads(tool.execute({
                "operation": "search",
                "memory_path": str(path),
                "query": "oxidation",
            }))
            summarized = json.loads(tool.execute({
                "operation": "summarize",
                "memory_path": str(path),
            }))

            self.assertEqual(recorded["status"], "recorded")
            self.assertEqual(len(load_memory_records(path)), 1)
            self.assertEqual(searched["count"], 1)
            self.assertEqual(summarized["summary"]["record_count"], 1)
            self.assertEqual(summarized["summary"]["issue_category_counts"]["conditions"], 1)

    def test_search_memory_respects_target_and_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            tool = ResearchMemoryTool()
            for smiles in ["CCO", "CC=O", "CC=O"]:
                tool.execute({
                    "operation": "record",
                    "memory_path": str(path),
                    "target_smiles": smiles,
                    "notes": f"target {smiles}",
                })

            records = search_memory_records(path, target_smiles="CC=O", limit=1)
            summary = summarize_memory_records(records)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].target_smiles, "CC=O")
            self.assertEqual(summary["record_count"], 1)

    def test_unknown_operation_returns_error(self):
        result = json.loads(ResearchMemoryTool().execute({"operation": "delete"}))

        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

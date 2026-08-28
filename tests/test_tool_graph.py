import unittest

from retroagent.environments import RetroEnvironment
from retroagent.tool_graph import next_tools, summarize_tool_graph


class _DummyTool:
    description = "dummy"

    def parameters_schema(self):
        return {"type": "object", "properties": {}}


class ToolGraphTests(unittest.TestCase):
    def test_next_tools_respects_available_tools(self):
        available = [
            "disconnect",
            "propose",
            "evaluate",
            "score_routes",
            "map_reaction",
            "chemical_critic",
            "design_experiments",
            "select_experiment",
            "update_hypotheses",
            "build_route_portfolio",
            "validate_route_portfolio",
            "build_research_report",
            "research_memory",
            "think",
        ]

        self.assertEqual(next_tools("disconnect", available), ["propose", "think"])
        self.assertEqual(next_tools("propose", available), ["evaluate", "score_routes", "map_reaction"])
        self.assertIn("chemical_critic", next_tools("score_routes", available))
        self.assertIn("chemical_critic", next_tools("map_reaction", available))
        self.assertIn("design_experiments", next_tools("chemical_critic", available))
        self.assertIn("build_research_report", next_tools("design_experiments", available))
        self.assertEqual(next_tools("select_experiment", available), ["update_hypotheses"])
        self.assertEqual(next_tools("build_route_portfolio", available), ["validate_route_portfolio"])
        self.assertIn("build_research_report", next_tools("validate_route_portfolio", available))
        self.assertEqual(next_tools("build_research_report", available), ["research_memory", "think"])

    def test_tool_graph_summary_contains_dependencies(self):
        summary = summarize_tool_graph([
            "disconnect",
            "propose",
            "evaluate",
            "score_routes",
            "chemical_critic",
            "design_experiments",
            "build_research_report",
            "research_memory",
        ])

        self.assertIn("`propose`", summary)
        self.assertIn("`score_routes`", summary)
        self.assertIn("`chemical_critic`", summary)
        self.assertIn("`design_experiments`", summary)
        self.assertIn("`build_research_report`", summary)
        self.assertIn("`research_memory`", summary)
        self.assertIn("disconnect", summary)
        self.assertIn("Suggested next", summary)

    def test_environment_exposes_tool_graph_summary(self):
        env = RetroEnvironment()
        env.register("disconnect", _DummyTool())
        env.register("propose", _DummyTool())

        summary = env.get_tool_graph_summary()

        self.assertIn("`disconnect`", summary)
        self.assertIn("`propose`", summary)


if __name__ == "__main__":
    unittest.main()

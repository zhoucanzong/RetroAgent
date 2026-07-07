import unittest

from retroagent.agents.planner import RetroPlanner
from retroagent.blackboard import SharedBlackboard
from retroagent.config import load_config


class _FakeEnv:
    def get_tools_spec(self):
        return []

    def get_tool_graph_summary(self):
        return "| Tool | Category |\n|------|----------|\n| `disconnect` | retrosynthesis |"


class _FakeModel:
    pass


class DiscoveryModeTests(unittest.TestCase):
    def test_discovery_mode_initializes_route_graph(self):
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")

        self.assertEqual(blackboard.mode, "discovery")
        self.assertEqual(blackboard.route_graph.stats()["molecule_count"], 1)
        self.assertEqual(blackboard.route_graph.stats()["target_id"], "mol_1")

    def test_discovery_prompt_uses_discovery_strategy(self):
        cfg = load_config()
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        planner = RetroPlanner(
            _FakeModel(),
            _FakeEnv(),
            blackboard,
            system_template=cfg.agent_system_template,
            instance_template=cfg.agent_instance_template,
        )
        planner.extra_template_vars |= {"task": "CC=O", "mode": "discovery"}

        system_prompt = planner._render_template(planner.config.system_template)
        instance_prompt = planner._render_template(planner.config.instance_template)

        self.assertIn("## Discovery Strategy", system_prompt)
        self.assertIn("## ToolGraph Workflow Hints", system_prompt)
        self.assertIn("`disconnect`", system_prompt)
        self.assertIn("Route graph", system_prompt)
        self.assertIn("chemical_critic", system_prompt)
        self.assertIn("design_experiments", system_prompt)
        self.assertIn("build_research_report", system_prompt)
        self.assertIn("research_memory", system_prompt)
        self.assertIn("Discover and evaluate diverse synthetic route hypotheses", instance_prompt)


if __name__ == "__main__":
    unittest.main()

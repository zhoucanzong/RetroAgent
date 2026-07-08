import unittest
import json

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


class _FakeEnvWithoutExamples:
    def get_tools_spec(self):
        return [
            {
                "name": "tool_without_examples",
                "description": "A tool spec without optional examples metadata.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def get_tool_graph_summary(self):
        return "| Tool | Category |\n|------|----------|\n| `tool_without_examples` | test |"


class _FakeRouteWorkflowEnv:
    def __init__(self):
        self.executed = []

    def get_tools_spec(self):
        return [
            {
                "name": "propose",
                "description": "Generate candidate retrosynthetic reactions.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "score_routes",
                "description": "Score route candidates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_smiles": {"type": "string"},
                        "reactions": {"type": "array"},
                    },
                    "required": ["target_smiles", "reactions"],
                },
            },
            {
                "name": "check_patent",
                "description": "Check patent context.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def get_tool_graph_summary(self):
        return "| Tool | Category |\n|------|----------|\n| `score_routes` | discovery |"

    def execute(self, action):
        self.executed.append(action)
        tool = action["tool"]
        if tool == "propose":
            return {
                "returncode": 0,
                "output": json.dumps({
                    "reactions": [
                        {
                            "id": "rxn_1",
                            "target": "CC=O",
                            "precursors": ["CCO"],
                            "template_index": 1,
                            "score": 0.42,
                        }
                    ]
                }),
            }
        if tool == "score_routes":
            return {
                "returncode": 0,
                "output": json.dumps({
                    "routes": [
                        {
                            "route_id": "route_1",
                            "score": 0.66,
                            "metadata": {"solved": False},
                            "steps": ["rxn_1"],
                        }
                    ],
                    "summary": {"best_score": 0.66, "solved_route_count": 0},
                }),
            }
        return {"returncode": 0, "output": "{}"}


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
        self.assertIn("score_routes", system_prompt)
        self.assertIn("chemical_critic", system_prompt)
        self.assertIn("design_experiments", system_prompt)
        self.assertIn("build_research_report", system_prompt)
        self.assertIn("research_memory", system_prompt)
        self.assertIn("Discover and evaluate diverse synthetic route hypotheses", instance_prompt)

    def test_prompt_renders_tools_without_examples_field(self):
        cfg = load_config()
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        planner = RetroPlanner(
            _FakeModel(),
            _FakeEnvWithoutExamples(),
            blackboard,
            system_template=cfg.agent_system_template,
            instance_template=cfg.agent_instance_template,
        )
        planner.extra_template_vars |= {"task": "CC=O", "mode": "discovery"}

        system_prompt = planner._render_template(planner.config.system_template)

        self.assertIn("tool_without_examples", system_prompt)

    def test_auto_tune_respects_explicit_low_step_limit(self):
        blackboard = SharedBlackboard()
        blackboard.initialize(
            "COc1cc(C(=O)N[C@H](C)c2ccncc2)cc(S(=O)(=O)N2CCOCC2)c1OC(F)(F)F",
            mode="discovery",
        )
        planner = RetroPlanner(
            _FakeModel(),
            _FakeEnv(),
            blackboard,
            step_limit=35,
            backtrack_patience=2,
        )

        planner._auto_tune_from_complexity()

        self.assertEqual(planner.config.step_limit, 35)
        self.assertEqual(planner.config.backtrack_patience, 3)

    def test_discovery_workflow_guard_requires_route_scoring_after_candidates(self):
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        blackboard.proposal_results = [{"id": "rxn_1"}]
        planner = RetroPlanner(_FakeModel(), _FakeEnv(), blackboard)
        planner.messages = [
            {"role": "user", "content": "<tool>propose</tool>\n<output>{\"reactions\":[]}</output>", "extra": {}}
        ]

        hint = planner._discovery_workflow_hint()

        self.assertIn("score_routes", hint)
        self.assertFalse(planner._discovery_submission_ready())

    def test_discovery_gate_blocks_side_tools_after_candidates_until_scored(self):
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        blackboard.proposal_results = [{"id": "rxn_1"}]
        planner = RetroPlanner(_FakeModel(), _FakeRouteWorkflowEnv(), blackboard)
        planner.messages = [
            {"role": "user", "content": "<tool>propose</tool>\n<output>{}</output>", "extra": {}}
        ]

        allowed, reason = planner._discovery_action_allowed({"tool": "check_patent", "parameters": {}})

        self.assertFalse(allowed)
        self.assertIn("score_routes", reason)

    def test_discovery_auto_scores_after_propose(self):
        env = _FakeRouteWorkflowEnv()
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        planner = RetroPlanner(_FakeModel(), env, blackboard)

        observations = planner.execute_actions({
            "extra": {"actions": [{"tool": "propose", "parameters": {}}]}
        })

        self.assertEqual([a["tool"] for a in env.executed], ["propose", "score_routes"])
        self.assertEqual(len(blackboard.routes), 1)
        self.assertIn("route_1", blackboard.scores)
        self.assertTrue(any("<tool>score_routes</tool>" in msg["content"] for msg in observations))
        self.assertTrue(any("chemical_critic" in msg["content"] for msg in observations))

    def test_discovery_submission_ready_after_required_workflow(self):
        blackboard = SharedBlackboard()
        blackboard.initialize("CC=O", mode="discovery")
        planner = RetroPlanner(_FakeModel(), _FakeEnv(), blackboard)
        planner.messages = [
            {"role": "user", "content": "<tool>score_routes</tool>\n<output>{}</output>", "extra": {}},
            {"role": "user", "content": "<tool>chemical_critic</tool>\n<output>{}</output>", "extra": {}},
            {"role": "user", "content": "<tool>design_experiments</tool>\n<output>{}</output>", "extra": {}},
            {"role": "user", "content": "<tool>build_research_report</tool>\n<output>{}</output>", "extra": {}},
            {
                "role": "user",
                "content": "<tool>research_memory</tool>\n<output>{\"status\":\"recorded\"}</output>",
                "extra": {},
            },
        ]

        self.assertEqual(planner._discovery_workflow_hint(), "")
        self.assertTrue(planner._discovery_submission_ready())


if __name__ == "__main__":
    unittest.main()

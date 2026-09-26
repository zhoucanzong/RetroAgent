import json
import math
import tempfile
import unittest
from pathlib import Path

from retroagent.agents.discovery_controller import DiscoveryController
from retroagent.research_execution import PermissionGrant, PermissionMode
from retroagent.research_state import ResearchLedger
from retroagent.benchmark import score_discovery_v3_events
from retroagent.environments import RetroEnvironment
from retroagent.tools.active_learning import BeliefUpdaterTool, ExperimentSelectorTool
from retroagent.tools.route_discovery import IndependentRouteValidatorTool, RoutePortfolioTool
from retroagent.tools.retrochimera import NeuralLocPredictionTool, TransformerPredictionTool


def _decision(action_type, action, target_ids, *, stop=False, conclusion="", evidence_ids=None):
    return {
        "research_question": "What action best reduces current route uncertainty?",
        "action_type": action_type,
        "action": action,
        "target_ids": target_ids,
        "rationale": "Use the latest route evidence to choose the next discriminating action.",
        "expected_observation": "The result distinguishes the active route hypotheses.",
        "falsification_condition": "The result does not discriminate between routes.",
        "estimated_cost": {"time_minutes": 1, "compute": "low"},
        "stop": stop,
        "conclusion": conclusion,
        "evidence_ids": list(evidence_ids or []),
    }


class _ScriptedModel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.observations = []
        self.tools_arguments = []

    def query(self, messages, tools=None):
        self.observations.append(json.loads(messages[-1]["content"]))
        self.tools_arguments.append(tools)
        decision = self.decisions.pop(0)
        return {"role": "assistant", "content": json.dumps(decision), "extra": {}}


class _FencedModel(_ScriptedModel):
    def query(self, messages, tools=None):
        self.observations.append(json.loads(messages[-1]["content"]))
        self.tools_arguments.append(tools)
        decision = self.decisions.pop(0)
        return {"role": "assistant", "content": f"```json\n{json.dumps(decision)}\n```", "extra": {}}


class _FeedbackEnv:
    def __init__(self):
        self.calls = []

    def get_tools_spec(self):
        return [
            {"name": "feedback_oracle", "description": "Deterministic feedback", "parameters": {}},
            {"name": "inspect_route", "description": "Inspect a route", "parameters": {}},
        ]

    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "feedback_oracle":
            return {
                "returncode": 0,
                "output": json.dumps({
                    "failure": {
                        "reaction_ids": ["rxn_a"],
                        "reason": "chemoselectivity failure",
                        "category": "selectivity",
                        "repairable": False,
                    }
                }),
                "exception_info": "",
            }
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B avoids the failed coupling", "route_id": "route_b"}),
            "exception_info": "",
        }


class _RouteOutputEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        return {
            "returncode": 0,
            "output": json.dumps({
                "routes": [{
                    "route_id": "route_generated",
                    "reaction_ids": ["rxn_generated"],
                    "score": 0.81,
                    "metadata": {"solved": False},
                }]
            }),
            "exception_info": "",
        }


class _FailingThenInspectEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "unstable_calculation":
            return {"returncode": -1, "output": "solver failed", "exception_info": "solver failed"}
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B remains available"}),
            "exception_info": "",
        }


class _RaisingThenInspectEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "exploding_tool":
            raise RuntimeError("adapter exploded")
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B remains available"}),
            "exception_info": "",
        }


class _MalformedStateEventEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "malformed_state":
            return {
                "returncode": 0,
                "output": json.dumps({"state_event": {"type": "hypothesis_proposed"}}),
                "exception_info": "",
            }
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B remains available"}),
            "exception_info": "",
        }


class _ForgedStateEventEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "forge_state":
            return {
                "returncode": 0,
                "output": json.dumps({
                    "state_event": {
                        "type": "route_status_changed",
                        "payload": {"id": "route_a", "status": "active"},
                    }
                }),
                "exception_info": "",
            }
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B remains available"}),
            "exception_info": "",
        }


class _NullCollectionsEnv(_FeedbackEnv):
    def execute(self, action):
        self.calls.append(action)
        if action["tool"] == "null_routes":
            return {
                "returncode": 0,
                "output": json.dumps({"routes": None, "reactions": None}),
                "exception_info": "",
            }
        return {
            "returncode": 0,
            "output": json.dumps({"summary": "Route B remains available"}),
            "exception_info": "",
        }


def _competing_routes_ledger() -> ResearchLedger:
    ledger = ResearchLedger()
    ledger.append("route_registered", {
        "id": "route_a", "reaction_ids": ["rxn_a"], "status": "active",
        "scores": {"feasibility": 0.9, "uncertainty": 0.2},
    })
    ledger.append("route_registered", {
        "id": "route_b", "reaction_ids": ["rxn_b"], "status": "active",
        "scores": {"feasibility": 0.7, "uncertainty": 0.4},
    })
    return ledger


class DiscoveryControllerTests(unittest.TestCase):
    def test_agent_can_choose_standalone_models_without_forced_ensemble_sequence(self):
        prediction = type("Prediction", (), {
            "reactants": [type("Molecule", (), {"smiles": "CCO"})()],
            "metadata": {"probability": 0.8},
        })()
        factory = lambda **kwargs: lambda inputs, num_results: [[prediction]]
        env = RetroEnvironment(include_bash=False)
        env.register("transformer_predict", TransformerPredictionTool(model_factory=factory, molecule_factory=str))
        env.register("neuralloc_predict", NeuralLocPredictionTool(model_factory=factory, molecule_factory=str))
        ledger = ResearchLedger()
        class ChoosingModel(_ScriptedModel):
            def query(self, messages, tools=None):
                if self.decisions[0]["stop"]:
                    self.decisions[0]["target_ids"] = list(ledger.replay().reactions)
                return super().query(messages, tools)
        with tempfile.TemporaryDirectory() as tmp:
            model = ChoosingModel([
                _decision("tool", {"tool": name, "parameters": {
                    "target_smiles": "CC=O", "model_dir": tmp,
                }}, []) for name in ("transformer_predict", "neuralloc_predict")
            ] + [_decision("stop", {}, [], stop=True,
                           conclusion="Two computational proposal sources were compared.",
                           evidence_ids=["evidence_0001", "evidence_0002"])])
            result = DiscoveryController(model, env, ledger=ledger, workspace=tmp,
                                         allow_execution_tools=False).run("Compare proposal sources")
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(len(ledger.replay().reactions), 2)
        self.assertEqual([e.payload["source"] for e in ledger.events if e.type == "evidence_added"],
                         ["transformer_predict", "neuralloc_predict"])

    def test_failure_feedback_is_observed_before_agent_selects_alternative_route(self):
        ledger = _competing_routes_ledger()
        model = _ScriptedModel([
            _decision("tool", {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision("stop", {}, ["route_b"], stop=True, conclusion="Route B is preferred.", evidence_ids=["evidence_0002"]),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=ledger, workspace=Path(tmp), max_cycles=5,
            ).run("find a robust route")

        state = ledger.replay()
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(state.routes["route_a"]["status"], "failed")
        self.assertEqual(state.routes["route_b"]["status"], "active")
        self.assertEqual([call["tool"] for call in env.calls], ["feedback_oracle", "inspect_route"])
        observed_routes = {route["id"]: route for route in model.observations[1]["state"]["routes"]}
        self.assertEqual(observed_routes["route_a"]["status"], "failed")
        self.assertEqual(model.observations[1]["state"]["replan_required"], True)
        self.assertIn("available_tools", model.observations[0])
        self.assertTrue(all(value is None for value in model.tools_arguments))

    def test_unchanged_failed_action_is_blocked_before_execution(self):
        ledger = _competing_routes_ledger()
        failed_action = {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}
        model = _ScriptedModel([
            _decision("tool", failed_action, ["route_a"]),
            _decision("tool", failed_action, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision("stop", {}, ["route_b"], stop=True, conclusion="Use route B.", evidence_ids=["evidence_0002"]),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, env, ledger=ledger, workspace=tmp, max_cycles=6).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["feedback_oracle", "inspect_route"])
        self.assertTrue(any(event.type == "decision_rejected" for event in ledger.events))

    def test_stop_without_valid_evidence_is_rejected(self):
        ledger = _competing_routes_ledger()
        model = _ScriptedModel([
            _decision("stop", {}, ["route_b"], stop=True, conclusion="Done", evidence_ids=[]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision("stop", {}, ["route_b"], stop=True, conclusion="Route B supported.", evidence_ids=["evidence_0001"]),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=ledger, workspace=tmp, max_cycles=5,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        rejections = [event for event in ledger.events if event.type == "decision_rejected"]
        self.assertEqual(len(rejections), 1)
        self.assertIn("evidence", rejections[0].payload["reason"])

    def test_targetless_discovery_can_complete_with_recorded_goal_evidence(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": {}}, []),
            _decision(
                "stop", {}, [], stop=True, conclusion="The mechanistic hypothesis is supported.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=ResearchLedger(), workspace=tmp,
            ).run("explain an open-ended reaction mechanism")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["target_ids"], [])

    def test_open_discovery_hypothesis_label_can_use_goal_level_evidence(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": {}}, []),
            _decision(
                "stop", {}, ["H2"], stop=True, conclusion="H2 is supported by the measurement.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=ResearchLedger(), workspace=tmp,
            ).run("choose among named mechanism hypotheses")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(result["target_ids"], ["H2"])

    def test_stop_must_satisfy_explicit_goal_criteria(self):
        ledger = _competing_routes_ledger()
        ledger.append("goal_set", {
            "task": "test", "stopping_criteria": ["route inspected", "failure alternatives considered"],
        })
        model = _ScriptedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Incomplete criteria.",
                evidence_ids=["evidence_0001"],
            ),
            {
                **_decision(
                    "stop", {}, ["route_b"], stop=True, conclusion="All criteria satisfied.",
                    evidence_ids=["evidence_0001"],
                ),
                "satisfied_criteria": ["route inspected", "failure alternatives considered"],
            },
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, _FeedbackEnv(), ledger=ledger, workspace=tmp).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        rejection = next(event for event in ledger.events if event.type == "decision_rejected")
        self.assertIn("stopping criteria", rejection.payload["reason"])

    def test_stop_criteria_accept_exact_prefix_followed_by_evidence_detail(self):
        ledger = ResearchLedger()
        ledger.append("goal_set", {
            "task": "test", "stopping_criteria": ["route inspected", "alternatives considered"],
        })
        completed = _decision(
            "stop", {}, [], stop=True, conclusion="The evidence supports the selected mechanism.",
            evidence_ids=["evidence_0001"],
        )
        completed["satisfied_criteria"] = [
            "route inspected: evidence_0001",
            "alternatives considered: H1 and H3 rejected",
        ]
        model = _ScriptedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": {}}, []),
            completed,
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=ledger, workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")

    def test_estimated_time_budget_rejects_unaffordable_action_before_execution(self):
        expensive = _decision(
            "tool", {"tool": "inspect_route", "parameters": {"route_id": "route_a"}}, ["route_a"],
        )
        expensive["estimated_cost"] = {"time_minutes": 10, "compute": "high"}
        model = _ScriptedModel([
            expensive,
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Affordable route inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        env = _FeedbackEnv()
        ledger = _competing_routes_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=ledger, workspace=tmp, max_estimated_minutes=5,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["inspect_route"])
        self.assertEqual(ledger.replay().estimated_minutes_used, 1.0)
        self.assertTrue(any(
            event.type == "decision_rejected" and "budget" in event.payload["reason"]
            for event in ledger.events
        ))

    def test_non_finite_estimated_cost_is_rejected_without_corrupting_budget(self):
        invalid = _decision(
            "tool", {"tool": "inspect_route", "parameters": {"route_id": "route_a"}}, ["route_a"],
        )
        invalid["estimated_cost"] = {"time_minutes": float("nan"), "compute": "low"}
        model = _ScriptedModel([
            invalid,
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Finite-cost action completed.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        ledger = _competing_routes_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, _FeedbackEnv(), ledger=ledger, workspace=tmp).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(math.isfinite(ledger.replay().estimated_minutes_used))
        self.assertEqual(ledger.replay().estimated_minutes_used, 1.0)

    def test_high_risk_action_pauses_for_approval_in_standard_mode(self):
        model = _ScriptedModel([
            _decision("command", {"command": "rm -rf generated-output"}, [], stop=False),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model,
                _FeedbackEnv(),
                ledger=ResearchLedger(),
                workspace=tmp,
                permission_grant=PermissionGrant(mode=PermissionMode.STANDARD),
                max_cycles=2,
            ).run("test")

        self.assertEqual(result["exit_status"], "AwaitingApproval")
        self.assertEqual(result["required_scope"], "system")
        self.assertTrue(any(event.type == "approval_requested" for event in result["events"]))

    def test_existing_score_routes_output_is_registered_in_research_state(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "score_routes", "parameters": {}}, []),
            _decision(
                "stop", {}, ["route_generated"], stop=True,
                conclusion="Generated route is the current computational candidate.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        ledger = ResearchLedger()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _RouteOutputEnv(), ledger=ledger, workspace=tmp, max_cycles=3,
            ).run("test")

        route = ledger.replay().routes["route_generated"]
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(route["reaction_ids"], ["rxn_generated"])
        self.assertEqual(route["scores"]["feasibility"], 0.81)

    def test_agent_can_propose_hypothesis_as_an_internal_research_action(self):
        model = _ScriptedModel([
            _decision("state", {
                "operation": "propose_hypothesis",
                "hypothesis": {
                    "id": "hyp_alt",
                    "claim": "A sulfonamide-first sequence avoids the failed coupling context.",
                    "confidence": 0.45,
                    "predicted_observation": "The amide precursor remains selectively activatable.",
                    "falsification_condition": "Condition screening shows the same selectivity collapse.",
                },
            }, ["route_b"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["hyp_alt", "route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B remains testable.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        ledger = _competing_routes_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, _FeedbackEnv(), ledger=ledger, workspace=tmp).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(ledger.replay().hypotheses["hyp_alt"]["confidence"], 0.45)

    def test_agent_can_author_and_run_python_before_concluding(self):
        model = _ScriptedModel([
            _decision("python", {"code": "print('computed descriptor')"}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="The computation completed.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        ledger = _competing_routes_ledger()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model,
                _FeedbackEnv(),
                ledger=ledger,
                workspace=tmp,
                permission_grant=PermissionGrant(
                    mode=PermissionMode.FULL, scopes={"code_execution"},
                ),
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        finished = [event for event in ledger.events if event.type == "execution_finished"]
        self.assertIn("computed descriptor", finished[0].payload["result"]["stdout"])

    def test_advertised_run_python_tool_alias_uses_code_execution_path(self):
        model = _ScriptedModel([
            _decision(
                "tool",
                {"tool": "run_python", "parameters": {"code": "print('alias executed')"}},
                ["route_b"],
            ),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="The alias computation completed.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model,
                _FeedbackEnv(),
                ledger=_competing_routes_ledger(),
                workspace=tmp,
                permission_grant=PermissionGrant(
                    mode=PermissionMode.FULL, scopes={"code_execution"},
                ),
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        finished = [event for event in result["events"] if event.type == "execution_finished"]
        self.assertIn("alias executed", finished[0].payload["result"]["stdout"])

    def test_execution_aliases_remain_blocked_in_restricted_mode(self):
        aliases = [
            {"tool": "run_python", "parameters": {"code": "print('must not run')"}},
            {"tool": "run_command", "parameters": {"command": "printf must-not-run"}},
        ]
        for action in aliases:
            with self.subTest(tool=action["tool"]), tempfile.TemporaryDirectory() as tmp:
                model = _ScriptedModel([_decision("tool", action, [], stop=False)])
                result = DiscoveryController(
                    model,
                    _FeedbackEnv(),
                    ledger=ResearchLedger(),
                    workspace=tmp,
                    permission_grant=PermissionGrant(mode=PermissionMode.RESTRICTED),
                    max_cycles=1,
                ).run("test")

                self.assertEqual(result["exit_status"], "StepLimit")
                self.assertFalse(any(
                    event.type == "execution_started" for event in result["events"]
                ))
                self.assertTrue(any(
                    event.type == "decision_rejected" and "restricted" in event.payload["reason"]
                    for event in result["events"]
                ))

    def test_malformed_tool_action_is_rejected_without_crashing_loop(self):
        model = _ScriptedModel([
            _decision("tool", {}, ["route_b"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(len([event for event in result["events"] if event.type == "decision_rejected"]), 1)

    def test_rejected_decision_records_bounded_model_diagnostics(self):
        malformed = _decision("tool", "think", "route_b")
        model = _ScriptedModel([
            malformed,
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        rejection = next(event for event in result["events"] if event.type == "decision_rejected")
        self.assertEqual(rejection.payload["decision"]["action"]["tool"], "think")
        self.assertEqual(rejection.payload["decision"]["target_ids"], "route_b")
        self.assertIn('"action": "think"', rejection.payload["model_response"])
        self.assertLessEqual(len(rejection.payload["model_response"]), 4000)

    def test_common_flat_tool_decision_is_normalized_before_validation(self):
        flattened = _decision("tool", "inspect_route", ["route_b"])
        flattened["estimated_cost"] = 0.25
        model = _ScriptedModel([
            flattened,
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(env.calls, [{"tool": "inspect_route", "parameters": {}}])
        self.assertFalse(any(event.type == "decision_rejected" for event in result["events"]))

    def test_schema_rejection_is_visible_to_the_next_decision_cycle(self):
        model = _ScriptedModel([
            _decision("state", "propose_hypothesis", []),
            _decision("tool", {"tool": "inspect_route", "parameters": {}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        rejection = model.observations[1]["state"]["recent_rejections"][0]
        self.assertIn("action", rejection["reason"])
        self.assertEqual(rejection["cycle"], 1)
        self.assertEqual(rejection["action"], "propose_hypothesis")

    def test_non_object_tool_parameters_are_rejected_without_crashing_loop(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": ["bad"]}, ["route_b"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(any(
            event.type == "decision_rejected" and "parameters" in event.payload["reason"]
            for event in result["events"]
        ))

    def test_failed_execution_is_recorded_and_unchanged_retry_is_blocked(self):
        failed = {"tool": "unstable_calculation", "parameters": {"route_id": "route_a"}}
        model = _ScriptedModel([
            _decision("tool", failed, ["route_a"]),
            _decision("tool", failed, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Use route B after solver failure.",
                evidence_ids=["evidence_0002"],
            ),
        ])
        env = _FailingThenInspectEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=_competing_routes_ledger(), workspace=tmp, max_cycles=6,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["unstable_calculation", "inspect_route"])
        self.assertTrue(any(event.type == "action_failed" for event in result["events"]))

    def test_changed_basis_flag_alone_cannot_bypass_failed_action_guard(self):
        failed = {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}
        retry = _decision("tool", failed, ["route_a"])
        retry["changed_basis"] = True
        model = _ScriptedModel([
            _decision("tool", failed, ["route_a"]),
            retry,
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Use route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=_competing_routes_ledger(), workspace=tmp, max_cycles=6,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["feedback_oracle", "inspect_route"])

    def test_old_evidence_cannot_bypass_failed_action_guard(self):
        ledger = _competing_routes_ledger()
        ledger.append("evidence_added", {
            "id": "ev_old", "direction": "observes", "target_ids": ["route_a"],
            "summary": "Evidence created before the failure",
        })
        failed = {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}
        retry = _decision("tool", failed, ["route_a"])
        retry["new_evidence_ids"] = ["ev_old"]
        model = _ScriptedModel([
            _decision("tool", failed, ["route_a"]),
            retry,
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Use route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, env, ledger=ledger, workspace=tmp, max_cycles=6).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["feedback_oracle", "inspect_route"])

    def test_spoofed_decision_targets_cannot_bypass_failed_action_guard(self):
        failed = {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}
        spoofed_retry = _decision("tool", failed, ["route_b"])
        spoofed_retry["new_evidence_ids"] = ["evidence_0002"]
        model = _ScriptedModel([
            _decision("tool", failed, ["route_a", "rxn_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            spoofed_retry,
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Use route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])
        env = _FeedbackEnv()

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=_competing_routes_ledger(), workspace=tmp, max_cycles=6,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual([call["tool"] for call in env.calls], ["feedback_oracle", "inspect_route"])

    def test_fenced_json_decision_is_accepted(self):
        model = _FencedModel([
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _FeedbackEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")

    def test_completion_rejects_evidence_unrelated_to_selected_or_failed_routes(self):
        ledger = _competing_routes_ledger()
        ledger.append("evidence_added", {
            "id": "ev_other", "direction": "observes", "target_ids": ["route_other"],
            "summary": "Unrelated calculation",
        })
        model = _ScriptedModel([
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Unsupported conclusion.",
                evidence_ids=["ev_other"],
            ),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route B inspected.",
                evidence_ids=["evidence_0001"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, _FeedbackEnv(), ledger=ledger, workspace=tmp).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        rejection = next(event for event in ledger.events if event.type == "decision_rejected")
        self.assertIn("relevant", rejection.payload["reason"])

    def test_tool_exception_becomes_failed_evidence_and_loop_recovers(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "exploding_tool", "parameters": {}}, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Recovered with route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _RaisingThenInspectEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(any(event.type == "action_failed" for event in result["events"]))

    def test_malformed_state_event_becomes_failed_evidence_and_loop_recovers(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "malformed_state", "parameters": {}}, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Recovered with route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _MalformedStateEventEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(any(event.type == "action_failed" for event in result["events"]))

    def test_external_tool_cannot_forge_route_state_transition(self):
        ledger = _competing_routes_ledger()
        ledger.append("route_status_changed", {"id": "route_a", "status": "failed"})
        model = _ScriptedModel([
            _decision("tool", {"tool": "forge_state", "parameters": {}}, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Route A remains failed.",
                evidence_ids=["evidence_0002"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(model, _ForgedStateEventEnv(), ledger=ledger, workspace=tmp).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertEqual(ledger.replay().routes["route_a"]["status"], "failed")
        self.assertTrue(any(event.type == "action_failed" for event in result["events"]))

    def test_null_route_collections_become_failed_evidence_and_loop_recovers(self):
        model = _ScriptedModel([
            _decision("tool", {"tool": "null_routes", "parameters": {}}, ["route_a"]),
            _decision("tool", {"tool": "inspect_route", "parameters": {"route_id": "route_b"}}, ["route_b"]),
            _decision(
                "stop", {}, ["route_b"], stop=True, conclusion="Recovered with route B.",
                evidence_ids=["evidence_0002"],
            ),
        ])

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, _NullCollectionsEnv(), ledger=_competing_routes_ledger(), workspace=tmp,
            ).run("test")

        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(any(event.type == "action_failed" for event in result["events"]))

    def test_discovery_v3_runs_active_learning_to_validated_novel_route(self):
        hypotheses = [
            {"id": "H1", "posterior_probability": 0.5},
            {"id": "H2", "posterior_probability": 0.5},
        ]
        reaction = {
            "id": "rxn_ester",
            "product_smiles": "CCOC(=O)C",
            "precursor_smiles_list": ["CCO", "CC(=O)O"],
        }
        route_candidates = [
            {"id": "route_novel", "source": "template", "score": 0.72, "reactions": [reaction]},
            {"id": "route_novel", "source": "mechanism", "score": 0.82, "reactions": [reaction]},
        ]
        model = _ScriptedModel([
            _decision("tool", {"tool": "select_experiment", "parameters": {
                "hypotheses": hypotheses,
                "experiments": [{
                    "id": "exp_kinetic_probe",
                    "outcomes": {
                        "fast": {"H1": 0.9, "H2": 0.1},
                        "slow": {"H1": 0.1, "H2": 0.9},
                    },
                    "cost": {"time_minutes": 15},
                }],
            }}, ["H1", "H2"]),
            _decision("tool", {"tool": "update_hypotheses", "parameters": {
                "hypotheses": hypotheses,
                "likelihoods": {"H1": 0.1, "H2": 0.9},
                "experiment_id": "exp_kinetic_probe",
                "observation": "slow",
            }}, ["H1", "H2"]),
            _decision("tool", {"tool": "build_route_portfolio", "parameters": {
                "target_smiles": "CCOC(=O)C",
                "candidates": route_candidates,
                "known_reactions": ["CCBr.O>>CCO"],
            }}, ["H2"]),
            _decision("tool", {"tool": "validate_route_portfolio", "parameters": {
                "target_smiles": "CCOC(=O)C",
                "routes": [{
                    "id": "route_novel", "target_smiles": "CCOC(C)=O",
                    "source_count": 2, "sources": ["mechanism", "template"],
                    "novelty_score": 0.5, "reactions": [reaction],
                }],
                "mapping_reports": [{
                    "reaction_id": "rxn_ester", "confidence": 0.91,
                    "mapped_reaction": "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>CCOC(=O)C",
                }],
                "forward_predictions": [{
                    "reaction_id": "rxn_ester", "predicted_product": "CCOC(=O)C", "confidence": 0.84,
                }],
                "condition_reports": [{
                    "reaction_id": "rxn_ester", "reagent": "DCC", "solvent": "CH2Cl2", "temperature": 20,
                }],
                "stock_hits": {"CCO": True, "CC(=O)O": True},
                "strict": True,
            }}, ["route_novel"]),
            _decision(
                "stop", {}, ["route_novel"], stop=True,
                conclusion="Route novel passed the independent evidence gate after H2 gained posterior support.",
                evidence_ids=["evidence_0004"],
            ),
        ])
        ledger = ResearchLedger()
        ledger.append("goal_set", {
            "task": "discover and validate a new route",
            "discovery_v3": {
                "min_competing_hypotheses": 2,
                "max_posterior_entropy": 0.5,
                "min_route_novelty": 0.1,
                "require_multi_source_route": True,
                "require_independent_validation": True,
            },
        })
        for hypothesis in hypotheses:
            ledger.append("hypothesis_proposed", hypothesis)
        env = RetroEnvironment(include_bash=False)
        env.register("select_experiment", ExperimentSelectorTool())
        env.register("update_hypotheses", BeliefUpdaterTool())
        env.register("build_route_portfolio", RoutePortfolioTool())
        env.register("validate_route_portfolio", IndependentRouteValidatorTool())

        with tempfile.TemporaryDirectory() as tmp:
            result = DiscoveryController(
                model, env, ledger=ledger, workspace=tmp, max_cycles=6,
                allow_execution_tools=False,
            ).run("discover and validate a new route")

        state = ledger.replay()
        metrics = score_discovery_v3_events(result["trajectory"])
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertAlmostEqual(state.hypotheses["H2"]["posterior_probability"], 0.9)
        self.assertEqual(state.routes["route_novel"]["validation_status"], "passed")
        self.assertTrue(metrics["discovery_v3_pass"], metrics)


if __name__ == "__main__":
    unittest.main()

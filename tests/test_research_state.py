import json
import tempfile
import unittest
from pathlib import Path

from retroagent.research_state import ResearchEvent, ResearchLedger


class ResearchStateTests(unittest.TestCase):
    def test_replay_builds_goal_hypotheses_routes_evidence_and_decisions(self):
        ledger = ResearchLedger()
        ledger.append("goal_set", {"task": "discover a route", "target_smiles": "CC=O"})
        ledger.append("hypothesis_proposed", {
            "id": "hyp_1",
            "claim": "Route A should work",
            "confidence": 0.6,
            "falsification_condition": "key coupling fails",
        })
        ledger.append("route_registered", {
            "id": "route_a",
            "reaction_ids": ["rxn_a"],
            "status": "active",
            "scores": {"feasibility": 0.8, "uncertainty": 0.3},
        })
        ledger.append("evidence_added", {
            "id": "ev_1",
            "direction": "supports",
            "summary": "Analog precedent",
            "target_ids": ["hyp_1", "route_a"],
        })
        ledger.append("decision_made", {
            "research_question": "Test the coupling",
            "action": {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}},
        })

        state = ledger.replay()

        self.assertEqual(state.goal["target_smiles"], "CC=O")
        self.assertEqual(state.hypotheses["hyp_1"]["claim"], "Route A should work")
        self.assertEqual(state.routes["route_a"]["reaction_ids"], ["rxn_a"])
        self.assertEqual(state.evidence["ev_1"]["direction"], "supports")
        self.assertEqual(len(state.decisions), 1)
        self.assertEqual(state.event_count, 5)

    def test_jsonl_ledger_round_trips_and_continues_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.jsonl"
            ledger = ResearchLedger(path)
            first = ledger.append("goal_set", {"task": "test"}, actor="user", cycle=0)

            loaded = ResearchLedger(path)
            second = loaded.append("hypothesis_proposed", {"id": "hyp_1", "claim": "x"})

            self.assertEqual(first.id, "event_000001")
            self.assertEqual(second.id, "event_000002")
            self.assertEqual(len(path.read_text().splitlines()), 2)
            self.assertEqual(loaded.replay().event_count, 2)

    def test_duplicate_event_id_is_rejected(self):
        ledger = ResearchLedger()
        event = ResearchEvent(
            id="event_external",
            type="goal_set",
            timestamp="2026-07-11T00:00:00+00:00",
            cycle=0,
            actor="user",
            payload={"task": "test"},
        )
        ledger.append_event(event)

        with self.assertRaisesRegex(ValueError, "Duplicate event id"):
            ledger.append_event(event)

    def test_generated_id_skips_existing_external_sequence_gap(self):
        ledger = ResearchLedger()
        ledger.append_event(ResearchEvent(
            id="event_000002",
            type="goal_set",
            timestamp="2026-07-11T00:00:00+00:00",
            cycle=0,
            actor="user",
            payload={"task": "test"},
        ))

        generated = ledger.append("hypothesis_proposed", {"id": "hyp_1", "claim": "x"})

        self.assertEqual(generated.id, "event_000001")
        following = ledger.append("subgoal_set", {"objective": "next"})
        self.assertEqual(following.id, "event_000003")

    def test_observation_is_compact_and_exposes_replan_condition(self):
        ledger = ResearchLedger()
        ledger.append("goal_set", {"task": "test"})
        ledger.append("route_registered", {
            "id": "route_a", "reaction_ids": ["rxn_a"], "status": "challenged",
        })
        ledger.append("replan_required", {"reason": "negative feedback"})

        observation = ledger.replay().to_observation()

        self.assertTrue(observation["replan_required"])
        self.assertEqual(observation["routes"][0]["id"], "route_a")
        self.assertNotIn("raw_events", observation)

    def test_observation_summarizes_large_evidence_result_without_mutating_ledger(self):
        ledger = ResearchLedger()
        ledger.append("evidence_added", {
            "id": "ev_large",
            "summary": "Route graph computed",
            "target_ids": ["route_a"],
            "result": {"route_graph": {"blob": "x" * 5000}, "routes": [1, 2]},
        })

        state = ledger.replay()
        observed = state.to_observation()["recent_evidence"][0]

        self.assertNotIn("result", observed)
        self.assertEqual(observed["result_keys"], ["route_graph", "routes"])
        self.assertGreater(observed["result_chars"], 5000)
        self.assertIn("result", state.evidence["ev_large"])

    def test_registration_cannot_resurrect_failed_route_or_reaction(self):
        ledger = ResearchLedger()
        ledger.append("route_registered", {
            "id": "route_a", "reaction_ids": ["rxn_a"], "status": "active",
            "scores": {"feasibility": 0.8},
        })
        ledger.append("reaction_status_changed", {"id": "rxn_a", "status": "failed"})
        ledger.append("route_status_changed", {
            "id": "route_a", "status": "failed", "scores": {"feasibility": 0.0},
        })
        ledger.append("reaction_registered", {"id": "rxn_a", "status": "active", "score": 0.9})
        ledger.append("route_registered", {
            "id": "route_a", "reaction_ids": ["rxn_a"], "status": "active",
            "scores": {"feasibility": 0.95},
        })

        state = ledger.replay()

        self.assertEqual(state.reactions["rxn_a"]["status"], "failed")
        self.assertEqual(state.routes["route_a"]["status"], "failed")
        self.assertEqual(state.routes["route_a"]["scores"]["feasibility"], 0.0)

    def test_ledger_payload_is_isolated_from_caller_mutation(self):
        payload = {"task": "original", "nested": {"value": 1}}
        ledger = ResearchLedger()
        returned = ledger.append("goal_set", payload)

        payload["nested"]["value"] = 2
        returned.payload["nested"]["value"] = 3
        exposed = ledger.events[0]
        exposed.payload["nested"]["value"] = 4

        self.assertEqual(ledger.replay().goal["nested"]["value"], 1)

    def test_replay_reports_semantic_corruption_location(self):
        ledger = ResearchLedger()
        ledger.append("goal_set", {"task": "valid"})
        ledger.append("hypothesis_proposed", {"claim": "missing id"})

        with self.assertRaisesRegex(ValueError, "event index 1.*last valid index 0"):
            ledger.replay()

    def test_duplicate_evidence_id_is_replay_corruption(self):
        ledger = ResearchLedger()
        ledger.append("evidence_added", {"id": "ev_1", "summary": "first"})
        ledger.append("evidence_added", {"id": "ev_1", "summary": "second"})

        with self.assertRaisesRegex(ValueError, "Duplicate evidence id"):
            ledger.replay()


if __name__ == "__main__":
    unittest.main()

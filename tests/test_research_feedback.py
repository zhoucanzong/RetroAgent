import unittest

from retroagent.research_feedback import EvidenceAssimilator, action_fingerprint
from retroagent.research_state import ResearchLedger


def _ledger_with_competing_routes() -> ResearchLedger:
    ledger = ResearchLedger()
    ledger.append("goal_set", {"task": "discover", "target_smiles": "CC=O"})
    ledger.append("route_registered", {
        "id": "route_a",
        "reaction_ids": ["rxn_a", "rxn_shared"],
        "status": "active",
        "scores": {"feasibility": 0.9, "evidence_strength": 0.7, "uncertainty": 0.2},
    })
    ledger.append("route_registered", {
        "id": "route_b",
        "reaction_ids": ["rxn_b", "rxn_shared"],
        "status": "active",
        "scores": {"feasibility": 0.7, "evidence_strength": 0.5, "uncertainty": 0.4},
    })
    return ledger


class ResearchFeedbackTests(unittest.TestCase):
    def test_nonrepairable_failure_invalidates_only_affected_routes(self):
        ledger = _ledger_with_competing_routes()
        action = {"tool": "feedback_oracle", "parameters": {"reaction_id": "rxn_a"}}

        EvidenceAssimilator().record_failure(
            ledger,
            reaction_ids=["rxn_a"],
            reason="chemoselectivity failure",
            category="selectivity",
            repairable=False,
            action=action,
            cycle=2,
        )
        state = ledger.replay()

        self.assertEqual(state.reactions["rxn_a"]["status"], "failed")
        self.assertEqual(state.routes["route_a"]["status"], "failed")
        self.assertEqual(state.routes["route_b"]["status"], "active")
        self.assertEqual(state.routes["route_a"]["scores"]["feasibility"], 0.0)
        self.assertEqual(state.routes["route_a"]["scores"]["uncertainty"], 1.0)
        self.assertIn(action_fingerprint(action), state.failed_action_fingerprints)
        self.assertTrue(state.replan_required)
        self.assertEqual(len(state.evidence), 1)

    def test_repairable_failure_challenges_and_downweights_route(self):
        ledger = _ledger_with_competing_routes()

        EvidenceAssimilator().record_failure(
            ledger,
            reaction_ids=["rxn_a"],
            reason="condition window too narrow",
            category="conditions",
            repairable=True,
            cycle=1,
        )
        route = ledger.replay().routes["route_a"]

        self.assertEqual(route["status"], "challenged")
        self.assertLess(route["scores"]["feasibility"], 0.9)
        self.assertLess(route["scores"]["evidence_strength"], 0.7)
        self.assertGreater(route["scores"]["uncertainty"], 0.2)

    def test_action_fingerprint_is_stable_but_changes_with_parameters(self):
        first = {"tool": "propose", "parameters": {"b": 2, "a": 1}}
        reordered = {"parameters": {"a": 1, "b": 2}, "tool": "propose"}
        changed = {"tool": "propose", "parameters": {"a": 1, "b": 3}}

        self.assertEqual(action_fingerprint(first), action_fingerprint(reordered))
        self.assertNotEqual(action_fingerprint(first), action_fingerprint(changed))

    def test_failure_evidence_id_skips_existing_sequence_gap(self):
        ledger = _ledger_with_competing_routes()
        ledger.append("evidence_added", {
            "id": "evidence_0002", "direction": "observes", "target_ids": ["route_b"],
        })

        EvidenceAssimilator().record_failure(
            ledger,
            reaction_ids=["rxn_a"],
            reason="failed",
            category="feasibility",
            repairable=False,
        )

        self.assertIn("evidence_0001", ledger.replay().evidence)


if __name__ == "__main__":
    unittest.main()

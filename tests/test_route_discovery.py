import json
import unittest

from retroagent.tools.route_discovery import (
    IndependentRouteValidatorTool,
    RoutePortfolioTool,
    build_route_portfolio,
    validate_route_portfolio,
)


TARGET = "CCOC(=O)C"
REACTION = {
    "id": "rxn_ester",
    "product_smiles": TARGET,
    "precursor_smiles_list": ["CCO", "CC(=O)O"],
}


def _portfolio():
    return build_route_portfolio(
        target_smiles=TARGET,
        candidates=[
            {"id": "route_template", "source": "template", "score": 0.72, "reactions": [REACTION]},
            {"id": "route_mechanism", "source": "mechanism", "score": 0.82, "reactions": [REACTION]},
        ],
        known_reactions=["CCBr.O>>CCO"],
    )


class RouteDiscoveryTests(unittest.TestCase):
    def test_portfolio_merges_equivalent_routes_and_preserves_source_disagreement(self):
        result = _portfolio()
        route = result["routes"][0]

        self.assertEqual(result["route_portfolio"]["candidate_count"], 2)
        self.assertEqual(result["route_portfolio"]["unique_route_count"], 1)
        self.assertEqual(route["sources"], ["mechanism", "template"])
        self.assertEqual(route["source_count"], 2)
        self.assertGreater(route["model_disagreement"], 0)
        self.assertGreaterEqual(route["novelty_score"], 0)
        self.assertLessEqual(route["novelty_score"], 1)
        self.assertIn("uncertainty", route["scores"])

    def test_portfolio_keeps_distinct_disconnections_separate(self):
        result = build_route_portfolio(
            target_smiles=TARGET,
            candidates=[
                {"id": "esterification", "source": "template", "score": 0.7, "reactions": [REACTION]},
                {
                    "id": "acyl_chloride",
                    "source": "template_free",
                    "score": 0.6,
                    "reactions": [{
                        "id": "rxn_acyl_chloride",
                        "product_smiles": TARGET,
                        "precursor_smiles_list": ["CCO", "CC(=O)Cl"],
                    }],
                },
            ],
        )

        self.assertEqual(result["route_portfolio"]["unique_route_count"], 2)
        self.assertEqual({route["id"] for route in result["routes"]}, {"esterification", "acyl_chloride"})

    def test_portfolio_rejects_candidate_that_does_not_reach_target(self):
        with self.assertRaisesRegex(ValueError, "does not match target"):
            build_route_portfolio(
                target_smiles=TARGET,
                candidates=[{
                    "id": "wrong",
                    "source": "mechanism",
                    "reactions": [{"product_smiles": "CCO", "precursor_smiles_list": ["CC"]}],
                }],
            )

    def test_strict_validator_requires_independent_evidence(self):
        route = _portfolio()["routes"][0]
        result = validate_route_portfolio(target_smiles=TARGET, routes=[route], strict=True)

        self.assertEqual(result["validation_reports"][0]["status"], "needs-evidence")
        self.assertGreater(result["summary"]["needs_evidence_count"], 0)

    def test_strict_validator_passes_complete_independent_evidence(self):
        route = _portfolio()["routes"][0]
        result = validate_route_portfolio(
            target_smiles=TARGET,
            routes=[route],
            mapping_reports=[{
                "reaction_id": "rxn_ester",
                "confidence": 0.91,
                "mapped_reaction": "[CH3:1][CH2:2][OH:3].[CH3:4][C:5](=[O:6])[OH:7]>>CCOC(=O)C",
            }],
            forward_predictions=[{
                "reaction_id": "rxn_ester", "predicted_product": TARGET, "confidence": 0.84,
            }],
            condition_reports=[{
                "reaction_id": "rxn_ester", "reagent": "DCC", "solvent": "CH2Cl2", "temperature": 20,
            }],
            stock_hits={"CCO": True, "CC(=O)O": {"in_stock": True}},
            strict=True,
        )

        report = result["validation_reports"][0]
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["confidence"], 1.0)
        self.assertEqual(result["routes"][0]["validation_status"], "passed")

    def test_validator_fails_contradictory_forward_prediction(self):
        route = _portfolio()["routes"][0]
        result = validate_route_portfolio(
            target_smiles=TARGET,
            routes=[route],
            forward_predictions=[{
                "reaction_id": "rxn_ester", "predicted_product": "CCO", "confidence": 0.9,
            }],
            required_checks=["structure", "target_coverage", "forward_prediction"],
        )

        self.assertEqual(result["validation_reports"][0]["status"], "failed")

    def test_tools_return_machine_readable_contracts(self):
        portfolio_tool = RoutePortfolioTool()
        validator_tool = IndependentRouteValidatorTool()
        portfolio = json.loads(portfolio_tool.execute({
            "target_smiles": TARGET,
            "candidates": [{"id": "route", "source": "template", "reactions": [REACTION]}],
        }))
        validation = json.loads(validator_tool.execute({
            "target_smiles": TARGET,
            "routes": portfolio["routes"],
        }))

        self.assertEqual(portfolio["route_portfolio"]["unique_route_count"], 1)
        self.assertIn("validation_reports", validation)
        self.assertIn("required", portfolio_tool.parameters_schema())
        self.assertIn("required", validator_tool.parameters_schema())


if __name__ == "__main__":
    unittest.main()

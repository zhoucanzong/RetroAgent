import json
import unittest

from retroagent.tools.evaluate import EvaluationTool
from retroagent.tools.propose import ProposalTool


class ReactionSchemaTests(unittest.TestCase):
    def test_proposal_record_uses_canonical_schema_with_legacy_alias(self):
        reaction = ProposalTool._build_reaction_record(
            product_smiles="CC(=O)Oc1ccccc1C(=O)O",
            precursor_smiles_list=["CC(=O)Cl", "O=C(O)c1ccccc1O"],
            template_index=123,
            classification="Acylation",
            library_occurrence=7,
        )

        self.assertEqual(reaction["product_smiles"], "CC(=O)Oc1ccccc1C(=O)O")
        self.assertEqual(reaction["precursor_smiles_list"], ["CC(=O)Cl", "O=C(O)c1ccccc1O"])
        self.assertEqual(reaction["precursors"], reaction["precursor_smiles_list"])
        self.assertEqual(reaction["template_index"], 123)

    def test_evaluate_normalizes_legacy_precursors_field(self):
        normalized = EvaluationTool._normalize_reaction(
            {"precursors": ["CCO"], "product": "CC=O"},
            target="fallback-target",
            index=2,
        )

        self.assertEqual(normalized["id"], "rxn_2")
        self.assertEqual(normalized["product_smiles"], "CC=O")
        self.assertEqual(normalized["precursor_smiles_list"], ["CCO"])

    def test_evaluate_accepts_proposal_style_reactions(self):
        tool = EvaluationTool.__new__(EvaluationTool)
        tool._predict_feasibility = lambda product, precursors: 0.75
        tool._check_stock_availability = lambda precursors: 0.5
        tool._trivial_precursor_score = lambda precursors: 0.0

        reaction = ProposalTool._build_reaction_record(
            product_smiles="CC=O",
            precursor_smiles_list=["CCO"],
            template_index=1,
            classification="Oxidation",
            library_occurrence=10,
        )
        reaction["id"] = "rxn_test"

        result = json.loads(tool.execute({"reactions": [reaction]}))

        self.assertIn("rxn_test", result["scores"])
        self.assertEqual(result["scores"]["rxn_test"]["precursor_count"], 1)
        self.assertEqual(result["scores"]["rxn_test"]["feasibility"], 0.75)
        self.assertEqual(result["scores"]["rxn_test"]["stock_availability"], 0.5)
        self.assertEqual(result["scores"]["rxn_test"]["total"], 0.65)


if __name__ == "__main__":
    unittest.main()

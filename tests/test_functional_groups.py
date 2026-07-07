import json
import unittest

from rdkit import Chem

from retroagent.tools.disconnect import DisconnectionTool
from retroagent.tools.functional_groups import (
    FunctionalGroupsTool,
    summarize_broad_functional_groups,
)


class FunctionalGroupTests(unittest.TestCase):
    def test_aspirin_functional_groups_are_not_cross_talked(self):
        result = json.loads(FunctionalGroupsTool().execute({
            "smiles": "CC(=O)Oc1ccccc1C(=O)O"
        }))

        present = result["functional_groups_present"]
        self.assertIn("carboxylic_acid", present)
        self.assertIn("ester", present)
        self.assertNotIn("phenol", present)
        self.assertNotIn("alcohol_primary", present)
        self.assertNotIn("alcohol_secondary", present)
        self.assertNotIn("alcohol_tertiary", present)
        self.assertNotIn("aldehyde", present)
        self.assertNotIn("ketone", present)
        self.assertNotIn("ether", present)

    def test_disconnect_uses_broad_groups_without_aspirin_false_positives(self):
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        result = DisconnectionTool._analyze_functional_groups(mol)

        self.assertIn("carboxylic_acid", result["present"])
        self.assertIn("ester", result["present"])
        self.assertIn("phenol", result["absent"])
        self.assertIn("alcohol", result["absent"])
        self.assertIn("aldehyde", result["absent"])
        self.assertIn("ketone", result["absent"])
        self.assertIn("ether", result["absent"])

    def test_representative_broad_groups(self):
        examples = {
            "CCO": ("alcohol",),
            "c1ccccc1O": ("phenol",),
            "CC(=O)C": ("ketone",),
            "CC=O": ("aldehyde",),
            "CCOC": ("ether",),
            "CCN": ("amine",),
            "CC(=O)N": ("amide",),
        }

        for smiles, expected_groups in examples.items():
            with self.subTest(smiles=smiles):
                mol = Chem.MolFromSmiles(smiles)
                summary = summarize_broad_functional_groups(mol)
                for group in expected_groups:
                    self.assertIn(group, summary["present"])


if __name__ == "__main__":
    unittest.main()

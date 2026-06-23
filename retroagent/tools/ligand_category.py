"""LigandCategoryTool: classify chiral ligand/catalyst scaffolds.

Uses substructure patterns to recognize common phosphine, NHC, and oxazoline
ligand families. Also reports denticity and coordinating atoms.
"""

import json
from rdkit import Chem
from rdkit.Chem import Descriptors


class LigandCategoryTool:
    name = "classify_ligand"
    description = (
        "Classify a chiral ligand or catalyst by scaffold family, denticity, "
        "and coordinating atoms. Useful for matching target molecules to known "
        "privileged ligand classes (BINAP, BIDIME, BIBOP, Josiphos, etc.)."
    )

    # Substructure SMARTS patterns for known ligand families
    SCAFFOLD_PATTERNS = {
        "BINAP": "c1ccc(-c2ccccc2-c2c(-c3ccccc3)ccc3ccccc23)cc1",
        "BIPHEP": "c1ccc(-c2ccccc2-c2ccccc2-c2ccccc2)cc1",
        "SEGPHOS": "c1ccc2c(c1)Oc1ccccc1O2",
        "DIFLUORPHOS": "FC(F)(F)c1ccccc1-c1ccccc1-c1ccccc1C(F)(F)F",
        "BIDIME": "c1cc(C(C)(C)C)cc(P)c1-c1c(P)cc(C(C)(C)C)cc1",
        "BIBOP": "c1cc(C(C)(C)C)cc(-c2c(-c3cc(C(C)(C)C)cc3)oc3ccccc23)c1",
        "JOSIPHOS": "[P]-c1ccccc1[P]-[Fe]",
        "WALPHOS": "[P]-c1ccccc1-c2ccccc2[P]",
        "PHOX": "[P]-c1ccccc1-c2ccccc2N1C=CO1",
        "BOX": "C1=N[C@@H](C2=NCO2)CO1",
        "PYBOX": "c1cc(-c2n3c(co2)C2C=NCO2)n(-c2n3c(co2)C2C=NCO2)c1",
        "NHC": "[n+]1ccn(-c2ccccc2)c1",
        "TUNEPHOS": "c1ccc2c(c1)Oc1ccccc1O2",
        "PHANEPHOS": "c1cc2ccc1-c1ccc(cc1)C2",
        "DIOP": "CC(C)(C)OCC(COc1ccccc1)Oc1ccccc1",
    }

    def execute(self, parameters: dict) -> str:
        smiles = parameters.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"}, ensure_ascii=False)

        matches = self._match_scaffolds(mol)
        denticity, donor_atoms = self._analyze_denticity(mol)

        result = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "matched_scaffolds": matches,
            "primary_scaffold": matches[0] if matches else "Unknown",
            "denticity": denticity,
            "coordinating_atoms": donor_atoms,
            "has_phosphorus": any(a.GetSymbol() == "P" for a in mol.GetAtoms()),
            "has_nitrogen": any(a.GetSymbol() == "N" for a in mol.GetAtoms()),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "num_aliphatic_rings": Descriptors.NumAliphaticRings(mol),
        }
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Input ligand SMILES"}
            },
            "required": ["smiles"]
        }

    def _match_scaffolds(self, mol: Chem.Mol) -> list[str]:
        matched = []
        for name, smarts in self.SCAFFOLD_PATTERNS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat and mol.HasSubstructMatch(pat):
                matched.append(name)
        return matched

    def _analyze_denticity(self, mol: Chem.Mol) -> tuple[str, list[dict]]:
        donor_atoms = []
        donor_symbols = ["P", "N", "O", "S", "C"]
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in donor_symbols:
                # Simple heuristic: lone-pair-bearing atoms are potential donors
                # P, N, O, S always count; C only if it's part of NHC/carbene (skipping here)
                if atom.GetSymbol() != "C":
                    donor_atoms.append({
                        "index": int(atom.GetIdx()),
                        "element": atom.GetSymbol(),
                        "aromatic": atom.GetIsAromatic(),
                    })

        count = len(donor_atoms)
        if count == 0:
            return "none", []
        if count == 1:
            return "monodentate", donor_atoms
        if count == 2:
            return "bidentate", donor_atoms
        if count == 3:
            return "tridentate", donor_atoms
        return "tetradentate_or_higher", donor_atoms

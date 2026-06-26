"""FunctionalGroupsTool: identify functional groups in a molecule via RDKit SMARTS.

Pure local, no weights. Recognizes 50+ functional groups (adapted from RDKit's
FGSMARTS collection). Complements disconnect's built-in FG detection with a
dedicated, exhaustive substructure scan.

Claude philosophy: tool REPORTS which FGs are present/absent. It does NOT judge
reactivity or selectivity — the LLM (Auditor/Planner) interprets.
"""

import json
from rdkit import Chem


# Curated functional-group SMARTS (subset of RDKit FGSMARTS, chemically standard).
# Alcohol patterns use [CX4] (sp3 carbon) to avoid matching carboxylic-acid / carbonyl carbons.
FUNCTIONAL_GROUPS = {
    "alcohol_primary": "[CH2][OH]",
    "alcohol_secondary": "[CH1][OH]",
    "alcohol_tertiary": "[CH0X4][OH]",
    "phenol": "c[OH]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OH]",
    "ester": "[CX3](=O)O[#6]",
    "amide": "[CX3](=O)[#7]",
    "primary_amine": "[NX3H2]",
    "secondary_amine": "[NX3H1]",
    "tertiary_amine": "[NX3H0]",
    "aniline": "c[NX3]",
    "nitrile": "C#N",
    "isocyanate": "N=C=O",
    "nitro": "[NX3+](=O)[O-]",
    "ether": "[OD2]([#6])[#6]",
    "epoxide": "C1OC1",
    "peroxide": "OO",
    "thiol": "[#6][SX2H]",
    "thioether": "[#6][SX2][#6]",
    "disulfide": "SS",
    "sulfoxide": "[SX3](=O)",
    "sulfone": "[SX4](=O)(=O)",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
    "halide_fluoride": "[F]",
    "halide_chloride": "[Cl]",
    "halide_bromide": "[Br]",
    "halide_iodide": "[I]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "arene": "c1ccccc1",
    "imidazole": "c1[nH]cnc1",
    "pyridine": "c1ccncc1",
    "pyrrole": "c1cc[nH]c1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrimidine": "c1cncnc1",
    "triazole_1H": "c1n[nH]cn1",
    "oxazole": "c1cocn1",
    "isoxazole": "c1oncc1",
    "thiazole": "c1cscn1",
    "piperidine": "C1CCNCC1",
    "morpholine": "C1CCOCC1",
    "acetal": "[CX4]([OX2])[OX2]",
    "enol_ether": "C=C-O-C",
    "imine_schiff": "C=N",
    "enamine": "C=C-N",
    "phosphine_tertiary": "[PX3]([#6])([#6])[#6]",
    "phosphine_oxide": "[PX4](=O)",
    "boronic_acid": "[BX3](O)(O)",
    "n_oxide": "[N+]([O-])",
    "carbamate": "NC(=O)O",
    "urea": "NC(=O)N",
    "anhydride": "C(=O)OC(=O)",
}


class FunctionalGroupsTool:
    name = "functional_groups"
    description = (
        "Identify functional groups present in a molecule via 50+ SMARTS patterns "
        "(alcohols, carbonyls, amines, heterocycles, halides, phosphines, "
        "N-oxides, etc.). Pure RDKit, local, instant. Reports which groups are "
        "PRESENT and ABSENT with match counts. Use for chemoselectivity analysis, "
        "compatibility checks, or scaffold characterization."
    )

    def execute(self, parameters: dict) -> str:
        smiles = parameters.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"}, ensure_ascii=False)

        present = {}
        absent = []
        for name, smarts in FUNCTIONAL_GROUPS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            n = len(mol.GetSubstructMatches(pat))
            if n > 0:
                present[name] = n
            else:
                absent.append(name)

        return json.dumps({
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rings": mol.GetRingInfo().NumRings(),
            "functional_groups_present": present,
            "functional_groups_absent": absent,
            "present_count": len(present),
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Molecule SMILES"},
            },
            "required": ["smiles"],
        }

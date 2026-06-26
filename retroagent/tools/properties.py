"""PropertiesTool: physicochemical property calculation via RDKit descriptors.

Calculates drug-likeness / physicochemical descriptors: MW, MolLogP, TPSA,
H-bond donors/acceptors, rotatable bonds, fraction sp3, aromatic/aliphatic rings,
Lipinski violations, etc. Pure RDKit, no weights, instant.

NOTE: This is a DESCRIPTOR-based tool (deterministic, transparent), not an ML
toxicity/solubility predictor. ML-based property prediction (Uni-Mol) was
investigated but blocked by environment (unicore C++ ext unavailable on py3.14;
unimol-tools ships backbone only). When a pretrained ML property model becomes
installable, it can be added alongside this tool.

Claude philosophy: tool REPORTS descriptor values. It does NOT decide 'drug-like'
or 'toxic' — the LLM interprets the numbers.
"""

import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors


class PropertiesTool:
    name = "molecule_properties"
    description = (
        "Calculate physicochemical descriptors of a molecule via RDKit: "
        "molecular weight, MolLogP (calculated lipophilicity), TPSA (polar surface "
        "area), H-bond donors/acceptors, rotatable bonds, fraction sp3, aromatic/"
        "aliphatic rings, formal charge, and Lipinski rule-of-5 violations. "
        "Instant, no weights. Use to assess drug-likeness, lipophilicity, "
        "permeability hints. Reports numbers only — you interpret."
    )

    def execute(self, parameters: dict) -> str:
        smiles = parameters.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"}, ensure_ascii=False)

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        n_aliph = rdMolDescriptors.CalcNumAliphaticRings(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        charge = Chem.GetFormalCharge(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        n_rings = mol.GetRingInfo().NumRings()
        n_hetero = Lipinski.NumHeteroatoms(mol)

        # Lipinski violations (MW<=500, logP<=5, HBD<=5, HBA<=10)
        violations = []
        if mw > 500: violations.append("MW>500")
        if logp > 5: violations.append("logP>5")
        if hbd > 5: violations.append("HBD>5")
        if hba > 10: violations.append("HBA>10")

        # PAINS-free heuristic flags (not exhaustive — just common reactivity alerts)
        alert_smarts = {
            "acyl_halide": "C(=O)[F,Cl,Br,I]",
            "aldehyde": "[CX3H1](=O)",
            "isocyanate": "N=C=O",
            "peroxide": "OO",
            "unstable_nitro": "[N+](=O)[O-]",
            "thiol": "[SX2H]",
        }
        reactive_alerts = []
        for name, s in alert_smarts.items():
            pat = Chem.MolFromSmarts(s)
            if pat and mol.HasSubstructMatch(pat):
                reactive_alerts.append(name)

        return json.dumps({
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "descriptors": {
                "molecular_weight": round(mw, 2),
                "mol_logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "h_bond_donors": hbd,
                "h_bond_acceptors": hba,
                "rotatable_bonds": rotatable,
                "fraction_sp3": round(fsp3, 3),
                "aromatic_rings": n_arom,
                "aliphatic_rings": n_aliph,
                "stereocenters": n_stereo,
                "formal_charge": charge,
                "heavy_atoms": n_heavy,
                "rings": n_rings,
                "heteroatoms": n_hetero,
            },
            "drug_likeness": {
                "lipinski_violations": violations,
                "lipinski_passes": len(violations) == 0,
                "veber_passes": (rotatable <= 10 and tpsa <= 140),  # oral bioavailability hint
            },
            "reactivity_alerts": reactive_alerts,
            "note": ("RDKit descriptors (deterministic). logP/TPSA are calculated, "
                     "not measured. For ML toxicity/solubility prediction, see "
                     "CHEMMCP_INTEGRATION.md Uni-Mol note (currently unavailable)."),
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Molecule SMILES"},
            },
            "required": ["smiles"],
        }

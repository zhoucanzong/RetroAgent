"""CatalystTool: a pure CALCULATOR for organometallic catalyst descriptors.

Claude-agent philosophy: this tool NEVER decides whether a catalyst is "good"
or "bad". It only COMPUTES and REPORTS objective chemical facts — coordination
number, d-electron count, labile-site estimate, symmetry candidates, oxidation
state validity. All chemical JUDGMENT (is this catalytically viable? should we
reject?) stays in the LLM (the Design Auditor).

This solves the organometallic-SMILES representation problem: instead of forcing
the model to write a single metal-complex SMILES (which RDKit and LLMs both
struggle with), the model emits a structured spec — organic ligand SMILES plus
metal/geometry fields — and this tool returns computed facts for the Auditor.
"""

import json
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# Common geometry → expected coordination number
GEOMETRY_CN = {
    "linear": 2,
    "trigonal_planar": 3,
    "tetrahedral": 4,
    "square_planar": 4,
    "trigonal_bipyramidal": 5,
    "square_pyramidal": 5,
    "octahedral": 6,
    "pentagonal_bipyramidal": 7,
}

# Group number (valence electrons) for common metals — used for d-electron count
# Oxidation state OS → d electrons = group - OS
METAL_GROUPS = {
    "Sc": 3, "Y": 3, "La": 3,
    "Ti": 4, "Zr": 4, "Hf": 4,
    "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 6, "Mo": 6, "W": 6,
    "Mn": 7, "Tc": 7, "Re": 7,
    "Fe": 8, "Ru": 8, "Os": 8,
    "Co": 9, "Rh": 9, "Ir": 9,
    "Ni": 10, "Pd": 10, "Pt": 10,
    "Cu": 11, "Ag": 11, "Au": 11,
    "Zn": 12, "Cd": 12, "Hg": 12,
    "Al": 3, "Ga": 3, "In": 3,
    "Sn": 4, "Pb": 4, "B": 3,
    "Li": 1, "Na": 1, "K": 1, "Mg": 2, "Ca": 2,
}

# Common/acceptable oxidation states (loose; reported as a sanity hint, not a verdict)
COMMON_OXIDATION_STATES = {
    "Sc": {3}, "Y": {3}, "La": {3},
    "Ti": {2, 3, 4}, "Zr": {4}, "Hf": {4},
    "V": {2, 3, 4, 5},
    "Cr": {2, 3, 6}, "Mo": {4, 6}, "W": {4, 6},
    "Mn": {2, 7},
    "Fe": {2, 3}, "Ru": {2, 3, 4}, "Os": {2, 3, 4, 8},
    "Co": {2, 3}, "Rh": {1, 3}, "Ir": {1, 3},
    "Ni": {0, 2}, "Pd": {0, 2}, "Pt": {0, 2, 4},
    "Cu": {1, 2}, "Ag": {1}, "Au": {1, 3},
    "Zn": {2}, "Al": {3}, "B": {3}, "Sn": {2, 4}, "Pb": {2, 4},
    "Li": {1}, "Na": {1}, "K": {1}, "Mg": {2}, "Ca": {2},
}

# Ligands commonly considered labile (weakly bound, easily displaced) — used to
# estimate how many coordination sites can open up for substrate binding.
# This is a HINT, not a verdict; the model decides relevance.
LABILE_SMARTS = [
    "CC#N",                  # acetonitrile
    "[OH2]",                 # water
    "O",                     # water (alt)
    "CO",                    # methanol
    "[Cl-]", "[Br-]", "[I-]",  # halides (context-dependent; reported as possibly labile)
    "C1CCOC1",               # THF
    "O=C=O",                 # CO2
    "[C-]#[O+]",             # CO
    "N",                     # ammonia
    "[N+](=O)([O-])",        # nitrate
    "O=S(=O)(C)(C)",         # DMSO-like
]


class CatalystTool:
    name = "design_catalyst"
    description = (
        "Compute objective chemical facts about an organometallic catalyst from a "
        "STRUCTURED descriptor (not a single SMILES). Input: metal, oxidation_state, "
        "geometry, and a list of ligands (each with SMILES, denticity, donor_atoms, "
        "and count). Returns computed facts: total coordination number, d-electron "
        "count, 18-electron-rule status, estimated labile-site count, symmetry "
        "candidates, and oxidation-state plausibility. This tool REPORTS facts only "
        "— it never judges whether the catalyst is good or bad. Use the Design "
        "Auditor (LLM) to interpret these facts."
    )

    def execute(self, parameters: dict) -> str:
        metal = parameters.get("metal", "").strip()
        oxidation_state = parameters.get("oxidation_state")
        geometry = parameters.get("geometry", "").strip()
        ligands = parameters.get("ligands", [])
        counterion = parameters.get("counterion", "")
        chirality_source = parameters.get("chirality_source", "")
        target_reaction = parameters.get("target_reaction", "")

        if not metal:
            return json.dumps({"error": "Missing 'metal'"}, ensure_ascii=False)
        if not isinstance(ligands, list) or not ligands:
            return json.dumps({"error": "Missing or empty 'ligands' list"}, ensure_ascii=False)

        # Validate each ligand SMILES and gather facts
        ligand_facts = []
        valid_smiles = True
        for i, lig in enumerate(ligands):
            smi = lig.get("smiles", "")
            denticity = lig.get("denticity", 1)
            count = lig.get("count", 1)
            donor_atoms = lig.get("donor_atoms", [])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                ligand_facts.append({
                    "index": i, "smiles": smi, "valid": False,
                    "error": "invalid SMILES",
                })
                valid_smiles = False
                continue
            ligand_facts.append({
                "index": i,
                "smiles": smi,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "valid": True,
                "denticity": denticity,
                "count": count,
                "donor_atoms": donor_atoms,
                "is_labile": self._is_labile(mol),
            })

        # Computed aggregate facts (the tool's job — pure calculation)
        total_cn = sum(
            lf.get("denticity", 1) * lf.get("count", 1)
            for lf in ligand_facts if lf.get("valid")
        )
        labile_sites = sum(
            lf.get("denticity", 1) * lf.get("count", 1)
            for lf in ligand_facts if lf.get("valid") and lf.get("is_labile")
        )
        bound_sites = total_cn - labile_sites

        d_electrons = None
        electron_count_18e = None
        os_plausible = None
        if metal in METAL_GROUPS and oxidation_state is not None:
            group = METAL_GROUPS[metal]
            d_electrons = group - oxidation_state
            # 18-electron count: d electrons + 2 per donor atom (covalent approx)
            electron_count_18e = d_electrons + 2 * total_cn
            common_os = COMMON_OXIDATION_STATES.get(metal)
            if common_os is not None:
                os_plausible = oxidation_state in common_os

        expected_cn = GEOMETRY_CN.get(geometry)
        cn_matches_geometry = None
        if expected_cn is not None:
            cn_matches_geometry = (total_cn == expected_cn)

        # Symmetry candidates: any ligand appearing >=2 times enables Cn
        symmetry_candidates = []
        for lf in ligand_facts:
            if lf.get("valid") and lf.get("count", 1) >= 2:
                symmetry_candidates.append(f"C{lf['count']} (from {lf.get('formula','ligand')} ×{lf['count']})")
        symmetry_candidates = sorted(set(symmetry_candidates))

        return json.dumps({
            "input_summary": {
                "metal": metal,
                "oxidation_state": oxidation_state,
                "geometry": geometry,
                "counterion": counterion,
                "chirality_source": chirality_source,
                "target_reaction": target_reaction,
            },
            "computed_facts": {
                "all_smiles_valid": valid_smiles,
                "total_coordination_number": total_cn,
                "expected_cn_for_geometry": expected_cn,
                "cn_matches_geometry": cn_matches_geometry,
                "d_electrons": d_electrons,
                "electron_count_18e": electron_count_18e,
                "saturated_18e": (electron_count_18e >= 18) if electron_count_18e is not None else None,
                "labile_sites": labile_sites,
                "strongly_bound_sites": bound_sites,
                "oxidation_state_plausible": os_plausible,
                "symmetry_candidates": symmetry_candidates,
            },
            "ligands": ligand_facts,
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "metal": {"type": "string", "description": "Metal symbol, e.g. 'Ir', 'Pd', 'Sc', 'Fe'"},
                "oxidation_state": {"type": "integer", "description": "Metal oxidation state, e.g. 3 for Ir(III)"},
                "geometry": {
                    "type": "string",
                    "enum": list(GEOMETRY_CN.keys()),
                    "description": "Coordination geometry",
                },
                "ligands": {
                    "type": "array",
                    "description": "List of ligand descriptors",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string", "description": "Ligand SMILES (organic, standalone — NOT the whole complex)"},
                            "denticity": {"type": "integer", "description": "Number of donor atoms this ligand binds through (1=monodentate, 2=bidentate...)"},
                            "donor_atoms": {
                                "type": "array", "items": {"type": "string"},
                                "description": "Donor atom symbols, e.g. ['C','N'] for a cyclometalated C^N ligand",
                            },
                            "count": {"type": "integer", "description": "How many copies of this ligand"},
                        },
                        "required": ["smiles", "denticity", "count"],
                    },
                },
                "counterion": {"type": "string", "description": "Counterion, e.g. 'PF6', 'OTf', 'BF4', 'Cl' (optional)"},
                "chirality_source": {"type": "string", "description": "Where chirality originates: 'metal Δ/Λ', 'ligand point chirality', 'axial', etc."},
                "target_reaction": {"type": "string", "description": "Intended catalytic reaction (for context only)"},
            },
            "required": ["metal", "oxidation_state", "geometry", "ligands"],
        }

    @staticmethod
    def _is_labile(mol) -> bool:
        """Hint: does this ligand match a known labile motif? (Reported, not judged.)"""
        for s in LABILE_SMARTS:
            pat = Chem.MolFromSmarts(s)
            if pat and mol.HasSubstructMatch(pat):
                return True
        return False

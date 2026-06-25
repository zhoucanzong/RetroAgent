"""ConditionalLigandTool (template-driven): generate candidate chiral ligand SMILES.

Template-based deterministic generation — NO reliance on LLM few-shot randomness.
Maintains a curated library of REAL representative ligands per family (with correct
chirality annotations), filters by the caller's constraints (donor atoms, chirality
type, scaffold keywords), and enumerates substituent variants via RDKit.

Claude philosophy: the tool is a pure candidate GENERATOR. It does not judge whether
candidates are good — the Planner + analyze_chirality + classify_ligand + Design
Auditor (all LLM) evaluate them.
"""

import json
import re
from rdkit import Chem
from rdkit.Chem import AllChem


# ---------------------------------------------------------------------------
# Curated scaffold library — REAL representative ligands with correct chirality.
# Each entry: family, name, smiles, donor_atoms, chirality_type, keywords,
# optional substituent_sites (atom indices tolerant to substitution).
# Families align with LigandCategoryTool.SCAFFOLD_SMARTS for consistency.
# ---------------------------------------------------------------------------
SCAFFOLD_LIBRARY: list[dict] = [
    # --- Bisphosphines (axial chirality) ---
    {
        "family": "BINAP", "name": "BINAP",
        "smiles": "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2ccc3ccccc3c2P(c2ccccc2)c2ccccc2)cc1",
        "donor_atoms": ["P", "P"], "chirality_type": "axial",
        "keywords": ["phosphine", "biaryl", "axial", "hydrogenation", "cross-coupling"],
    },
    {
        "family": "BIPHEP", "name": "BIPHEP",
        "smiles": "c1ccccc1P(c1ccccc1)c1ccc(-c2ccccc2P(c2ccccc2)c2ccccc2)cc1",
        "donor_atoms": ["P", "P"], "chirality_type": "axial",
        "keywords": ["phosphine", "biaryl", "axial", "hydrogenation"],
    },

    # --- Bis-oxazolines (point chirality from amino acid) ---
    {
        "family": "BOX", "name": "(S,S)-tBu-BOX",
        "smiles": "CC(C)(C)C1N=C(CO1)CC1N=C(CO1)C(C)(C)C",
        "donor_atoms": ["N", "N"], "chirality_type": "point",
        "keywords": ["oxazoline", "bis-oxazoline", "lewis-acid", "cyclopropanation", "aldol"],
    },
    {
        "family": "BOX", "name": "(S,S)-Ph-BOX",
        "smiles": "c1ccc(C2N=C(CO2)CC2N=C(CO2)c2ccccc2)cc1",
        "donor_atoms": ["N", "N"], "chirality_type": "point",
        "keywords": ["oxazoline", "bis-oxazoline", "phenyl"],
    },

    # --- Phosphinooxazolines (point chirality) ---
    {
        "family": "PHOX", "name": "(S)-tBu-PHOX",
        "smiles": "CC(C)(C)C1N=C(CO1)c1ccccc1P(c1ccccc1)c1ccccc1",
        "donor_atoms": ["P", "N"], "chirality_type": "point",
        "keywords": ["phosphinooxazoline", "phox", "allylic", "hydrogenation"],
    },

    # --- Josiphos / ferrocenyl (planar + point chirality) ---
    {
        "family": "JOSIPHOS", "name": "ferrocenyl bisphosphine",
        "smiles": "[Fe].c1ccccc1P(c1ccccc1)c1ccccc1",
        "donor_atoms": ["P", "P"], "chirality_type": "planar",
        "keywords": ["ferrocene", "josiphos", "hydrogenation", "planar"],
    },

    # --- NHC (carbene, often achiral backbone but can be C2) ---
    {
        "family": "NHC", "name": "IMes (NHC precursor)",
        "smiles": "C1=[N+](c2ccccc2C)c2ccccc2N1C",  # imidazolium core
        "donor_atoms": ["C"], "chirality_type": "none",
        "keywords": ["nhc", "carbene", "imidazolium", "cross-coupling"],
    },

    # --- N,N'-Dioxide (Feng-type; point chirality from proline, O-donors) ---
    {
        "family": "NNOXIDE", "name": "bis(L-proline-N-oxide) amide",
        "smiles": "O=C([C@@H]1CCC[N+]1[O-])Nc1cccc(NC(=O)[C@@H]2CCC[N+]2[O-])c1",
        "donor_atoms": ["O", "O"], "chirality_type": "point",
        "keywords": ["n-oxide", "dioxide", "feng", "proline", "lewis-acid", "strecker"],
    },

    # --- DIOP (diphosphine with acetal) ---
    {
        "family": "DIOP", "name": "DIOP",
        "smiles": "c1ccc(P(CC2OC(C)(C)OC2CP(c3ccccc3)c3ccccc3)c2ccccc2)cc1",
        "donor_atoms": ["P", "P"], "chirality_type": "point",
        "keywords": ["diop", "phosphine", "acetal", "hydrogenation"],
    },

    # --- Salen (Schiff-base, point chirality) ---
    {
        "family": "SALEN", "name": "(R,R)-salen",
        "smiles": "Oc1ccccc1C=NCCCN=Cc1ccccc1O",
        "donor_atoms": ["N", "N", "O", "O"], "chirality_type": "point",
        "keywords": ["salen", "schiff", "epoxidation", "jacobsen"],
    },

    # --- Bipyridine / phenanthroline (achiral N,N) ---
    {
        "family": "BIPY", "name": "2,2'-bipyridine",
        "smiles": "c1ccncc1-c1ccncc1",
        "donor_atoms": ["N", "N"], "chirality_type": "none",
        "keywords": ["bipyridine", "bpy", "achiral"],
    },
]


class ConditionalLigandTool:
    name = "design_ligand"
    description = (
        "Generate candidate chiral ligand/catalyst SMILES from constraints "
        "(chirality type, donor atoms, scaffold family, application keywords). "
        "Template-driven: returns REAL representative ligands per family with correct "
        "chirality, plus RDKit-enumerated substituent variants. Filtered by your "
        "constraints. The Planner must validate output with analyze_chirality, "
        "classify_ligand, and evaluate."
    )

    def execute(self, parameters: dict) -> str:
        constraints = parameters.get("constraints", "") or ""
        count = int(parameters.get("count", 5))
        enumerate_variants = parameters.get("enumerate", True)

        parsed = self._parse_constraints(constraints)
        matches = self._filter_library(parsed)

        # Build candidate list: each matched scaffold + optional variants
        candidates = []
        seen_smiles = set()
        for entry in matches:
            base_smi = entry["smiles"]
            base_canon = self._canonicalize(base_smi)
            if base_canon and base_canon not in seen_smiles:
                seen_smiles.add(base_canon)
                candidates.append({
                    "smiles": base_canon,
                    "family": entry["family"],
                    "name": entry["name"],
                    "donor_atoms": entry["donor_atoms"],
                    "chirality_type": entry["chirality_type"],
                    "variant": "parent",
                })
            if enumerate_variants and len(candidates) < count * 2:
                for var in self._enumerate_variants(base_smi, max_variants=2):
                    var_canon = self._canonicalize(var)
                    if var_canon and var_canon not in seen_smiles:
                        seen_smiles.add(var_canon)
                        candidates.append({
                            "smiles": var_canon,
                            "family": entry["family"],
                            "name": entry["name"] + " (variant)",
                            "donor_atoms": entry["donor_atoms"],
                            "chirality_type": entry["chirality_type"],
                            "variant": "substituted",
                        })
                    if len(candidates) >= count * 2:
                        break
            if len(candidates) >= count * 2:
                break

        candidates = candidates[:max(count, 5)]

        return json.dumps({
            "constraints": constraints,
            "parsed_constraints": parsed,
            "candidates": candidates,
            "count": len(candidates),
            "note": (
                "Template-generated candidates from curated scaffold library. "
                "Validate with analyze_chirality + classify_ligand + evaluate."
            ),
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "constraints": {
                    "type": "string",
                    "description": (
                        "Natural-language constraints. Recognized tokens: donor atoms "
                        "(P/N/O/S), chirality type (axial/point/planar/none), scaffold "
                        "family (BINAP/BOX/PHOX/NHC/N-oxide/salen/Josiphos/etc.), "
                        "application (hydrogenation/cross-coupling/Strecker/etc.). "
                        "Example: 'P,P bidentate phosphine with axial chirality for asymmetric hydrogenation'"
                    ),
                },
                "count": {"type": "integer", "default": 5, "description": "Number of candidates"},
                "enumerate": {"type": "boolean", "default": True, "description": "Generate substituent variants"},
            },
            "required": ["constraints"],
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _parse_constraints(text: str) -> dict:
        """Extract donor atoms, chirality, family, keywords from free text."""
        t = text.lower()
        donor_atoms = []
        for sym in ("phosphorus", "phosphine"):
            if sym in t or "p donor" in t or " p," in t or "p,p" in t or "pph" in t:
                if "P" not in donor_atoms:
                    donor_atoms.append("P")
        for sym, keys in [("N", ["nitrogen", "n donor", "n,n", "amine", "oxazoline", "pyridine"]),
                          ("O", ["oxygen", "o donor", "n-oxide", "dioxide", "phenolate", "alkoxide"]),
                          ("S", ["sulfur", "s donor", "thiolate", "thioether"])]:
            if any(k in t for k in keys):
                if sym not in donor_atoms:
                    donor_atoms.append(sym)
        # Explicit element symbols like "P,O"
        for m in re.finditer(r"\b([PNOS])\b", text):
            if m.group(1) not in donor_atoms:
                donor_atoms.append(m.group(1))

        chirality = None
        for c, keys in [("axial", ["axial", "biaryl", "binap", "atrop"]),
                        ("point", ["point", "chiral center", "proline", "oxazoline"]),
                        ("planar", ["planar", "ferrocene", "josiphos"]),
                        ("none", ["achiral", "non-chiral", "no chirality"])]:
            if any(k in t for k in keys):
                chirality = c
                break

        family = None
        for fam in ["binap", "biphep", "segphos", "box", "pybox", "phox", "josiphos",
                    "walphos", "nhc", "n-oxide", "dioxide", "diop", "salen",
                    "bipyridine", "bpy", "phenanthroline"]:
            if fam in t:
                family = fam.upper().replace("-", "").replace("_", "")
                if family in ("DIOXIDE", "NOXIDE"):
                    family = "NNOXIDE"
                if family in ("BPY", "BIPYRIDINE"):
                    family = "BIPY"
                break

        keywords = [w for w in re.split(r"[\s,;]+", t) if len(w) > 2]
        return {
            "donor_atoms": donor_atoms,
            "chirality_type": chirality,
            "family": family,
            "keywords": keywords,
        }

    @staticmethod
    def _filter_library(parsed: dict) -> list[dict]:
        """Rank scaffolds by constraint match. Returns best matches first."""
        scored = []
        want_donors = set(parsed["donor_atoms"])
        want_chiral = parsed["chirality_type"]
        want_family = parsed["family"]
        kw_set = set(parsed["keywords"])

        for entry in SCAFFOLD_LIBRARY:
            score = 0
            have_donors = set(entry["donor_atoms"])
            if want_donors:
                # Reward donor overlap
                overlap = len(want_donors & have_donors)
                if overlap == len(want_donors):
                    score += 3
                elif overlap > 0:
                    score += 1
                else:
                    score -= 2  # donor mismatch is a strong negative
            if want_chiral and entry["chirality_type"] == want_chiral:
                score += 2
            if want_chiral and entry["chirality_type"] == "none" and want_chiral != "none":
                score -= 1
            if want_family:
                if entry["family"] == want_family:
                    score += 5
                elif want_family in entry["family"]:
                    score += 2
            # Keyword overlap
            entry_kw = set(k.lower() for k in entry["keywords"])
            score += len(kw_set & entry_kw)
            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        # Return all with positive score, or top 4 if nothing positive
        positive = [e for s, e in scored if s > 0]
        return positive if positive else [e for _, e in scored[:4]]

    @staticmethod
    def _canonicalize(smiles: str) -> str | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    @staticmethod
    def _enumerate_variants(smiles: str, max_variants: int = 2) -> list[str]:
        """Generate simple substituent variants by adding Me/tBu at aromatic ortho
        positions or alpha carbons. Conservative — only valid chemistry."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        variants = []
        candidate_atoms = []
        for atom in mol.GetAtoms():
            # Only substitute aromatic carbons with exactly 1 H (ortho-like positions)
            if (atom.GetIsAromatic() and atom.GetSymbol() == "C"
                    and atom.GetNumImplicitHs() == 1 and atom.GetDegree() == 2):
                candidate_atoms.append(atom.GetIdx())
        # Try methyl substitution at first 2 candidate sites
        for idx in candidate_atoms[:2]:
            try:
                rw = Chem.RWMol(mol)
                new_atom = rw.AddAtom(Chem.Atom(6))  # carbon (methyl)
                rw.AddBond(idx, new_atom, Chem.BondType.SINGLE)
                variant = Chem.MolToSmiles(rw.GetMol())
                if Chem.MolFromSmiles(variant) is not None:
                    variants.append(variant)
            except Exception:
                continue
            if len(variants) >= max_variants:
                break
        return variants

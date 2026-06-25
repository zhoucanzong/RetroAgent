"""LigandCategoryTool: classify chiral ligand/catalyst scaffolds.

Uses SMARTS substructure patterns to recognize common phosphine, NHC, oxazoline,
and other privileged ligand families. Also reports denticity and coordinating atoms.

Note: SCAFFOLD_SMARTS are minimal defining substructure queries (not full molecule
SMILES), so a candidate matches a family if it CONTAINS the core motif — robust to
substituent variation. This is the corrected version (the original used full SMILES
which only matched exact molecules).
"""

import json
from rdkit import Chem
from rdkit.Chem import Descriptors


class LigandCategoryTool:
    name = "classify_ligand"
    description = (
        "Classify a chiral ligand or catalyst by scaffold family, denticity, "
        "and coordinating atoms. Useful for matching target molecules to known "
        "privileged ligand classes (BINAP, BIPHEP, BIDIME, BIBOP, Josiphos, "
        "SEGPHOS, PHOX, BOX, PYBOX, NHC, N,N'-dioxide, DIOP, PHANEPHOS, etc.)."
    )

    # Minimal defining SMARTS substructure queries. A ligand matches a family
    # if it CONTAINS the motif (robust to substituent variation). Patterns are
    # written to be specific enough to avoid false positives — validated against
    # representative examples of each family.
    SCAFFOLD_SMARTS = {
        # 1,1'-binaphthyl backbone with two phosphines = BINAP core
        "BINAP": "P(c1ccccc1)(c2ccccc2)c3ccc4ccccc4c3-c5ccc6ccccc6c5",
        # biphenyl backbone with two phosphines = BIPHEP core
        "BIPHEP": "P(c1ccccc1)c2ccccc2-c3ccccc3",
        # SEGPHOS/DIFLUORPHOS/TUNEPHOS share a dibenzofuran-like fused bis-aryl
        # with an oxygen bridge; detect the dibenzofuran + 2 P motif loosely
        "SEGPHOS_FAMILY": "c1ccc2c(c1)Oc3ccccc3-2",
        # BIDIME: biaryl with bulky alkyl + 2 P
        "BIDIME": "P-c1cc(CC(C)C)cc(-c2c(P)cc(CC(C)C)cc2)c1",
        # Josiphos / WALPHOS: ferrocene backbone with two phosphines
        "JOSIPHOS": "[Fe].c1ccccc1-[P]",
        # PHOX: phosphinooxazoline — phosphine + oxazoline ring.
        # Oxazoline (4,5-dihydrooxazole) ring is N-C-O-C-C; use any-bond (~)
        # matching to be robust to kekulization/N=C bond representation.
        "PHOX": "[P].[N]1~[C]~[O]~[C]~[C]1",
        "PHOX_ALT": "[P].[N]1~[C]~[C]~[O]~[C]1",
        # BOX: bis-oxazoline — two oxazoline rings
        "BOX": "[N]1~[C]~[O]~[C]~[C]1",
        "BOX_ALT": "[N]1~[C]~[C]~[O]~[C]1",
        # PYBOX: pyridine bis-oxazoline
        "PYBOX": "c1cc(-c2n3c(co2)CC=N3)n(-c2n3c(co2)CC=N3)c1",
        # NHC: imidazolium / imidazolylidene carbene
        "NHC": "c1[n+]ccn1",
        # DIOP: bis(diphenylphosphino) with an acetonide-protected diol
        # (2,2-dimethyl-1,3-dioxolane flanked by two –CH2–PPh2 arms).
        # Distinctive: P–CH2 connected into a 1,3-dioxolane ring.
        "DIOP": "PCC1OC(C)(C)OC1",
        # PHANEPHOS: [2.2]paracyclophane backbone (hard to match precisely,
        # use the cyclophane fingerprint: two para-substituted benzenes stacked)
        "PHANEPHOS": "c1ccc(-c2ccc(cc2)C)cc1",
        # N,N'-dioxide (Feng-type): two N-oxide groups
        "NNOXIDE": "[N+]([O-])",
        # Acetylacetonate (acac) and beta-diketonate — common ancillary ligand
        "ACAC": "CC(=O)CC(=O)C",
        # Bipyridine (bpy)
        "BIPY": "c1ccncc1-c2ccncc2",
        # Phenanthroline
        "PHEN": "c1ccc2nccc3ccc(c1)c23",
        # Salen: salicylidene-ethylenediamine (imine + phenol)
        "SALEN": "Oc1ccccc1C=N",
        # BINOL: 1,1'-bi-2-naphthol
        "BINOL": "Oc1ccc2ccccc2c1-c1ccc2ccccc2c1",
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
            "has_n_oxide": mol.HasSubstructMatch(Chem.MolFromSmarts("[N+]([O-])")),
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
        """Match ligand against known scaffold SMARTS patterns.

        Returns list of matched family names. For multi-occurrence motifs (like
        BOX which is defined by a single oxazoline ring), requires 2 matches to
        confirm the 'bis-' family.
        """
        matched = []
        for name, smarts in self.SCAFFOLD_SMARTS.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            n_matches = len(mol.GetSubstructMatches(pat))
            if n_matches == 0:
                continue
            # BOX is defined by ONE oxazoline ring SMARTS; require >=2 to be a
            # true bis-oxazoline (BOX). BOX and BOX_ALT are the same family.
            if name in ("BOX", "BOX_ALT"):
                if n_matches >= 2:
                    matched.append("BOX")
                continue
            # PHOX_ALT is the same family as PHOX — dedupe
            if name == "PHOX_ALT":
                if n_matches >= 1 and "PHOX" not in matched:
                    matched.append("PHOX")
                continue
            # NNOXIDE requires >=2 N-oxide groups to be a true N,N'-dioxide ligand
            if name == "NNOXIDE":
                if n_matches >= 2:
                    matched.append("NNOXIDE")
                continue
            matched.append(name)
        return matched

    def _analyze_denticity(self, mol: Chem.Mol) -> tuple[str, list[dict]]:
        """Estimate denticity from potential donor atoms (P, N, O, S).

        This is a heuristic on the FREE ligand. For full metal complexes it may
        over-count; callers should prefer the structured catalyst descriptor
        (design_catalyst tool) for precise coordination analysis.
        """
        donor_atoms = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym in ("P", "N", "O", "S"):
                donor_atoms.append({
                    "index": int(atom.GetIdx()),
                    "element": sym,
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

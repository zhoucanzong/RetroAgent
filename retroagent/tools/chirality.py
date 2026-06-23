"""ChiralityTool: analyze stereochemistry of a molecule.

Detects stereocenters, chirality type (point/axial/planar/helical),
and reports R/S configuration per stereocenter.
"""

import json
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class ChiralityTool:
    name = "analyze_chirality"
    description = (
        "Analyze the stereochemistry of a molecule. "
        "Returns chirality type (Point/Axial/Planar/Helical), stereocenter count, "
        "and R/S labels for each stereocenter. Use this when designing or evaluating "
        "chiral ligands and catalysts."
    )

    def execute(self, parameters: dict) -> str:
        smiles = parameters.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}"}, ensure_ascii=False)

        # Assign stereochemistry explicitly
        Chem.AssignStereochemistryFrom3D(mol) if mol.GetNumConformers() else None
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

        stereocenters = self._get_stereocenters(mol)
        double_bond_stereo = self._get_double_bond_stereo(mol)

        result = {
            "smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
            "chirality_type": self._classify_chirality_type(mol, stereocenters, double_bond_stereo),
            "num_stereocenters": len(stereocenters),
            "stereocenters": stereocenters,
            "double_bond_stereo": double_bond_stereo,
            "has_stereochemistry": bool(stereocenters or double_bond_stereo),
            "enantiomer_smiles": self._canonical_enantiomer(mol) if stereocenters else None,
        }
        return json.dumps(result, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Input SMILES string"}
            },
            "required": ["smiles"]
        }

    @staticmethod
    def _get_stereocenters(mol: Chem.Mol) -> list[dict]:
        centers = []
        # Atom stereocenters
        chiral_at_idx = Chem.FindMolChiralCenters(mol, includeUnassigned=True, includeCIP=True, useLegacyImplementation=False)
        for idx, cip in chiral_at_idx:
            atom = mol.GetAtomWithIdx(idx)
            centers.append({
                "type": "atom",
                "index": int(idx),
                "element": atom.GetSymbol(),
                "cip_label": cip if cip else "unassigned",
                "smarts_symbol": atom.GetSmarts(),
            })
        return centers

    @staticmethod
    def _get_double_bond_stereo(mol: Chem.Mol) -> list[dict]:
        bonds = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                stereo = bond.GetStereo()
                if stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS):
                    begin = bond.GetBeginAtomIdx()
                    end = bond.GetEndAtomIdx()
                    bonds.append({
                        "type": "double_bond",
                        "begin_atom": int(begin),
                        "end_atom": int(end),
                        "stereo": str(stereo),
                    })
        return bonds

    @staticmethod
    def _classify_chirality_type(mol: Chem.Mol, stereocenters: list, double_bond_stereo: list) -> str:
        if not stereocenters:
            return "Achiral"

        atom_stereos = [c for c in stereocenters if c["type"] == "atom"]
        if not atom_stereos:
            if double_bond_stereo:
                return "Planar"  # E/Z as planar chirality proxy
            return "Achiral"

        # Heuristic classification
        # Axial: multiple aromatic rings with steric hindrance (BINAP-like)
        # Planar: all-carbon fused rings with out-of-plane substituents
        # Helical: helicene-like extended fused systems
        # Point: single atom stereocenter
        ri = mol.GetRingInfo()
        ring_count = ri.NumRings()
        aromatic_rings = sum(1 for ring in ri.AtomRings() if all(mol.GetAtomWithIdx(a).GetIsAromatic() for a in ring))

        if aromatic_rings >= 2 and ring_count >= 4:
            return "Axial"
        if aromatic_rings >= 1 and ring_count >= 3:
            return "Planar"
        if ring_count >= 4 and len(atom_stereos) >= 1:
            return "Helical"
        return "Point"

    @staticmethod
    def _canonical_enantiomer(mol: Chem.Mol) -> str | None:
        try:
            inv = Chem.Mol(mol)
            for atom in inv.GetAtoms():
                chiral_tag = atom.GetChiralTag()
                if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                    atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            Chem.AssignStereochemistry(inv, force=True, cleanIt=True)
            return Chem.MolToSmiles(inv, canonical=True)
        except Exception:
            return None

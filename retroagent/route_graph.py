"""AND-OR route graph primitives for retrosynthesis planning.

Molecule nodes are OR nodes: a molecule may be prepared by any one of several
reaction nodes. Reaction nodes are AND nodes: all precursor molecules must be
available or recursively solved.
"""

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from retroagent.schemas import MoleculeNode, ReactionRecord, normalize_reaction_record


@dataclass
class RouteGraph:
    target_id: str = ""
    molecules: dict[str, MoleculeNode] = field(default_factory=dict)
    molecule_by_smiles: dict[str, str] = field(default_factory=dict)
    reactions: dict[str, ReactionRecord] = field(default_factory=dict)
    product_to_reactions: dict[str, list[str]] = field(default_factory=dict)
    reaction_to_precursors: dict[str, list[str]] = field(default_factory=dict)

    def add_target(self, smiles: str) -> str:
        self.target_id = self.add_molecule(smiles, role="target")
        return self.target_id

    def add_molecule(self, smiles: str, *, role: str = "intermediate",
                     stock_status: str = "unknown") -> str:
        canonical = canonicalize_smiles(smiles)
        if canonical in self.molecule_by_smiles:
            mol_id = self.molecule_by_smiles[canonical]
            node = self.molecules[mol_id]
            if node.role != "target" and role == "target":
                node.role = "target"
            if node.stock_status == "unknown" and stock_status != "unknown":
                node.stock_status = stock_status
            return mol_id

        mol_id = f"mol_{len(self.molecules) + 1}"
        self.molecule_by_smiles[canonical] = mol_id
        self.molecules[mol_id] = MoleculeNode(
            id=mol_id,
            smiles=smiles,
            canonical_smiles=canonical,
            role=role,
            stock_status=stock_status,
        )
        return mol_id

    def add_reaction(self, reaction: dict[str, Any], *, target: str = "") -> str:
        normalized = normalize_reaction_record(
            reaction,
            target=target,
            index=len(self.reactions) + 1,
        )
        product_id = self.add_molecule(normalized["product_smiles"])
        precursor_ids = [
            self.add_molecule(smiles, role="precursor")
            for smiles in normalized["precursor_smiles_list"]
        ]

        reaction_id = normalized["id"] or f"rxn_{len(self.reactions) + 1}"
        if reaction_id in self.reactions:
            suffix = len(self.reactions) + 1
            reaction_id = f"{reaction_id}_{suffix}"
            normalized["id"] = reaction_id

        self.reactions[reaction_id] = ReactionRecord(
            id=reaction_id,
            product_smiles=normalized["product_smiles"],
            precursor_smiles_list=list(normalized["precursor_smiles_list"]),
            template_index=normalized.get("template_index"),
            classification=normalized.get("classification", ""),
            library_occurrence=normalized.get("library_occurrence"),
            evidence=normalized.get("evidence", []),
            risk_flags=normalized.get("risk_flags", []),
            metadata=normalized.get("metadata", {}),
        )
        self.product_to_reactions.setdefault(product_id, []).append(reaction_id)
        self.reaction_to_precursors[reaction_id] = precursor_ids
        return reaction_id

    def mark_stock(self, smiles: str, source: str = "stock") -> None:
        mol_id = self.add_molecule(smiles, stock_status=source)
        self.molecules[mol_id].stock_status = source

    def unsolved_molecule_ids(self) -> list[str]:
        unsolved = []
        for mol_id, node in self.molecules.items():
            has_route = bool(self.product_to_reactions.get(mol_id))
            in_stock = node.stock_status not in ("unknown", "", "not_in_stock")
            if not has_route and not in_stock:
                unsolved.append(mol_id)
        return unsolved

    def stats(self) -> dict[str, int | str]:
        return {
            "target_id": self.target_id,
            "molecule_count": len(self.molecules),
            "reaction_count": len(self.reactions),
            "unsolved_molecule_count": len(self.unsolved_molecule_ids()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "molecules": {key: value.to_dict() for key, value in self.molecules.items()},
            "reactions": {key: value.to_dict() for key, value in self.reactions.items()},
            "product_to_reactions": self.product_to_reactions,
            "reaction_to_precursors": self.reaction_to_precursors,
            "stats": self.stats(),
        }


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)

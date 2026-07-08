"""AND-OR route graph primitives for retrosynthesis planning.

Molecule nodes are OR nodes: a molecule may be prepared by any one of several
reaction nodes. Reaction nodes are AND nodes: all precursor molecules must be
available or recursively solved.
"""

from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

from retroagent.schemas import MoleculeNode, ReactionRecord, Route, normalize_reaction_record


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

    def enumerate_routes(
        self,
        *,
        target_id: str = "",
        max_depth: int = 4,
        max_routes: int = 20,
    ) -> list[Route]:
        """Enumerate candidate AND-OR routes from a target molecule.

        A route is "solved" when every leaf precursor is marked stock. Unsolved
        leaves are still returned so discovery mode can score partial routes.
        """
        root_id = target_id or self.target_id
        if not root_id or root_id not in self.molecules:
            return []
        route_dicts = self._expand_molecule(
            root_id,
            depth=max_depth,
            seen_molecules=set(),
            max_routes=max_routes,
        )
        routes = []
        for index, route_data in enumerate(route_dicts[:max_routes], start=1):
            metadata = _route_metadata(route_data, self)
            routes.append(Route(
                id=f"route_{index}",
                target_smiles=self.molecules[root_id].smiles,
                reaction_ids=list(route_data["reaction_ids"]),
                score=score_route_metadata(metadata),
                metadata=metadata,
            ))
        return sorted(routes, key=lambda route: route.score or 0.0, reverse=True)

    def scored_routes(
        self,
        *,
        max_depth: int = 4,
        max_routes: int = 20,
    ) -> list[dict[str, Any]]:
        return [route.to_dict() for route in self.enumerate_routes(max_depth=max_depth, max_routes=max_routes)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "molecules": {key: value.to_dict() for key, value in self.molecules.items()},
            "reactions": {key: value.to_dict() for key, value in self.reactions.items()},
            "product_to_reactions": self.product_to_reactions,
            "reaction_to_precursors": self.reaction_to_precursors,
            "stats": self.stats(),
            "candidate_routes": self.scored_routes(max_routes=10),
        }

    def _expand_molecule(
        self,
        mol_id: str,
        *,
        depth: int,
        seen_molecules: set[str],
        max_routes: int,
    ) -> list[dict[str, Any]]:
        node = self.molecules[mol_id]
        if _is_stock(node):
            return [_leaf_route(stock_ids={mol_id})]
        if depth <= 0 or mol_id in seen_molecules:
            return [_leaf_route(unresolved_ids={mol_id})]

        reaction_ids = self.product_to_reactions.get(mol_id, [])
        if not reaction_ids:
            return [_leaf_route(unresolved_ids={mol_id})]

        routes = []
        next_seen = set(seen_molecules)
        next_seen.add(mol_id)
        for reaction_id in reaction_ids:
            precursor_ids = self.reaction_to_precursors.get(reaction_id, [])
            precursor_options = [
                self._expand_molecule(
                    precursor_id,
                    depth=depth - 1,
                    seen_molecules=next_seen,
                    max_routes=max_routes,
                )
                for precursor_id in precursor_ids
            ]
            for combo in _limited_product(precursor_options, max_routes=max_routes):
                merged = _merge_route_parts(combo)
                merged["reaction_ids"].insert(0, reaction_id)
                routes.append(merged)
                if len(routes) >= max_routes:
                    break
            if len(routes) >= max_routes:
                break
        return routes or [_leaf_route(unresolved_ids={mol_id})]


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def score_route_metadata(metadata: dict[str, Any]) -> float:
    score = 1.0
    if not metadata["solved"]:
        score -= min(0.5, 0.15 * metadata["unresolved_count"])
    score -= min(0.3, 0.04 * max(metadata["step_count"] - 1, 0))
    if metadata["convergent_step_count"]:
        score += 0.08
    if metadata["reaction_class_count"] >= 2:
        score += 0.07
    if metadata["protection_like_reaction_count"]:
        score -= min(0.2, 0.06 * metadata["protection_like_reaction_count"])
    return round(max(0.0, min(1.0, score)), 3)


def _route_metadata(route_data: dict[str, Any], graph: RouteGraph) -> dict[str, Any]:
    reaction_ids = list(route_data["reaction_ids"])
    reaction_classes = sorted({
        graph.reactions[reaction_id].classification
        for reaction_id in reaction_ids
        if graph.reactions[reaction_id].classification
    })
    leaf_ids = sorted(route_data["stock_ids"] | route_data["unresolved_ids"])
    unresolved_ids = sorted(route_data["unresolved_ids"])
    stock_ids = sorted(route_data["stock_ids"])
    convergent_steps = [
        reaction_id for reaction_id in reaction_ids
        if len(graph.reaction_to_precursors.get(reaction_id, [])) > 1
    ]
    protection_like = [
        reaction_id for reaction_id in reaction_ids
        if _is_protection_like(graph.reactions[reaction_id].classification)
    ]
    return {
        "solved": not unresolved_ids,
        "step_count": len(reaction_ids),
        "reaction_count": len(reaction_ids),
        "reaction_classes": reaction_classes,
        "reaction_class_count": len(reaction_classes),
        "leaf_precursor_ids": leaf_ids,
        "leaf_precursor_smiles": [_molecule_smiles(graph, mol_id) for mol_id in leaf_ids],
        "stock_precursor_ids": stock_ids,
        "stock_precursor_smiles": [_molecule_smiles(graph, mol_id) for mol_id in stock_ids],
        "unresolved_molecule_ids": unresolved_ids,
        "unresolved_smiles": [_molecule_smiles(graph, mol_id) for mol_id in unresolved_ids],
        "unresolved_count": len(unresolved_ids),
        "convergent_step_count": len(convergent_steps),
        "convergent_reaction_ids": convergent_steps,
        "protection_like_reaction_count": len(protection_like),
        "protection_like_reaction_ids": protection_like,
    }


def _molecule_smiles(graph: RouteGraph, mol_id: str) -> str:
    node = graph.molecules[mol_id]
    return node.canonical_smiles or node.smiles


def _is_stock(node: MoleculeNode) -> bool:
    return node.stock_status not in ("unknown", "", "not_in_stock")


def _leaf_route(
    *,
    stock_ids: set[str] | None = None,
    unresolved_ids: set[str] | None = None,
) -> dict[str, Any]:
    return {
        "reaction_ids": [],
        "stock_ids": set(stock_ids or set()),
        "unresolved_ids": set(unresolved_ids or set()),
    }


def _merge_route_parts(parts: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    merged = _leaf_route()
    for part in parts:
        merged["reaction_ids"].extend(part["reaction_ids"])
        merged["stock_ids"].update(part["stock_ids"])
        merged["unresolved_ids"].update(part["unresolved_ids"])
    return merged


def _limited_product(option_groups: list[list[dict[str, Any]]], *, max_routes: int) -> list[tuple[dict[str, Any], ...]]:
    combos: list[tuple[dict[str, Any], ...]] = [()]
    for options in option_groups:
        next_combos = []
        for combo in combos:
            for option in options:
                next_combos.append((*combo, option))
                if len(next_combos) >= max_routes:
                    break
            if len(next_combos) >= max_routes:
                break
        combos = next_combos
        if not combos:
            break
    return combos[:max_routes]


def _is_protection_like(classification: str) -> bool:
    text = classification.lower()
    return any(pattern in text for pattern in ("protection", "deprotection", "protect", "deprotect", "boc", "cbz", "silyl"))

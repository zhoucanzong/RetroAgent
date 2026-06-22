"""SharedBlackboard — pure state container, no decision logic."""

import time
from dataclasses import dataclass, field


@dataclass
class SharedBlackboard:
    """Maintains the complete state of one retrosynthetic planning task.

    Responsibilities:
    1. Hold all structured state (target, search tree, routes, scores, etc.)
    2. Serialize state as Jinja2 template variables for the Planner prompt
    3. Accept tool outputs via update() — pure data merge, no decisions
    """

    target_smiles: str = ""
    search_tree: dict | None = None
    routes: list[dict] = field(default_factory=list)
    stock_hits: dict[str, str] = field(default_factory=dict)  # inchi_key -> source
    scores: dict[str, float] = field(default_factory=dict)  # route_id -> total_score
    literature_precedents: list[dict] = field(default_factory=list)
    disconnection_results: list[dict] = field(default_factory=list)
    proposal_results: list[dict] = field(default_factory=list)
    iteration_count: int = 0
    start_time: float = 0.0
    num_atoms: int = 0
    rings: int = 0
    functional_groups: list[str] = field(default_factory=list)

    def initialize(self, smiles: str) -> None:
        self.target_smiles = smiles
        self.start_time = time.time()
        self.iteration_count = 0
        self.routes.clear()
        self.scores.clear()
        self.stock_hits.clear()
        self.literature_precedents.clear()
        self.disconnection_results.clear()
        self.proposal_results.clear()
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                self.num_atoms = mol.GetNumAtoms()
                self.rings = mol.GetRingInfo().NumRings()
        except Exception:
            pass

    def to_template_vars(self) -> dict:
        elapsed = time.time() - self.start_time if self.start_time else 0
        return {
            "blackboard": {
                "target": self.target_smiles,
                "num_atoms": self.num_atoms,
                "rings": self.rings,
                "route_count": len(self.routes),
                "best_score": max(self.scores.values()) if self.scores else None,
                "stock_hits_count": len(self.stock_hits),
                "literature_count": len(self.literature_precedents),
                "disconnections": len(self.disconnection_results),
                "proposals": len(self.proposal_results),
                "iteration": self.iteration_count,
                "elapsed": f"{elapsed:.1f}s",
            }
        }

    def update(self, tool_name: str, result: dict) -> None:
        """Merge tool output into the blackboard. Pure data operation."""
        self.iteration_count += 1
        match tool_name:
            case "disconnect":
                self.disconnection_results = result.get("bonds", [])
            case "propose":
                self.proposal_results = result.get("reactions", [])
            case "evaluate":
                self.scores |= result.get("scores", {})
            case "search_literature":
                self.literature_precedents.extend(result.get("known_routes", []))
            case "check_stock":
                for smi, src in result.get("in_stock", {}).items():
                    self.stock_hits[smi] = src

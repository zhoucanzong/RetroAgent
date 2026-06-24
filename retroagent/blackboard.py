"""SharedBlackboard — pure state container, no decision logic.

Phase 4: Branch tracking for route exploration status table.
"""

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
    mode: str = "retrosynthesis"
    design_candidates: list[dict] = field(default_factory=list)
    design_evaluations: list[dict] = field(default_factory=list)

    # Phase 4: Branch tracking for route exploration status
    # Each branch: {id, bond_name, template_index, classification, best_score,
    #               precursor_count, stock_count, status}
    # status: "exploring" | "evaluated" | "high_score" | "submitted" | "abandoned"
    branches: list[dict] = field(default_factory=list)

    def initialize(self, smiles: str, mode: str = "retrosynthesis") -> None:
        self.target_smiles = smiles
        self.mode = mode
        self.start_time = time.time()
        self.iteration_count = 0
        self.routes.clear()
        self.scores.clear()
        self.stock_hits.clear()
        self.literature_precedents.clear()
        self.disconnection_results.clear()
        self.proposal_results.clear()
        self.design_candidates.clear()
        self.design_evaluations.clear()
        self.branches.clear()
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
        # Build branch status table
        branch_table = self._build_branch_table()
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
                "mode": self.mode,
                "design_candidates": len(self.design_candidates),
                "design_evaluations": len(self.design_evaluations),
                "iteration": self.iteration_count,
                "elapsed": f"{elapsed:.1f}s",
                # Phase 4: branch tracking
                "branch_count": len(self.branches),
                "branch_table": branch_table,
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
            case "analyze_chirality":
                self.design_evaluations.append({"tool": "analyze_chirality", "result": result})
            case "classify_ligand":
                self.design_evaluations.append({"tool": "classify_ligand", "result": result})
            case "design_ligand":
                self.design_candidates.extend(result.get("candidates", []))

    def _build_branch_table(self) -> str:
        """Build a Markdown branch exploration status table."""
        if not self.branches:
            return "_No branches tracked yet._"

        lines = [
            "| # | Bond / Template | Class | Best Score | Stock | Status |",
            "|---|-----------------|-------|------------|-------|--------|",
        ]
        for i, b in enumerate(self.branches[:10], 1):
            bond = b.get("bond_name", f"T#{b.get('template_index', '?')}")
            if len(bond) > 20:
                bond = bond[:17] + "..."
            cls = b.get("classification", "?")
            if len(cls) > 8:
                cls = cls[:5] + "..."
            score = b.get("best_score")
            score_str = f"{score:.3f}" if score is not None else "—"
            stock = f"{b.get('stock_count', 0)}/{b.get('precursor_count', '?')}"
            status = b.get("status", "exploring")
            status_icon = {
                "exploring": "🔍 exploring",
                "evaluated": "📊 evaluated",
                "high_score": "⭐ high",
                "submitted": "✅ submitted",
                "abandoned": "❌ abandoned",
            }.get(status, status)
            lines.append(f"| {i} | {bond} | {cls} | {score_str} | {stock} | {status_icon} |")

        active = sum(1 for b in self.branches if b.get("status") not in ("submitted", "abandoned"))
        submitted = sum(1 for b in self.branches if b.get("status") == "submitted")
        abandoned = sum(1 for b in self.branches if b.get("status") == "abandoned")
        lines.append(
            f"\n**Summary**: {active} active | {submitted} submitted | {abandoned} abandoned"
        )
        return "\n".join(lines)

    def track_branch(self, bond_name: str = "", template_index: int = 0,
                     classification: str = "", precursor_count: int = 0,
                     status: str = "exploring") -> int:
        """Register a new exploration branch and return its ID."""
        branch_id = len(self.branches) + 1
        self.branches.append({
            "id": f"B{branch_id}",
            "bond_name": bond_name,
            "template_index": template_index,
            "classification": classification,
            "best_score": None,
            "precursor_count": precursor_count,
            "stock_count": 0,
            "status": status,
        })
        return branch_id

    def update_branch(self, branch_index: int, **kwargs) -> None:
        """Update fields on a tracked branch."""
        if 0 <= branch_index < len(self.branches):
            self.branches[branch_index].update(kwargs)

"""ProposalTool: apply retrosynthetic templates to generate precursor molecules.
Uses AiZynthFinder's chemistry layer via filesystem import for robust RDChiral application.

When the top-N model-predicted templates don't match (common for simple molecules),
falls back to substructure-based scanning of the full template library and returns
results ranked by template library occurrence frequency.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Try to import aizynthfinder from the filesystem
try:
    _aizynthfinder_path = str(Path(__file__).resolve().parent.parent.parent / "aizynthfinder-master")
    if _aizynthfinder_path not in sys.path:
        sys.path.insert(0, _aizynthfinder_path)
    from aizynthfinder.chem import TreeMolecule
    from aizynthfinder.chem.reaction import TemplatedRetroReaction
    _has_aizynthfinder = True
except ImportError:
    _has_aizynthfinder = False


class ProposalTool:
    name = "propose"
    description = (
        "Apply retrosynthetic templates to generate precursor molecules. "
        "Takes a SMILES and a list of template indices. "
        "Set use_fallback=False to ONLY try the given templates. "
        "Set use_fallback=True if the given templates fail to match, "
        "and the tool will scan the template library for matching reactions "
        "(slower but more comprehensive). "
        "Returns precursor SMILES with template classification and occurrence metadata."
    )

    def __init__(self, templates_path: str | Path,
                 filter_model_path: str | Path | None = None,
                 filter_cutoff: float = 0.05,
                 fallback_scan_limit: int = 5000):
        self._templates = pd.read_hdf(templates_path, "table")
        self.filter_cutoff = filter_cutoff
        self._fallback_scan_limit = fallback_scan_limit
        self._fp_dim = 2048
        self._filter_session = None
        self._filter_input_names = []
        self._filter_output_name = ""

        if filter_model_path and Path(filter_model_path).exists():
            from onnxruntime import InferenceSession
            self._filter_session = InferenceSession(str(filter_model_path))
            self._filter_input_names = [i.name for i in self._filter_session.get_inputs()]
            self._filter_output_name = self._filter_session.get_outputs()[0].name
            self._fp_dim = self._filter_session.get_inputs()[0].shape[1]

    def execute(self, parameters: dict) -> str:
        smiles = parameters["smiles"]
        template_indices = parameters.get("template_indices", [])
        max_results = parameters.get("max_results", 50)
        use_fallback = parameters.get("use_fallback", True)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}", "reactions": []}, ensure_ascii=False)

        if not _has_aizynthfinder:
            return json.dumps({"error": "AiZynthFinder chemistry module not available", "reactions": []},
                              ensure_ascii=False)

        tree_mol = TreeMolecule(smiles=smiles, parent=None, sanitize=True)
        reactions = []

        # Phase 1: try the provided template indices
        attempts = template_indices[:max_results] if template_indices else []
        for tid in attempts:
            row = self._templates.iloc[tid]
            rxns = self._try_template(tree_mol, tid, row)
            reactions.extend(rxns)
        model_match_count = len(reactions)

        # Phase 2: if requested AND model predictions failed, fall back to substructure scan
        if not model_match_count and use_fallback and _has_aizynthfinder:
            reactions = self._fallback_scan(tree_mol, max_results)

        # Dedup
        seen = set()
        unique = []
        for r in reactions:
            key = ".".join(sorted(r["precursors"]))
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return json.dumps({
            "reactions": unique,
            "count": len(unique),
            "model_matches": model_match_count if attempts else 0,
            "fallback_used": bool(not model_match_count and use_fallback and reactions),
        }, ensure_ascii=False)

    def _try_template(self, tree_mol, tid: int, row) -> list[dict]:
        """Try to apply a single template to the molecule."""
        smarts = row["retro_template"]
        classification = str(row.get("classification", ""))
        occurrence = int(row.get("library_occurence", 0))
        results = []
        try:
            reaction = TemplatedRetroReaction(tree_mol, smarts=smarts, use_rdchiral=True, metadata={})
            for rct_set in reaction.reactants:
                precursor_clean = []
                for m in rct_set:
                    precursor_clean.append(Chem.MolToSmiles(m.rd_mol, canonical=True))
                results.append({
                    "template_index": int(tid),
                    "classification": classification,
                    "library_occurrence": occurrence,
                    "precursors": precursor_clean,
                })
        except Exception:
            pass
        return results

    def _fallback_scan(self, tree_mol, max_results: int) -> list[dict]:
        """Scan template library for any that match the molecule.
        Returns results ranked by library_occurrence (most common reactions first)."""
        all_matches = []
        limit = min(self._fallback_scan_limit, len(self._templates))
        for i in range(limit):
            row = self._templates.iloc[i]
            rxns = self._try_template(tree_mol, i, row)
            all_matches.extend(rxns)
            if len(all_matches) >= max_results * 3:
                break

        # Sort by occurrence count (more common = more reliable)
        all_matches.sort(key=lambda x: -x["library_occurrence"])
        return all_matches[:max_results]

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "Target molecule SMILES"},
                "template_indices": {
                    "type": "array", "items": {"type": "integer"},
                    "description": "Template indices from disconnect results. Pass empty array to skip model predictions and directly fallback-scan."
                },
                "max_results": {"type": "integer", "description": "Max reactions to return (default 50)"},
                "use_fallback": {
                    "type": "boolean",
                    "description": "Set to true if model-predicted templates don't match the molecule (check disconnect's `matching` flag and functional_groups). Scans the full template library for applicable reactions."
                },
            },
            "required": ["smiles"]
        }

"""LiteratureTool: search for known synthetic routes / literature precedents.

Phase 1: searches the internal template database for template frequency,
classification, and metadata. Phase 2 will add RAG over full-text literature. """

import json
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class LiteratureTool:
    name = "search_literature"
    description = (
        "Search for known synthetic routes and literature precedents for a target "
        "molecule or substructure. Returns template frequencies, known reaction types, "
        "and (in future versions) full literature references."
    )

    def __init__(self, templates_path: str | Path | None = None):
        self._templates = None
        if templates_path and Path(templates_path).exists():
            self._templates = pd.read_hdf(templates_path, "table")

    def execute(self, parameters: dict) -> str:
        smiles = parameters.get("smiles", "")
        mode = parameters.get("mode", "substructure")

        if mode == "template_info":
            return self._template_info(parameters.get("template_indices", []))
        elif mode == "classification":
            return self._classification_summary(smiles)
        else:
            return self._search_by_substructure(smiles)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES string to research"},
                "mode": {
                    "type": "string",
                    "enum": ["substructure", "template_info", "classification"],
                    "description": "Search mode"
                },
                "template_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Template indices for template_info mode"
                },
            },
            "required": ["smiles"]
        }

    def _template_info(self, indices: list[int]) -> str:
        if self._templates is None:
            return json.dumps({"error": "Template library not loaded"})
        rows = self._templates.iloc[indices]
        results = []
        for _, row in rows.iterrows():
            results.append({
                "template": row.get("retro_template", ""),
                "classification": str(row.get("classification", "")),
                "library_occurence": int(row.get("library_occurence", 0)),
            })
        return json.dumps({"templates": results}, ensure_ascii=False)

    def _classification_summary(self, smiles: str) -> str:
        if self._templates is None:
            return json.dumps({"error": "Template library not loaded"})
        classifications = self._templates["classification"].value_counts().head(30).to_dict()
        return json.dumps({
            "smiles": smiles,
            "total_templates": len(self._templates),
            "common_classifications": classifications,
        }, ensure_ascii=False)

    def _search_by_substructure(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return json.dumps({"error": f"Invalid SMILES: {smiles}", "known_routes": []})

        # For now, return classifications that might be relevant
        result = {
            "smiles": smiles,
            "num_atoms": mol.GetNumAtoms(),
            "rings": mol.GetRingInfo().NumRings(),
            "known_routes": [],
            "note": "Full literature search (RAG over Reaxys/patents) will be in Phase 2.",
        }

        if self._templates is not None:
            result["total_templates_available"] = len(self._templates)
            result["common_classifications"] = (
                self._templates["classification"].value_counts().head(10).to_dict()
            )

        return json.dumps(result, ensure_ascii=False)

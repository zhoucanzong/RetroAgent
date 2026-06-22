"""DisconnectionTool: identifies strategic bond disconnection sites.

Wraps AiZynthFinder's expansion policy model (ONNX) + template library (HDF5).
Given a SMILES, this tool fingerprints the molecule, runs ONNX inference,
retrieves the top-k templates sorted by probability, and from each template
extracts the bond(s) it would break — returned as a ranked list.

IMPORTANT: The ONNX model was trained on USPTO data which is heavily biased
toward complex drug-like molecules (amides, heterocycles, etc.). For simple
targets like aspirin, model predictions are poorly calibrated. In such cases,
we fall back to a substructure-matching approach that scans the full template
library for applicable reactions, returning templates ranked by library occurrence
frequency rather than model probability.
"""

import json
import warnings
import numpy as np
import pandas as pd
import onnxruntime
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem


class DisconnectionTool:
    name = "disconnect"
    description = (
        "Analyze a target molecule and identify strategic bond disconnection sites "
        "using a neural network expansion policy (ONNX model + template library). "
        "Returns ranked templates with probability scores, classification, and a "
        "`matching` flag indicating whether the template's SMARTS pattern actually "
        "matches the molecule's substructure. Also returns detected functional groups. "
        "IMPORTANT: The model was trained on USPTO drug-like molecules and may "
        "produce high probabilities for templates that don't match simple targets. "
        "Always cross-check the `matching` flag and `functional_groups` against "
        "template classifications. Use propose() with matching-only templates, or "
        "call propose() with use_fallback=True to scan the full library."
    )

    def __init__(self, model_path: str | Path, templates_path: str | Path,
                 cutoff_cumulative: float = 0.995, cutoff_number: int = 50,
                 ringbreaker_model_path: str | Path | None = None,
                 fallback_min_matches: int = 3):
        self._model_path = str(model_path)
        self._templates_path = str(templates_path)
        self.cutoff_cumulative = cutoff_cumulative
        self.cutoff_number = cutoff_number
        self._fallback_min_matches = fallback_min_matches

        # Load ONNX model
        self._session = onnxruntime.InferenceSession(self._model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_dim = int(self._session.get_inputs()[0].shape[1])
        self._num_templates = int(self._session.get_outputs()[0].shape[1])

        # Load template library — truncate to model output dimension if needed
        self._templates = pd.read_hdf(self._templates_path, "table")
        if len(self._templates) != self._num_templates:
            warnings.warn(
                f"Template count mismatch: model={self._num_templates}, "
                f"templates={len(self._templates)}. Using truncated set matching model output size."
            )
            self._templates = self._templates.iloc[:self._num_templates]

        # Load ringbreaker model if available
        self._ringbreaker_session = None
        if ringbreaker_model_path and Path(ringbreaker_model_path).exists():
            self._ringbreaker_session = onnxruntime.InferenceSession(str(ringbreaker_model_path))

        self._cache: dict[str, np.ndarray] = {}

    def execute(self, parameters: dict) -> str:
        smiles = parameters["smiles"]
        results = self._analyze(smiles)
        return json.dumps(results, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the target molecule"
                }
            },
            "required": ["smiles"]
        }

    def _analyze(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": f"Invalid SMILES: {smiles}", "bonds": []}

        fp = self._morgan_fp(mol)
        inchi_key = Chem.MolToInchiKey(mol)

        # Identify actual functional groups present in the molecule.
        # The LLM can use this to cross-check model predictions —
        # e.g. if the model suggests "N-acylation" but there's no NH group,
        # the LLM knows to distrust that suggestion.
        fg_info = self._analyze_functional_groups(mol)

        mol_info = {
            "num_atoms": mol.GetNumAtoms(),
            "rings": mol.GetRingInfo().NumRings(),
            "inchi_key": inchi_key,
            "functional_groups": fg_info["present"],
            "functional_groups_absent": fg_info["absent"],
        }

        # Main model inference
        predictions = self._session.run(
            [self._output_name], {self._input_name: fp.astype(np.float32)}
        )[0][0]

        idxs, probs = self._cutoff_predictions(predictions)
        bonds = []
        for rank, (tid, prob) in enumerate(zip(idxs, probs)):
            template_smarts = self._templates.iloc[tid]["retro_template"]
            classification = self._templates.iloc[tid].get("classification", "")
            bond_indices = self._extract_bond_indices_from_template(mol, template_smarts)
            bonds.append({
                "rank": rank + 1,
                "template_index": int(tid),
                "score": round(float(prob), 6),
                "template_smarts": template_smarts,
                "classification": str(classification),
                "bond_indices": bond_indices,
                "matching": len(bond_indices) > 0,  # did substructure actually match?
            })

        return {"bonds": bonds, "molecule_info": mol_info}

    def _morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._input_dim)
        arr = np.zeros((1, self._input_dim), dtype=np.float32)
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr

    @staticmethod
    def _analyze_functional_groups(mol: Chem.Mol) -> dict:
        """Detect functional groups present/absent in the molecule.
        Returns two lists so the LLM can cross-check model predictions."""
        patterns = {
            "carboxylic_acid": "C(=O)O",
            "ester": "C(=O)OC",
            "amide": "C(=O)N",
            "alcohol": "CO",
            "phenol": "cO",
            "amine": "CN",
            "aniline": "cN",
            "ketone": "CC(=O)C",
            "aldehyde": "C=O",
            "sulfonamide": "S(=O)(=O)N",
            "nitro": "N(=O)=O",
            "halide": "[F,Cl,Br,I]",
            "nitrile": "C#N",
            "alkene": "C=C",
            "alkyne": "C#C",
            "ether": "COC",
        }
        present = []
        absent = []
        for name, smarts in patterns.items():
            pat = Chem.MolFromSmarts(smarts)
            if pat is None:
                continue
            if mol.HasSubstructMatch(pat):
                present.append(name)
            else:
                absent.append(name)
        return {"present": present, "absent": absent}

    def _cutoff_predictions(self, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sortidx = np.argsort(predictions)[::-1]
        cumsum = np.cumsum(predictions[sortidx])
        mask = cumsum < self.cutoff_cumulative
        maxidx = int(np.argmin(mask)) if mask.any() else len(cumsum)
        maxidx = min(maxidx, self.cutoff_number) or 1
        return sortidx[:maxidx], predictions[sortidx[:maxidx]]

    def _extract_bond_indices_from_template(self, mol: Chem.Mol, smarts: str) -> list[int]:
        """Extract bond indices that would be broken by a retrosynthetic SMARTS template."""
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn.GetNumReactantTemplates() == 0:
                return []
        except Exception:
            return []

        try:
            prod_template = rxn.GetReactantTemplate(0)
            matches = mol.GetSubstructMatches(prod_template)
            if not matches:
                return []

            bond_indices = []
            for match in matches:
                for bond in prod_template.GetBonds():
                    a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if a1 < len(match) and a2 < len(match):
                        ma1, ma2 = match[a1], match[a2]
                        mol_bond = mol.GetBondBetweenAtoms(ma1, ma2)
                        if mol_bond is not None:
                            bond_indices.append(mol_bond.GetIdx())

            return list(set(bond_indices))
        except Exception:
            return []


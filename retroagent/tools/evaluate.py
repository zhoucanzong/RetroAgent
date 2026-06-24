"""EvaluationTool: assess retrosynthetic route quality.

Combines the filter ONNX model (reaction feasibility) with ZINC stock lookup
(material availability) and basic cheminformatic checks to produce a multi-
dimensional score for each candidate route or reaction. """

import json
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class EvaluationTool:
    name = "evaluate"
    description = (
        "Evaluate the quality and feasibility of retrosynthetic routes. "
        "Returns a multi-dimensional score for each route: feasibility (filter model), "
        "stock availability (ZINC lookup), step count, and a weighted total. "
        "Use this to compare candidate routes before committing to a branch."
    )

    def __init__(self, filter_model_path: str | Path, stock_path: str | Path | None = None,
                 filter_cutoff: float = 0.05):
        self._filter_session = InferenceSession(str(filter_model_path))
        self._filter_input_names = [i.name for i in self._filter_session.get_inputs()]
        self._filter_output_name = self._filter_session.get_outputs()[0].name
        self._fp_dim = self._filter_session.get_inputs()[0].shape[1]  # 2048
        self.filter_cutoff = filter_cutoff

        # Load stock
        self._stock_keys: set[str] = set()
        self._stock_loaded = False
        if stock_path and Path(stock_path).exists():
            stock_df = pd.read_hdf(stock_path, "table")
            self._stock_keys = set(stock_df["inchi_key"].dropna().str.strip())
            self._stock_loaded = True

        # Trivial precursors chemotypes — these can be safely assumed purchasable
        self._trivial_chemotypes = {
            'water': Chem.MolFromSmarts('[OH2]'),
            'simple_alcohol': Chem.MolFromSmarts('[CH3,CH2,CH1][OH]'),
            'simple_acid': Chem.MolFromSmarts('[CH3,CH2,CH1]C(=O)[OH]'),
            'simple_amine': Chem.MolFromSmarts('[CH3,CH2,CH1][NH2,NH]'),
            'simple_ester': Chem.MolFromSmarts('[CH3,CH2,CH1]OC(=O)[CH3,CH2,CH1]'),
            'simple_anhydride': Chem.MolFromSmarts('C(=O)OC(=O)'),
            'methyl_ester': Chem.MolFromSmarts('COC(=O)'),
            'acetyl': Chem.MolFromSmarts('CC(=O)'),
        }

    def execute(self, parameters: dict) -> str:
        route_ids = parameters.get("route_ids", [])
        target = parameters.get("target", "")
        reactions = parameters.get("reactions", [])  # list of {id, product_smiles, precursor_smiles_list}
        scores = {}
        for rxn in reactions:
            rid = rxn.get("id", "unknown")
            precursor_list = rxn.get("precursor_smiles_list", [])
            feasibility = self._predict_feasibility(rxn.get("product_smiles", ""), precursor_list)
            stock_avail = self._check_stock_availability(precursor_list)
            trivial_bonus = self._trivial_precursor_score(precursor_list)
            scores[rid] = {
                "feasibility": round(feasibility, 4),
                "stock_availability": round(stock_avail, 4),
                "precursor_count": len(precursor_list),
                "total": round(0.6 * feasibility + 0.4 * stock_avail, 4),
                "trivial_precursor_score": round(trivial_bonus, 4),
            }
        return json.dumps({"scores": scores, "target": target}, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "route_ids": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string", "description": "Target molecule SMILES"},
                "reactions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "product_smiles": {"type": "string"},
                            "precursor_smiles_list": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
            "required": ["reactions"]
        }

    def _predict_feasibility(self, product_smiles: str, precursor_smiles_list: list[str]) -> float:
        if not product_smiles or not precursor_smiles_list:
            return 0.0
        try:
            prod_mol = Chem.MolFromSmiles(product_smiles)
            if prod_mol is None:
                return 0.0
            prod_fp = self._morgan_fp(prod_mol)

            # Reaction diff fingerprint
            rxn_fp = np.zeros((1, self._fp_dim), dtype=np.float32)
            for smi in precursor_smiles_list:
                m = Chem.MolFromSmiles(smi)
                if m:
                    fp = np.zeros((1, self._fp_dim), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(
                        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=self._fp_dim), fp[0])
                    rxn_fp += fp
            rxn_fp -= prod_fp

            result = self._filter_session.run(
                [self._filter_output_name],
                {self._filter_input_names[0]: prod_fp.astype(np.float32),
                 self._filter_input_names[1]: rxn_fp.astype(np.float32)},
            )[0][0][0]
            return float(result)
        except Exception:
            return 0.0

    def _check_stock_availability(self, precursor_smiles_list: list[str]) -> float:
        if not self._stock_loaded or not precursor_smiles_list:
            return 0.5
        in_stock = 0
        for smi in precursor_smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol and Chem.MolToInchiKey(mol) in self._stock_keys:
                    in_stock += 1
            except Exception:
                pass
        return in_stock / len(precursor_smiles_list) if precursor_smiles_list else 0.0

    def _trivial_precursor_score(self, smiles_list: list[str]) -> float:
        """Heuristic score for precursors that are trivially purchasable or synthesizable."""
        trivial_count = 0
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            for _, pattern in self._trivial_chemotypes.items():
                if mol.HasSubstructMatch(pattern):
                    trivial_count += 1
                    break
        return trivial_count / len(smiles_list) if smiles_list else 0.0

    def _morgan_fp(self, mol: Chem.Mol) -> np.ndarray:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self._fp_dim)
        arr = np.zeros((1, self._fp_dim), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr[0])
        return arr

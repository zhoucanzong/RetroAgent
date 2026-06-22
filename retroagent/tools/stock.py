"""StockTool: check whether molecules are commercially available.

Wraps the ZINC stock HDF5 file (17.4M InChI Keys). Given a list of SMILES,
returns which ones are purchasable starting materials. """

import json
import pandas as pd
from pathlib import Path
from rdkit import Chem


class StockTool:
    name = "check_stock"
    description = (
        "Check whether molecule(s) are commercially available (in the ZINC database). "
        "Input a list of SMILES, returns which are in stock and which are not. "
        "A complete synthetic route must have ALL precursors in stock."
    )

    def __init__(self, stock_path: str | Path):
        stock_df = pd.read_hdf(stock_path, "table")
        self._stock_keys: set[str] = set(stock_df["inchi_key"].dropna().str.strip())
        self._stock_count = len(self._stock_keys)

    def execute(self, parameters: dict) -> str:
        smiles_list = parameters.get("smiles_list", [])
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        in_stock: dict[str, str] = {}
        not_in_stock: list[str] = []
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi.strip())
                if mol is None:
                    not_in_stock.append(smi)
                    continue
                inchi_key = Chem.MolToInchiKey(mol)
                if inchi_key in self._stock_keys:
                    in_stock[smi] = "ZINC"
                else:
                    not_in_stock.append(smi)
            except Exception:
                not_in_stock.append(smi)

        return json.dumps({
            "in_stock": in_stock,
            "not_in_stock": not_in_stock,
            "all_available": len(not_in_stock) == 0,
            "total_checked": len(smiles_list),
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings to check"
                }
            },
            "required": ["smiles_list"]
        }

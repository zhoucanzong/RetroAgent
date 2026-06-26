"""PatentTool: check whether a molecule is purchasable / in-stock / patented.

Uses molbloom (https://github.com/whitead/molbloom) — a tiny local library that
bundles ~10M InChIKey bloom filters from ZINC/Emolecules inventories. No network,
no API key. Complements RetroAgent's ZINC-based check_stock: molbloom answers
'is this molecule in a building-block catalog' which is closer to commercial
purchasability than ZINC's screening-compound set.

Claude philosophy: tool REPORTS the boolean + which catalog. It does NOT decide
whether to keep/discard a route — the LLM (Auditor) interprets.
"""

import json
from rdkit import Chem


class PatentTool:
    name = "check_patent"
    description = (
        "Check whether a molecule is purchasable / in a building-block catalog "
        "via molbloom (local bloom filters over ZINC/Emolecules inventories, ~10M "
        "compounds). No network/key needed. More reliable for common reagents and "
        "building blocks than ZINC screening-compound lookup. Returns a boolean "
        "per molecule plus which catalog matched. REPORTS facts only."
    )

    def execute(self, parameters: dict) -> str:
        smiles_list = parameters.get("smiles_list", [])
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        if not smiles_list:
            return json.dumps({"error": "Missing 'smiles_list'"}, ensure_ascii=False)

        try:
            from molbloom import buy
        except ImportError:
            return json.dumps({"error": "molbloom not installed (pip install molbloom)"},
                              ensure_ascii=False)

        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append({"smiles": smi, "valid": False, "purchasable": False,
                                "note": "invalid SMILES"})
                continue
            canon = Chem.MolToSmiles(mol, canonical=True)
            inchikey = Chem.MolToInchiKey(mol)
            # molbloom v2+: buy() returns True/False, or a catalog string on hit
            try:
                hit = buy(canon)
            except Exception as e:
                hit = False
                note = f"lookup error: {e}"
            # molbloom may return True or a catalog name (str); normalize
            purchasable = bool(hit)
            catalog = hit if isinstance(hit, str) else ("building-blocks" if hit else None)
            results.append({
                "smiles": canon,
                "inchikey": inchikey,
                "valid": True,
                "purchasable": purchasable,
                "catalog": catalog,
            })

        n_hit = sum(1 for r in results if r["purchasable"])
        return json.dumps({
            "checked": len(results),
            "purchasable_count": n_hit,
            "results": results,
            "note": ("molbloom bloom-filter lookup. A hit means the molecule is "
                     "likely in a building-block catalog (commercially purchasable). "
                     "A miss is NOT definitive for novel/specialty compounds."),
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array", "items": {"type": "string"},
                    "description": "SMILES strings to check for purchasability",
                },
            },
            "required": ["smiles_list"],
        }

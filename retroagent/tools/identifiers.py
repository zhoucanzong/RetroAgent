"""IdentifierTool: bidirectional conversion between chemistry identifiers.

Uses PubChem PUG REST (free, no key). Converts between:
  name  <-> SMILES
  IUPAC <-> SMILES
  SMILES -> CAS, molecular formula

Claude philosophy: tool only FETCHES and converts. No judgment.
"""

import json
import urllib.parse
import logging

logger = logging.getLogger("retroagent")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_TIMEOUT = 20
_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class IdentifierTool:
    name = "convert_identifier"
    description = (
        "Convert between chemistry identifiers using PubChem PUG REST (free, no key). "
        "Supports: name->SMILES, SMILES->IUPAC, IUPAC->SMILES, SMILES->CAS, "
        "SMILES->formula. Useful when the task names a molecule in words/IUPAC "
        "and you need its SMILES, or to look up a molecule's CAS/formula."
    )

    def execute(self, parameters: dict) -> str:
        if not _HAS_REQUESTS:
            return json.dumps({"error": "requests not installed"}, ensure_ascii=False)
        conversion = parameters.get("conversion", "")
        value = (parameters.get("value") or "").strip()
        if not conversion or not value:
            return json.dumps({"error": "Need 'conversion' and 'value'"}, ensure_ascii=False)

        try:
            if conversion == "name_to_smiles":
                return self._wrap(self._pug_get(f"compound/name/{urllib.parse.quote(value)}/property/CanonicalSMILES,ConnectivitySMILES/JSON"),
                                  out_keys=["CanonicalSMILES", "ConnectivitySMILES"],
                                  label="canonical_smiles")
            if conversion == "smiles_to_iupac":
                return self._wrap(self._pug_post_smiles(value, "IUPACName"),
                                  out_keys=["IUPACName"], label="iupac_name")
            if conversion == "iupac_to_smiles":
                return self._wrap(self._pug_get(f"compound/name/{urllib.parse.quote(value)}/property/CanonicalSMILES,ConnectivitySMILES/JSON"),
                                  out_keys=["CanonicalSMILES", "ConnectivitySMILES"],
                                  label="canonical_smiles")
            if conversion == "smiles_to_cas":
                return self._wrap_cas(value)
            if conversion == "smiles_to_formula":
                return self._wrap(self._pug_post_smiles(value, "MolecularFormula"),
                                  out_keys=["MolecularFormula"], label="molecular_formula")
            return json.dumps({"error": f"Unknown conversion: {conversion}. "
                                        "Use name_to_smiles|smiles_to_iupac|iupac_to_smiles|"
                                        "smiles_to_cas|smiles_to_formula"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "conversion": {
                    "type": "string",
                    "enum": ["name_to_smiles", "smiles_to_iupac", "iupac_to_smiles",
                             "smiles_to_cas", "smiles_to_formula"],
                },
                "value": {"type": "string", "description": "Input identifier (name, SMILES, or IUPAC)"},
            },
            "required": ["conversion", "value"],
        }

    examples = [
        {"input": {"conversion": "name_to_smiles", "value": "ibuprofen"},
         "output": {"canonical_smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1"}},
        {"input": {"conversion": "smiles_to_iupac", "value": "CC(=O)Oc1ccccc1C(=O)O"},
         "output": {"iupac_name": "2-acetyloxybenzoic acid"}},
    ]

    # ------------------------------------------------------------------

    @staticmethod
    def _pug_get(path: str) -> dict:
        r = requests.get(f"{_PUG}/{path}", timeout=_TIMEOUT)
        if not r.ok:
            raise RuntimeError(f"PubChem HTTP {r.status_code}")
        return r.json()

    @staticmethod
    def _pug_post_smiles(smiles: str, prop: str) -> dict:
        # POST because SMILES may be long / contain special chars
        r = requests.post(f"{_PUG}/compound/smiles/property/{prop}/JSON",
                          data={"smiles": smiles}, timeout=_TIMEOUT)
        if not r.ok:
            raise RuntimeError(f"PubChem HTTP {r.status_code}")
        return r.json()

    @staticmethod
    def _wrap(resp: dict, out_keys: list, label: str) -> str:
        """Extract first non-null value among out_keys per property row."""
        props = resp.get("PropertyTable", {}).get("Properties", [])
        results = []
        for p in props:
            val = None
            for k in out_keys:
                if p.get(k):
                    val = p[k]
                    break
            results.append({"cid": p.get("CID"), label: val})
        return json.dumps({"conversion": label, "results": results,
                           "count": len(results)}, ensure_ascii=False)

    @staticmethod
    def _wrap_cas(smiles: str) -> str:
        # CAS isn't a standard PUG property; query the Synonyms list and extract CAS-like
        r = requests.post(f"{_PUG}/compound/smiles/synonyms/TXT",
                          data={"smiles": smiles}, timeout=_TIMEOUT)
        if not r.ok:
            return json.dumps({"error": f"PubChem HTTP {r.status_code}"}, ensure_ascii=False)
        import re
        cas_pattern = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
        cas_hits = sorted(set(cas_pattern.findall(r.text)))
        return json.dumps({"conversion": "cas", "cas_numbers": cas_hits,
                           "count": len(cas_hits)}, ensure_ascii=False)

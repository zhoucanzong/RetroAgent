"""ConditionTool: recommend reaction conditions (catalyst, solvent, temperature, etc.).

Phase 1: returns typical conditions based on template classification statistics.
Phase 2: will train a dedicated condition prediction model from literature data. """

import json


class ConditionTool:
    name = "recommend_conditions"
    description = (
        "Recommend reaction conditions (catalyst, solvent, temperature, time) "
        "for a given reaction. Phase 1: returns typical conditions based on "
        "reaction classification. Phase 2: ML-predicted conditions."
    )

    def execute(self, parameters: dict) -> str:
        product = parameters.get("product", "")
        precursors = parameters.get("precursors", [])
        reaction_class = parameters.get("reaction_classification", "")

        conditions = self._lookup_conditions(reaction_class, product, precursors)
        return json.dumps({"conditions": conditions}, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "product": {"type": "string", "description": "Product SMILES"},
                "precursors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Precursor SMILES list"
                },
                "reaction_classification": {
                    "type": "string",
                    "description": "Reaction classification (e.g., 'ester hydrolysis', 'amide coupling')"
                },
            },
            "required": ["product", "precursors"]
        }

    def _lookup_conditions(self, reaction_class: str, product: str, precursors: list[str]) -> dict:
        # Phase 1: rule-of-thumb conditions by reaction class
        lookup = {
            "ester hydrolysis": {
                "catalyst": "NaOH or LiOH", "solvent": "THF/H2O or MeOH/H2O",
                "temperature": "0°C to rt", "time": "1-4h",
                "note": "Basic hydrolysis; use LiOH for sensitive substrates"
            },
            "amide coupling": {
                "catalyst": "EDC/HOBt or HATU/DIPEA", "solvent": "DMF or DCM",
                "temperature": "0°C to rt", "time": "2-16h",
            },
            "NH deprotections": {
                "catalyst": "TFA or Pd/C, H2", "solvent": "DCM or MeOH",
                "temperature": "rt", "time": "1-4h",
            },
            "suzuki coupling": {
                "catalyst": "Pd(PPh3)4 or Pd(dppf)Cl2", "solvent": "DME/H2O or toluene/EtOH",
                "temperature": "80-100°C", "time": "4-24h",
                "additive": "K2CO3 or Na2CO3 (base)",
            },
            "reductive amination": {
                "catalyst": "NaBH(OAc)3 or NaBH3CN", "solvent": "DCE or THF",
                "temperature": "rt", "time": "1-16h",
            },
            "boc deprotection": {
                "catalyst": "TFA or HCl/dioxane", "solvent": "DCM",
                "temperature": "0°C to rt", "time": "1-4h",
            },
        }

        key = reaction_class.lower().strip()
        for pattern, conds in lookup.items():
            if pattern in key:
                return conds

        return {
            "catalyst": "to be determined",
            "solvent": "to be determined",
            "temperature": "to be determined",
            "time": "to be determined",
            "note": f"No pre-computed conditions for '{reaction_class}'. Full ML model in Phase 2."
        }

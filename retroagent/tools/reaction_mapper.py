"""ReactionMapperTool: atom-to-atom mapping of a reaction via rxnmapper.

rxnmapper (IBM, open source) maps which reactant atom becomes which product atom
using a transformer model. Given an unmapped reaction SMILES, returns the
atom-mapped reaction. Useful for: verifying bond-making/breaking consistency,
mechanism analysis, checking template application correctness.

Claude philosophy: tool only COMPUTES the mapping. It does NOT judge whether the
reaction is feasible — the LLM interprets the mapping.

Note: the rxnmapper model (~200MB) downloads from HuggingFace on first use and is
cached locally. No API key. CPU inference, ~1-3s per reaction.
"""

import json
import logging

logger = logging.getLogger("retroagent")


class ReactionMapperTool:
    name = "map_reaction"
    description = (
        "Atom-to-atom map a chemical reaction using rxnmapper (IBM transformer "
        "model, local, no key). Input a reaction SMILES 'reactants>>product', "
        "returns the atom-mapped reaction with [atom:n] labels showing which "
        "reactant atom becomes which product atom. Use to verify bond-breaking/"
        "forming consistency, understand mechanisms, or check that a proposed "
        "disconnection is chemically sound."
    )

    def __init__(self):
        self._mapper = None  # lazy-load on first use

    def _get_mapper(self):
        if self._mapper is None:
            from rxnmapper import RXNMapper
            logger.info("Loading rxnmapper model (first use, may download ~200MB)...")
            self._mapper = RXNMapper()
        return self._mapper

    def execute(self, parameters: dict) -> str:
        rxn = (parameters.get("reaction") or "").strip()
        if not rxn:
            return json.dumps({"error": "Missing 'reaction' (format: reactants>>product)"},
                              ensure_ascii=False)
        if ">>" not in rxn:
            return json.dumps({"error": "Reaction must contain '>>' separator "
                                        "(format: reactants>>product)"}, ensure_ascii=False)

        try:
            mapper = self._get_mapper()
            results = mapper.get_attention_guided_atom_maps([rxn])
            if not results:
                return json.dumps({"error": "No mapping produced", "input": rxn},
                                  ensure_ascii=False)
            mapped = results[0]
            return json.dumps({
                "input_reaction": rxn,
                "mapped_reaction": mapped.get("mapped_rxn", ""),
                "confidence": round(mapped.get("confidence", 0.0), 4),
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"{type(e).__name__}: {e}", "input": rxn},
                              ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "reaction": {
                    "type": "string",
                    "description": "Reaction SMILES, e.g. 'CC(=O)O.OCC>>CC(=O)OCC.O' (esterification)",
                },
            },
            "required": ["reaction"],
        }

    examples = [
        {"input": {"reaction": "CC(=O)O.OCC>>CC(=O)OCC.O"},
         "output": {"mapped_reaction": "[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:4][CH2:6][CH3:7].[OH:5]",
                    "confidence": 0.95}},
    ]

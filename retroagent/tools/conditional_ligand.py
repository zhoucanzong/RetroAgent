"""ConditionalLigandTool: generate candidate chiral ligand SMILES from constraints.

Uses LLM few-shot examples sampled from CIC-DB's gen_conditional task.
The tool is intentionally pure data-provider: it returns candidates, and the
Planner (LLM) evaluates them with ChiralityTool / LigandCategoryTool / evaluate.
"""

import json
import random
from pathlib import Path

from rdkit import Chem


class ConditionalLigandTool:
    name = "design_ligand"
    description = (
        "Generate candidate chiral ligand/catalyst SMILES from a set of constraints "
        "(chirality type, scaffold family, coordinating atoms, property targets). "
        "Returns multiple candidates. The Planner must validate the output with "
        "analyze_chirality, classify_ligand, and evaluate tools."
    )

    def __init__(self, examples_path: str | None = None, n_examples: int = 3):
        self.n_examples = n_examples
        self._examples: list[dict] = []
        if examples_path:
            self._load_examples(examples_path)

    def _load_examples(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p) as f:
                for line in f:
                    data = json.loads(line)
                    msgs = data.get("messages", [])
                    if len(msgs) >= 3:
                        self._examples.append({
                            "prompt": msgs[1]["content"],
                            "smiles": msgs[2]["content"],
                        })
        except Exception:
            self._examples = []

    def execute(self, parameters: dict) -> str:
        constraints = parameters.get("constraints", "")
        count = parameters.get("count", 3)
        model_name = parameters.get("model")  # optional override

        # Build few-shot prompt
        examples = self._sample_examples(self.n_examples)
        prompt = self._build_prompt(constraints, examples, count)

        # Generate candidates via LLM
        candidates = self._generate(prompt, model_name, count)

        # Validate and enrich
        validated = []
        for c in candidates:
            c = c.strip()
            mol = Chem.MolFromSmiles(c)
            if mol is None:
                continue
            validated.append({
                "smiles": c,
                "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
                "valid": True,
            })

        return json.dumps({
            "constraints": constraints,
            "candidates": validated,
            "count": len(validated),
            "prompt": prompt if parameters.get("include_prompt") else None,
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "constraints": {
                    "type": "string",
                    "description": "Natural-language constraints, e.g. 'Point chirality ligand with P and O donor atoms, BIBOP-like scaffold'"
                },
                "count": {"type": "integer", "default": 3, "description": "Number of candidates to generate"},
                "model": {"type": "string", "description": "Optional LLM model override"},
                "include_prompt": {"type": "boolean", "description": "Return the generated prompt for debugging"},
            },
            "required": ["constraints"]
        }

    def _sample_examples(self, n: int) -> list[dict]:
        if not self._examples:
            return []
        return random.sample(self._examples, min(n, len(self._examples)))

    @staticmethod
    def _build_prompt(constraints: str, examples: list[dict], count: int) -> str:
        parts = [
            "You are a cheminformatics assistant specialized in designing chiral ligands and catalysts.",
            "Given constraints, return only valid SMILES strings, one per line, no explanations.",
            "Each SMILES should encode a stable, realistic chiral ligand or catalyst matching the constraints.",
            "",
            "Examples:",
        ]
        for ex in examples:
            parts.append(f"Constraints: {ex['prompt']}")
            parts.append(f"SMILES: {ex['smiles']}")
            parts.append("")
        parts.append(f"Constraints: {constraints}")
        parts.append(f"Generate {count} candidate SMILES:")
        return "\n".join(parts)

    def _generate(self, prompt: str, model_name: str | None, count: int) -> list[str]:
        # Use the configured LLM if available; otherwise deterministic fallback
        candidates = []
        try:
            from retroagent.config import get_config
            cfg = get_config()
            from openai import OpenAI
            client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
            response = client.chat.completions.create(
                model=model_name or cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
                n=1,
            )
            text = response.choices[0].message.content or ""
            # Extract SMILES-looking lines
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Remove prefixes like "SMILES: "
                if "SMILES:" in line:
                    line = line.split("SMILES:", 1)[-1].strip()
                if line and len(line) > 3:
                    candidates.append(line)
            if candidates:
                return candidates[:count * 2]
        except Exception:
            pass

        # Fallback deterministic placeholder: return common chiral phosphine-ish cores
        fallback = [
            "CC(C)(C)[P@](c1ccccc1)(c2ccccc2)[P@](c3ccccc3)(c4ccccc4)C(C)(C)C",
            "CC[C@@H]1OC2=CC=CC(=C2[P@]1C(C)(C)C)C3=CC=CC=C3",
            "CC(C)(C)OC1=N[C@@H](C2=NOC(C(C)(C)C)=N2)CO1",
        ]
        return fallback[:count] if count <= len(fallback) else fallback

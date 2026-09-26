"""Run real RetroChimera weights through the RetroAgent adapter on a CPU."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path


def main():
    import torch
    from rdkit import Chem
    from retroagent.tools.retrochimera import (
        RetroChimeraPredictionTool, NeuralLocPredictionTool, TransformerPredictionTool,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--target", default="Oc1ccc(OCc2ccccc2)c(Br)c1")
    parser.add_argument("--output", type=Path, default=Path("logs/retrochimera_smoke/result.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tool", choices=["retrochimera", "neuralloc", "transformer"], default="retrochimera")
    parser.add_argument("--quick", action="store_true", help="Reduce beam/augmentation for a wiring test, not benchmark reproduction")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    options = {
        "smiles_transformer": {},
        "template_localization": {"num_processes": 2},
    }
    if args.quick:
        options["smiles_transformer"].update(beam_size=5, augmentation_size=1)
    parameters = {
        "target_smiles": args.target, "model_dir": str(args.model_dir),
        "device": "cpu", "num_results": 5, "call_submodels_in_parallel": False,
        "submodel_kwargs": options,
    }
    classes = {"retrochimera": RetroChimeraPredictionTool,
               "neuralloc": NeuralLocPredictionTool, "transformer": TransformerPredictionTool}
    if args.tool != "retrochimera":
        parameters.pop("submodel_kwargs")
        parameters.pop("call_submodels_in_parallel")
        component = "template_localization" if args.tool == "neuralloc" else "smiles_transformer"
        parameters["model_kwargs"] = options[component]
    tool = classes[args.tool]()
    started = time.monotonic()
    print(f"Loading weights and running {tool.name}...", flush=True)
    try:
        result = json.loads(tool.execute(parameters))
        elapsed = time.monotonic() - started
        assert result["count"] > 0, "No predictions returned"
        for reaction in result["reactions"]:
            assert all(Chem.MolFromSmiles(s) for s in reaction["precursor_smiles_list"])
        expected = set(options) if args.tool == "retrochimera" else {component}
        assert set(result["reactions"][0]["individual_ranks"]) == expected
        versions = {name: importlib.metadata.version(name) for name in (
            "torch", "retrochimera", "syntheseus", "rdkit", "pytorch-lightning",
        )}
        provenance = args.model_dir / "provenance.json"
        record = {
            "parameters": parameters, "elapsed_seconds": elapsed,
            "versions": versions, "platform": platform.platform(),
            "checkpoint_provenance": json.loads(provenance.read_text()) if provenance.exists() else None,
            "result": result,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"elapsed_seconds": elapsed, "count": result["count"],
                          "reactions": result["reactions"], "output": str(args.output)}, indent=2), flush=True)
    finally:
        tool.close()


if __name__ == "__main__":
    main()

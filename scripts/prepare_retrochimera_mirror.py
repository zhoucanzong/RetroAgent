"""Verify and reconstruct inference checkpoints from a pinned safetensors mirror.

The mirror is community-maintained, not a Microsoft release endpoint. Verification
checks integrity against that mirror, not equivalence to the unavailable original.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Pinned Hugging Face commit SHA")
    parser.add_argument("--output", type=Path, default=Path("models/retrochimera"))
    args = parser.parse_args()
    prefix = "checkpoints/retrochimera_hybrid/pistachio"
    root = args.output.resolve()
    snapshot_download(
        repo_id="Ian-L/backwards-models-v2", repo_type="dataset",
        revision=args.revision, local_dir=root / "mirror",
        allow_patterns=[f"{prefix}/*"], ignore_patterns=[f"{prefix}/benchmarks/*"],
        max_workers=3,
    )
    source = root / "mirror" / prefix
    verified = {}
    for line in (source / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        file = source / relative.lstrip("*")
        if "benchmarks" in file.parts:
            continue
        if not file.resolve().is_relative_to(source):
            raise ValueError("Checksum path escapes bundle")
        with file.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {relative}")
        verified[relative] = actual
        print(f"Verified {relative}", flush=True)
    output = root / "pistachio"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "aux/models.json", output / "models.json")
    for name in ("smiles_transformer", "template_localization"):
        destination = output / name
        destination.mkdir(exist_ok=True)
        aux = source / name / "aux"
        metadata = json.loads((aux / "checkpoint_metadata.json").read_text())
        params = json.loads((aux / "hyper_parameters.json").read_text())
        checkpoint = {
            "state_dict": load_file(str(source / name / "model.safetensors")),
            "hyper_parameters": params["hyper_parameters"],
            "pytorch-lightning_version": metadata["pytorch_lightning_version"],
            "epoch": metadata["epoch"], "global_step": metadata["global_step"],
        }
        torch.save(checkpoint, destination / "inference.ckpt")
        del checkpoint
        sidecar = "vocab.txt" if name == "smiles_transformer" else "template_lib.json"
        shutil.copy2(aux / sidecar, destination / sidecar)
        print(f"Reconstructed {name}", flush=True)
    (output / "provenance.json").write_text(json.dumps({
        "mirror": "https://huggingface.co/datasets/Ian-L/backwards-models-v2",
        "revision": args.revision, "verified_sha256": verified,
        "original": json.loads((source / "provenance.json").read_text()),
        "note": "Inference-only reconstruction; original archive equivalence not verified.",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

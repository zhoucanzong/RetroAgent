# RetroChimera CPU Inference

The isolated environment uses Python 3.11 and PyTorch 2.2.2 on Apple Silicon.
It does not replace RetroAgent's primary Python environment.

```bash
uv venv --python 3.11 .venv/retrochimera
uv pip install --python .venv/retrochimera/bin/python -r requirements-retrochimera-cpu.txt -e .
```

Download a Pistachio checkpoint using the links in the
[Microsoft RetroChimera README](https://github.com/microsoft/retrochimera).
The model directory must contain `models.json`, `smiles_transformer/`, and
`template_localization/`. Use trusted checkpoints: upstream Lightning loads
PyTorch serialized files.

## Community Mirror Fallback

The official Figshare endpoints returned HTTP 403 from this machine during the
2026-09-27 run. The fallback below uses the community-maintained
[Ian-L/backwards-models-v2 dataset](https://huggingface.co/datasets/Ian-L/backwards-models-v2),
pinned to a specific revision. This is not a Microsoft-hosted distribution.

```bash
.venv/retrochimera/bin/python scripts/prepare_retrochimera_mirror.py \
  --revision 30cc1962fa8de3decdc32f624b9c2ece324d2b45
```

The script verifies downloaded files against the pinned bundle's SHA256SUMS,
loads safetensors and JSON configuration, and creates inference-only Lightning
checkpoints. It does not recover optimizer states. The output provenance records
the mirror revision, verified hashes, and mirror-provided original archive
metadata. Hash verification proves integrity relative to the mirror; equivalence
to the inaccessible original archive has not been independently verified.

## Run Both Submodels

```bash
PYTHONPATH=. WANDB_MODE=disabled .venv/retrochimera/bin/python \
  scripts/smoke_retrochimera.py \
  --model-dir models/retrochimera/pistachio
```

The script calls RetroAgent's `RetroChimeraPredictionTool` with real weights and
uses the official example target `Oc1ccc(OCc2ccccc2)c(Br)c1`. It requests five
predictions, checks precursor SMILES, requires ranks from both constituent
models, and writes timing, versions, provenance, and predictions to
`logs/retrochimera_smoke/result.json`. This is an inference integration test, not
a blind chemical-reasoning benchmark or experimental validation.

Default inference retains the upstream beam size of 20 and augmentation size
of 10. `--quick` uses beam size 5 and augmentation size 1 for a faster wiring
test. The CPU runner uses four Torch threads, two rule-application workers, and
sequential submodel execution. `submodel_kwargs` are included in the adapter's
model cache key. The adapter disables autograd for actual model inference.

`canonicalization_processes` belongs to a newer upstream main revision and is
not passed to the pinned 1.3.0 release. Model files, environment files, and raw
run outputs are ignored by Git; the requirements and scripts are versioned.

## Verified Run: 2026-09-27

Both real submodels completed through `RetroChimeraPredictionTool` on Apple
Silicon CPU with the default decoding settings above. Cold initialization plus
prediction took 172.95 seconds. The rule workers loaded 73,128 templates each
(146,256 total). Five predictions were returned; all precursor SMILES parsed.

| Rank | Reactants | Ensemble probability | Transformer / NeuralLoc rank (1-based) |
| --- | --- | --- | --- |
| 1 | `BrCc1ccccc1` + `Oc1ccc(O)c(Br)c1` | 95.72% | 1 / 1 |
| 2 | `O=C(OO)c1cccc(Cl)c1` + `O=Cc1ccc(OCc2ccccc2)c(Br)c1` | 3.92% | 2 / 2 |
| 3 | `CC(=O)Oc1ccc(OCc2ccccc2)c(Br)c1` | 0.21% | 3 / 4 |

These probabilities are normalized ranking probabilities, not success rates.
The raw upstream `individual_ranks` in the JSON output are zero-based.
Full results: `logs/retrochimera_smoke/result.json`; full runtime log:
`logs/retrochimera_smoke/full_inference.log`.

The process exited with status 0. Upstream emitted deprecation notices and one
semaphore-cleanup warning at shutdown; no prediction worker remained running.

## Agent-Selected Standalone Tools

RetroAgent exposes three independent tools. `neuralloc_predict` calls only
NeuralLoc; `transformer_predict` calls only the chemistry Transformer (R-SMILES 2);
`retrochimera_predict` calls the upstream ensemble. No model-call sequence is
mandatory. In a persistent Agent environment they share a bounded cache: changing
the model/configuration releases the previous model and its template workers.

```bash
# Reproduce either standalone model through the same adapter used by the Agent.
PYTHONPATH=. .venv/retrochimera/bin/python scripts/smoke_retrochimera.py \
  --tool neuralloc --model-dir models/retrochimera/pistachio \
  --output logs/retrochimera_smoke/neuralloc_standalone.json
PYTHONPATH=. .venv/retrochimera/bin/python scripts/smoke_retrochimera.py \
  --tool transformer --model-dir models/retrochimera/pistachio \
  --output logs/retrochimera_smoke/transformer_standalone.json

# Agent-native CLI in the compatible neural runtime.
.venv/retrochimera/bin/retroagent project init ./my-research
.venv/retrochimera/bin/retroagent tool call transformer_predict \
  --project ./my-research --json --parameters \
  '{"target_smiles":"Oc1ccc(OCc2ccccc2)c(Br)c1","model_dir":"models/retrochimera/pistachio","num_results":5}'
```

Each response includes an `ensemble_source` object. Pass the two objects directly
as `ensemble_disconnections.sources` to fuse predictions, preserving their
checkpoint identities. Renaming a source does not make repeated calls independent.
The fusion tool rejects an ensemble combined with its own constituent sources.
Model logs (including worker stdout) go to stderr so `--json` stays parseable.
Prediction and fusion calls are recorded separately from validation metrics.

# Optional Retrosynthesis Scaffolding

| Tool | Capability | Typical reason to call |
| --- | --- | --- |
| `neuralloc_predict` | Template selection and reaction-site localization | Inspect an editing model's proposals |
| `transformer_predict` | R-SMILES 2 precursor generation | Seek complementary de-novo proposals |
| `retrochimera_predict` | Upstream learned ensemble of both models | Obtain a combined candidate ranking directly |
| `ensemble_disconnections` | Canonical reactant-set fusion | Compare independently sourced ranked candidates |

The chemistry Transformer is not the host Agent's language model. The Agent chooses
which prediction tools are useful, may reject their outputs, and may develop other
hypotheses. Calling all tools is not a completion requirement.

All three model tools accept `target_smiles`, `model_dir`, `num_results` (1-10), and
`device`. A single-model tool accepts the ensemble root or its component directory
and optional `model_kwargs`; the ensemble accepts `submodel_kwargs` and
`consensus_only`. Inspect the current schema before constructing arguments.

Use each returned `ensemble_source` unchanged when composing fusion inputs. Stable
`source_ids` identify checkpoint paths; fusion rejects overlapping identities, e.g.
the ensemble plus its own submodels, or renamed copies of the same call. Independent
names are not proof of independent training, and copying weights to a new path does
not create an independent model. Do not discard provenance to evade overlap checks.

Raw `individual_ranks` from upstream models are zero-based; top-level ranks are
one-based and labelled. Ensemble scores and normalized probabilities are ranking
signals, not reaction yields or success probabilities. Do not average arbitrary
raw scores from different architectures as calibrated feasibility.

Runtime model selection retains one active neural configuration per environment.
Switching models releases the previous template workers; repeated calls with the
same settings reuse the loaded model. Prefer the persistent MCP tool for repeated
inference. Prior local CPU smoke tests found NeuralLoc cold-start dominated runtime;
do not assume those timings apply to a different host.

These tools integrate [Microsoft RetroChimera](https://github.com/microsoft/retrochimera)
and its [paper](https://arxiv.org/abs/2412.05269). Upstream predictions and local
validation gates still require independent chemical assessment for real experiments.

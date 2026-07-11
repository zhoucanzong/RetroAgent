# RetroAgent Benchmarks

This directory contains small, versionable benchmark suites for agent behavior.

## discovery_small.jsonl

Initial discovery-mode smoke benchmark. These tasks are not meant to prove
experimental success. They check whether RetroAgent can produce a portfolio of
route hypotheses with:

- route diversity
- evidence labels (`known`, `analog-supported`, `tool-supported`, `speculative`)
- risk flags
- stock/literature/tool validation notes
- next experimental checks

The tasks intentionally avoid aspirin-style toy validation as a success signal.

Dry-run the suite without LLM calls:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery --dry-run
```

Run against the configured LLM:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery --run --limit 1
```

Results are written to `logs/discovery_benchmark/results.jsonl` by default.

Each non-dry-run result includes:

- `passes_shape_check`: submission contains a route portfolio, evidence labels,
  risk language, and next-experiment language
- `rubric.shape_pass`
- `rubric.route_diversity_pass`
- `rubric.evidence_grounding_pass`
- `rubric.validation_tool_pass`
- `rubric.risk_review_pass`
- `rubric.next_experiment_pass`
- `rubric.overall_pass`

## discovery_feedback.jsonl

Failure-driven replanning benchmark for `DiscoveryController v2`. Each case
starts with competing route beliefs and exposes a withheld deterministic
computational failure through `feedback_oracle`.

Dry-run validates the case without an LLM call:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-feedback --dry-run
```

Run the autonomous loop against the configured LLM:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-feedback --run
```

The benchmark passes only when all event-level checks pass:

- `failure_recorded`
- `failed_route_abandoned`
- `distinct_replan`
- `evidence_cited`
- `failed_action_not_repeated`
- `failure_evidence_cited`

`closed_loop_pass` is the conjunction of all six checks.

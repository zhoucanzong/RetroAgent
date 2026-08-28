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

## blind_studies.jsonl

Recent-study blind benchmark for testing inference beyond likely training data.
Each case keeps the publication identity, expected outcome, hidden measurements,
and scoring rubric outside the Agent prompt. The Agent receives only the initial
scientific problem and a list of measurements that it may actively request from
`blind_study_oracle`.

The first case uses a Nature paper published on 2026-07-15. It asks whether
styrene or a harder-to-reduce cyclopropyl ketone controls an
electrophotocatalytic reaction. The oracle withholds the published product
assignment and exposes only requested transient-absorption, electrochemical,
post-SET barrier, or structural-analogue measurements.

The second case uses a Nature Chemistry paper published on 2026-07-13. It asks
the Agent to distinguish a concerted amination/migration, a stepwise
aziridinium/backside-migration mechanism, and radical or ordinary
aziridination. Its oracle exposes reaction-progress IR, computed energy,
spectral-assignment, and stereochemical/reagent-compatibility measurements.
This case tests temporal ordering, intermediate identification, stereochemical
inference, and whether the Agent carries every observation needed to explain
self-limiting chemoselectivity into its conclusion.

Validate the case without an LLM call:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --dry-run
```

Run the strict offline blind loop:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --run
```

Run one case independently:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind \
  --run --case-id boronate-amination-2026-002
```

`blind_pass` requires all of the following:

- at least three distinct oracle measurements requested by the Agent
- the published major outcome identified without seeing the answer
- the complete forward-SET and post-SET/BET mechanism chain
- no contradictory major-product claim

Each case selects a hidden directional scoring profile. The photoreduction and
aziridinium profiles have separate negation, reversed-causality, and
counterfactual-falsifier checks so wording from one mechanism cannot satisfy
the other.

Runtime trajectories and results are written under `logs/`, which is ignored by
Git. The versioned case contains the source and hidden answer so the evaluator
can audit the result; restricted benchmark permissions prevent the Agent from
reading project files or using network tools during the run.

## Discovery Loop v3 trajectory scoring

Score any real Controller JSON or JSONL trajectory without another LLM call:

```bash
.venv/bin/python -m retroagent.run.retro score-discovery-v3 trajectory.json --require-pass
```

`discovery_v3_pass` requires all six event-level properties:

- at least two recorded competing hypotheses
- a rank-1 experiment with positive expected information gain
- normalized posterior updates that reduce entropy
- a route portfolio containing multi-source support
- a route passing the independent validation gate
- a final decision that selects that validated route and cites recorded evidence

The scorer evaluates the replayable trajectory, not keywords in the final answer.

# Second Strict Blind Paper Case Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a second strict blind paper case that tests temporal inference of a stepwise aziridinium/1,2-migration mechanism.

**Architecture:** Keep the existing hidden-measurement oracle and restricted environment. Add a named scoring profile to each blind case so the current photoreduction directional scorer and the new aziridinium directional scorer remain isolated, then add a CLI case filter for reproducible one-case real-model runs.

**Tech Stack:** Python 3, dataclasses, regular expressions, Typer, JSONL, `unittest`, existing `DiscoveryController` and `BlindStudyOracleTool`.

## Global Constraints

- Do not expose source identity, expected outcome, hidden results, or scoring rules in the Agent prompt.
- Require at least three distinct Agent-selected oracle measurements.
- Keep the strict four-tool blind allowlist and restricted execution permissions.
- Preserve the existing photoreduction case behavior and scorer.
- Update `benchmarks/README.md` and `word.md` after the real run.
- Do not commit or push until the user explicitly requests it.

---

### Task 1: Add Case-Specific Directional Scoring

**Files:**
- Modify: `retroagent/benchmark.py`
- Modify: `tests/test_blind_study_benchmark.py`

**Interfaces:**
- Consumes: `DiscoveryBlindCase.from_dict(data)` and `score_blind_conclusion(conclusion, case, queried_experiment_ids=...)`.
- Produces: `DiscoveryBlindCase.scoring_profile: str` and profile-dispatched outcome/mechanism/contradiction metrics with the existing return shape.

- [ ] **Step 1: Write failing profile tests**

Add tests that load two cases, preserve the photoreduction pass, accept a correct H2 aziridinium conclusion, and reject H1, H3, reversed step order, negated intermediate accumulation, and missing self-limiting explanation.

```python
score = score_blind_conclusion(
    "H2 is supported: rapid aziridinium formation precedes slower backside "
    "1,2-migration, giving anti product. DPPH is consumed before free amine "
    "accumulates, limiting overreaction.",
    aziridinium_case,
    queried_experiment_ids=[
        "reaction_progress_ir", "computed_energy_surface", "spectral_assignment"
    ],
)
self.assertTrue(score["blind_pass"])
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests.test_blind_study_benchmark -v`

Expected: failure because only one case/profile exists and the scorer assumes the photoreduction H2 mechanism.

- [ ] **Step 3: Implement minimal profile dispatch**

Add `scoring_profile` to the case dataclass, validate supported values, retain the current logic in a `photoreduction_post_set` helper, and add an `aziridinium_migration` helper that evaluates sentence-level directional assertions for:

```python
[
    h2_selected_and_not_rejected,
    rapid_aziridinium_before_slower_migration,
    transient_accumulates_and_is_spectrally_assigned,
    backside_migration_explains_anti_selectivity,
    dpph_consumed_before_amine_and_low_effective_concentration_limits_overreaction,
]
```

Record explicit contradictions for H1/H3 selection, migration-before-aziridinium, denial of the intermediate, syn/frontside migration, and DPPH persisting with free amine.

- [ ] **Step 4: Run focused tests and verify pass**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests.test_blind_study_benchmark -v`

Expected: all blind benchmark unit tests pass.

### Task 2: Add the 2026 Nature Chemistry Case and One-Case Runner

**Files:**
- Modify: `benchmarks/blind_studies.jsonl`
- Modify: `retroagent/run/retro.py`
- Modify: `tests/test_discovery_integration.py`

**Interfaces:**
- Consumes: JSONL case schema and `benchmark-discovery-blind` CLI.
- Produces: case `boronate-amination-2026-002` and optional `--case-id TEXT` filtering.

- [ ] **Step 1: Write failing loader, leak, oracle, and CLI-filter tests**

Assert two unique cases load, the new prompt omits DOI/title/results/rubric, oracle schema withholds `1195`, `17.2`, and `19.6`, and `--case-id boronate-amination-2026-002 --dry-run` writes exactly one result row. Assert an unknown ID exits with a concise parameter error.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests.test_blind_study_benchmark tests.test_discovery_integration -v`

Expected: failures for missing second case and unsupported `--case-id`.

- [ ] **Step 3: Add the hidden case record**

Store the frozen task, four measurements, expected H2 outcome, `aziridinium_migration` profile, contradiction phrases, minimum experiment count 3, and audit criteria. Keep numerical results only inside `experiments.*.result`.

- [ ] **Step 4: Implement the CLI filter**

Add:

```python
case_id: str | None = typer.Option(None, "--case-id", help="Run one blind case by id")
```

Filter after loading and raise `typer.BadParameter` when no case matches.

- [ ] **Step 5: Run focused tests and verify pass**

Run: `PYTHONPATH=. .venv/bin/python -m unittest tests.test_blind_study_benchmark tests.test_discovery_integration -v`

Expected: all focused tests pass.

### Task 3: Run, Audit, and Document the Second Blind Test

**Files:**
- Modify: `benchmarks/README.md`
- Modify: `word.md`
- Runtime only: `logs/discovery_blind_benchmark_boronate_*`

**Interfaces:**
- Consumes: the new case, scorer, and `--case-id` runner.
- Produces: ignored event ledger/result JSONL and versioned documentation of the verified outcome.

- [ ] **Step 1: Run full static verification**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. .venv/bin/python -m compileall -q retroagent tests
.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind --dry-run --output-dir /tmp/retroagent_blind_two_case_dryrun
git diff --check
```

Expected: complete suite passes, compile exits 0, dry-run reports two cases, and diff check exits 0.

- [ ] **Step 2: Run only the new case against the configured model**

Run:

```bash
.venv/bin/python -m retroagent.run.retro benchmark-discovery-blind \
  --run --case-id boronate-amination-2026-002 --max-cycles 16 \
  --output-dir logs/discovery_blind_benchmark_boronate_final
```

Expected: `Blind passed: 1/1` or a concrete evidence trail explaining a legitimate failure.

- [ ] **Step 3: Audit the run rather than trusting the aggregate score**

Check that at least three distinct oracle calls occurred; all execution tools are in the strict allowlist; the prompt lacks title, DOI, hidden numbers, expected outcome, and scoring profile; the conclusion preserves correct temporal direction and cites the collected evidence.

- [ ] **Step 4: Fix only benchmark or controller defects exposed by the run**

When a failure is a genuine reasoning failure, record it without weakening the scorer. When it is a protocol/scoring defect, add a failing regression test before the minimal fix and rerun the case independently.

- [ ] **Step 5: Update documentation and rerun verification**

Document the second case, actual result, limitations, commands, and trajectory path in `benchmarks/README.md` and `word.md`; rerun the full suite, compile check, dry-run, result assertions, prompt-leak audit, allowlist audit, and `git diff --check`.

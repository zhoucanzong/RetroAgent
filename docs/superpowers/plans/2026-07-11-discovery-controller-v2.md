# DiscoveryController v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a replayable autonomous computational chemistry loop that records negative feedback, propagates route failure, and replans through a materially different action.

**Architecture:** A typed append-only ledger projects into `ResearchState`; `DiscoveryController` asks for one structured decision per cycle and delegates authorization, execution, and evidence assimilation to focused services. Existing chemistry tools are adapted without changing retrosynthesis mode.

**Tech Stack:** Python 3.10+, dataclasses, JSON/JSONL, subprocess, pathlib, RDKit, unittest, existing RetroEnvironment and LLM client interfaces.

## Global Constraints

- Agent research choices remain autonomous; deterministic code owns state, provenance, permission, budgets, and validation.
- Standard mode requires approval for high-risk actions; full mode skips per-action approval only under an explicit scoped grant.
- Temporary code runs inside a per-run workspace and cannot modify project source by default.
- Retrosynthesis and design modes continue using `RetroPlanner` unchanged.
- No laboratory hardware, real quantum backend, dependency installer, or multi-agent orchestration in this release.

---

### Task 1: Event Ledger And Research State

**Files:**
- Create: `retroagent/research_state.py`
- Test: `tests/test_research_state.py`

**Interfaces:**
- Produces: `ResearchEvent`, `ResearchLedger.append()`, `ResearchLedger.replay()`, `ResearchState.apply()`, `ResearchState.to_observation()`.
- Consumes: JSON-compatible payloads and optional JSONL path.

- [x] Write failing tests for ordered replay, optional JSONL persistence, duplicate event rejection, hypotheses, evidence, route registration, and decision history.
- [x] Run `PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -p 'test_research_state.py' -v` and verify failures identify missing state classes.
- [x] Implement enums/dataclasses and deterministic event application with stable serialization.
- [x] Run the focused tests and verify all pass.

### Task 2: Feedback Assimilation And Route Beliefs

**Files:**
- Create: `retroagent/research_feedback.py`
- Modify: `retroagent/research_state.py`
- Test: `tests/test_research_feedback.py`

**Interfaces:**
- Consumes: `ResearchState`, execution result dictionaries, and affected ids.
- Produces: `EvidenceAssimilator.assimilate() -> list[ResearchEvent]`, `failure_events()`, `action_fingerprint()`.

- [x] Write failing tests in which route A and route B share no key reaction, negative feedback invalidates route A, repairable feedback challenges rather than invalidates, and unchanged failed action fingerprints are recognized.
- [x] Run the focused tests and confirm the failure propagation behavior is absent.
- [x] Implement evidence creation, failure impact propagation, multidimensional score updates, and stable action fingerprints.
- [x] Run both research-state and feedback test modules.

### Task 3: Permission Policy And Research Executor

**Files:**
- Create: `retroagent/research_execution.py`
- Test: `tests/test_research_execution.py`

**Interfaces:**
- Produces: `PermissionMode`, `PermissionGrant`, `PolicyDecision`, `PolicyEngine.authorize()`, `ResearchExecutor.execute_python()`, and `ResearchExecutor.execute_command()`.
- Consumes: structured actions, run workspace path, timeout, active grant, and optional registered-tool callback.

- [x] Write failing tests for restricted/standard/full decisions, expired or out-of-scope grants, project-path write rejection, bounded Python execution, stdout/stderr capture, and timeout reporting.
- [x] Run the focused test and verify the policy/executor classes are missing.
- [x] Implement conservative capability-based risk classification, resolved-path containment, subprocess timeout handling, and execution records.
- [x] Run the focused tests and compile the new module.

### Task 4: Autonomous Discovery Controller

**Files:**
- Create: `retroagent/agents/discovery_controller.py`
- Test: `tests/test_discovery_controller.py`

**Interfaces:**
- Consumes: model with `query(messages, tools=None)`, `RetroEnvironment`, `ResearchLedger`, `PermissionGrant`, max cycles, and run workspace.
- Produces: `DiscoveryController.run(task) -> dict` containing exit status, conclusion, evidence ids, state, and trajectory.

- [x] Write a scripted-model test for observe-decide-authorize-execute-assimilate-replan and a test that stop without evidence is rejected.
- [x] Add a failure scenario: first decision selects route A, oracle feedback fails its key reaction, second decision chooses route B, and unchanged route-A repetition is blocked.
- [x] Run the focused tests and verify the controller does not exist.
- [x] Implement decision parsing/validation, event recording, policy dispatch, environment adapter, feedback assimilation, repeated-failure guard, and evidence-backed completion.
- [x] Run all v2 focused tests.

### Task 5: CLI Integration And Closed-Loop Benchmark

**Files:**
- Modify: `retroagent/run/retro.py`
- Create: `benchmarks/discovery_feedback.jsonl`
- Modify: `retroagent/benchmark.py`
- Test: `tests/test_discovery_integration.py`
- Modify: `word.md`

**Interfaces:**
- Discovery mode constructs `DiscoveryController`; other modes construct `RetroPlanner`.
- Benchmark metrics include `failure_recorded`, `failed_route_abandoned`, `distinct_replan`, `evidence_cited`, and `closed_loop_pass`.

- [x] Write failing tests proving CLI controller selection, benchmark case validation, and deterministic closed-loop metrics.
- [x] Run the integration tests and confirm discovery still routes through `RetroPlanner`.
- [x] Add CLI options for permission mode, run workspace, ledger output, and optional full grant scope; connect discovery mode to v2.
- [x] Add the deterministic feedback case and benchmark analyzer.
- [x] Update `word.md` with architecture, permission semantics, benchmark definition, and exact validation commands.
- [x] Run focused integration tests, all unit tests, compileall, and benchmark dry-run.

### Task 6: Completion Audit

**Files:**
- Review all files above plus `.gitignore`.

**Interfaces:**
- Consumes: design requirements and current repository state.
- Produces: evidence that every scoped requirement is implemented or an explicit unresolved item.

- [x] Search for placeholders and accidental fixed discovery gates in the v2 path.
- [x] Confirm `.superpowers/`, logs, ledgers, and run workspaces are ignored.
- [x] Run the full unittest suite and `compileall` from a clean process.
- [x] Run the deterministic closed-loop benchmark and inspect its output fields.
- [x] Inspect `git diff --check`, `git status --short`, and the final diff before requesting commit approval.

# Skill Edition Verification (2026-10-22)

Release branch: `skill`. Backend version: `0.3.0`.
The host Agent owns research decisions; the backend does not query an LLM during
`research` or chemistry-tool commands. Legacy autonomous modes remain available.

## Automated Checks

- Full suite: 208 tests passed in both the development Python 3.14 environment
  and the isolated Python 3.11 chemistry environment.
- Skill Creator validator: valid frontmatter and structure.
- CLI helper: discovers 32 configured tool schemas without initializing weights.
- Wheel build and isolated `--target` installation: bundle and all four references
  plus helper exported successfully outside the checkout.
- `compileall` and `git diff --check` passed.

Tests include host/legacy result assimilation, reaction and route registration,
negative feedback and replanning, posterior/experiment events, atomic rejection
of malformed output, request deduplication, concurrent identical requests,
interrupted-request refusal, exact one-time approval consumption, evidence-linked
completion, condition placeholders, withdrawal provenance, and complete Skill export.

## Real Persistent MCP Inference

The pinned community Pistachio weights described in `retrochimera-cpu.md` were used
through the typed `retroagent_tool` endpoint. This run did not call an LLM API.

- Two distinct targets: `CCOC(=O)C` and `CC(=O)Nc1ccccc1`.
- `transformer_predict`, CPU, beam size 5, augmentation size 1, three results each.
- Four protocol responses (initialize, tool discovery, two calls), all parseable.
- One tokenizer/model initialization in the diagnostics across both calls.
- Two evidence records and six reaction records replayed from the project ledger.
- Session completed and released the environment at EOF.

Ignored local artifacts: `logs/skill-mcp-stdout.jsonl`,
`logs/skill-mcp-stderr.log`, `logs/skill-release-mcp-20261022/.retroagent/`.
This verifies integration and model retention, not route novelty or feasibility.

## Independent Host-Agent Exercise

An independent evaluator used only the Skill and backend interface for an offline
comparison of two ethyl acetate disconnections under moisture/corrosion constraints.
It created a study, recorded hypotheses, ran seven local tool observations, identified
missing condition/forward evidence, and saved a qualified conclusion. It made no
network, wet-lab, install, or LLM API calls and did not inspect implementation code.
Strict ledger validation passed for 43 events. No experimentally validated route
was claimed. This was a behavior test, not a held-out chemistry benchmark.

Observed problems and corrections:

- Agent withdrawal was attributed to the user: added `research withdraw`, recording
  `approval_withdrawn` with `actor=host_agent` and clearing only the pending request.
- Empty condition suggestions were ambiguous: added `needs-evidence` status and
  explicit unvalidated classification; placeholders cannot pass the condition gate.
- Arbitrary confidence values: generic observations now leave confidence unset.
- Reproducibility gap: execution evidence now records Python/backend and installed
  chemistry-package versions (installed packages are not a claim all were used).

The evaluator repeated these targeted checks in a fresh isolated project. It
confirmed the corrections, no unauthorized execution, and strict validation of
14 events. Remaining semantic distinction: a completed lookup can have
`success=true` while `assessment_status=needs-evidence`; the Skill requires reading
the substantive result, not treating transport success as chemical validation.

Local evaluation artifacts (not committed):
`/tmp/retroagent-skill-forward-20261022/` and
`/tmp/retroagent-skill-regression-20261022/`.

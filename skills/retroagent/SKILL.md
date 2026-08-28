---
name: retroagent
description: Run auditable, evidence-seeking chemistry research with stable resources, approvals, replay, and deterministic tools.
---

# RetroAgent

Use the `retroagent` CLI for chemistry research projects. Start with `retroagent help --json` and
`retroagent tool list --json`; do not guess command or tool parameters.

## Strategy

1. L1 observe: use `status`, `events`, `view`, `query`, `explain`, and `validate`.
2. L2 mutate structured research state: use `batch --dry-run` before `batch`.
3. L3 execute code, system commands, or network tools only through policy-controlled commands.
4. Address durable entities as `molecule:mol_0001`, `route:route_0001`, `reaction:rxn_0001`,
   `hypothesis:hyp_0001`, `evidence:evidence_0001`, and `experiment:exp_0001`.
5. After every mutation or execution, inspect the resulting events and validate the project.
6. Never claim that a physical experiment or external action was rolled back. CLI atomicity applies to ledger state only.

## Commands

- `project init`: Create a persistent .retroagent research workspace. Risk: `low`.
- `status`: Inspect project, run, entity, and pending-approval status. Risk: `low`.
- `agent run`: Start an autonomous evidence-seeking chemistry loop. Risk: `controlled`.
- `agent resume`: Resume a ledger-backed run, including an approved pending action. Risk: `controlled`.
- `agent stop`: Mark a project stopped without deleting its ledger. Risk: `low`.
- `approve`: Approve exactly one pending high-risk action. Risk: `high`.
- `reject`: Reject a pending action and require replanning. Risk: `low`.
- `events`: Read the append-only research event stream. Risk: `low`.
- `view`: Render project or stable resources as text, JSON, HTML, SVG, or PNG. Risk: `low`.
- `query`: Filter resources using deterministic field predicates. Risk: `low`.
- `explain`: Resolve an entity and show linked events and evidence. Risk: `low`.
- `validate`: Validate manifest, ledger replay, references, and approval state. Risk: `low`.
- `batch`: Atomically apply structured state operations; supports dry-run. Risk: `controlled`.
- `checkpoint create`: Snapshot the current event ledger. Risk: `low`.
- `checkpoint diff`: Compare current or checkpoint-projected research states. Risk: `low`.
- `checkpoint restore`: Restore state after creating an automatic backup checkpoint. Risk: `high`.
- `replay`: Rebuild deterministic research state from ledger events. Risk: `low`.
- `tool list`: List registered chemistry tools and JSON parameter schemas. Risk: `low`.
- `tool call`: Call a registered chemistry tool under project permissions. Risk: `policy`.
- `exec python`: Run temporary Python in the project run workspace. Risk: `high`.
- `exec command`: Run a system command in the project run workspace. Risk: `high`.
- `help`: Return this machine-readable command schema. Risk: `low`.
- `capability skill`: Export an agent Skill generated from this schema. Risk: `low`.
- `capability mcp`: Describe MCP tools generated from this schema. Risk: `low`.
- `mcp serve`: Expose the same CLI contract through MCP stdio. Risk: `policy`.

## Structured Output

Add `--json` for a stable envelope. Success returns `ok=true`; failures return `ok=false` with a
machine-readable error code. The append-only ledger is `.retroagent/ledger.jsonl`.

## Approval

When a command returns `E_APPROVAL_REQUIRED`, stop and request human approval. After `retroagent approve`,
use `retroagent agent resume`; approval applies only to the exact pending action unless the project was
explicitly initialized with full permission scopes.

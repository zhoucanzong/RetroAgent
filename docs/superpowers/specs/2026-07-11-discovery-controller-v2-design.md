# DiscoveryController v2 Design

## Purpose

RetroAgent discovery mode will become an autonomous computational chemistry research loop. The agent may define sub-goals, maintain competing route hypotheses, choose registered tools, author and execute temporary Python or shell analyses, assimilate external feedback, and replan after failure. Deterministic infrastructure owns state, provenance, permissions, budgets, and auditability.

The first release must prove one behavior: when a preferred reaction or route receives explicit negative feedback, the system records the failure, propagates it to every affected route, prevents an unchanged repeat, and lets the agent choose a materially different research action or route.

## Scope

This release includes:

- An append-only event ledger and replayable research state.
- Typed hypotheses, evidence, route beliefs, research decisions, execution records, and permission grants.
- Reaction failure propagation and multi-dimensional route assessment.
- An autonomous observe-decide-authorize-execute-assimilate-replan controller.
- Registered tool execution and temporary Python/shell execution in an isolated run directory.
- Standard, restricted, and full authorization modes.
- A deterministic feedback oracle and benchmark scenario for failure-driven replanning.
- Integration with the existing discovery CLI while preserving retrosynthesis behavior.

This release does not include laboratory hardware, real quantum-chemistry backends, automatic dependency installation, multi-agent role orchestration, or autonomous modification of RetroAgent source code. These remain adapter-level extensions.

## Architecture

`DiscoveryController` coordinates six stages without prescribing a fixed tool sequence:

1. `OBSERVE`: project the event ledger into a compact `ResearchState` view.
2. `DECIDE`: ask the model for one structured `ResearchDecision` with a research question, rationale, expected observation, falsification condition, cost estimate, and action.
3. `AUTHORIZE`: evaluate the action with `PolicyEngine` and the active `PermissionGrant`.
4. `EXECUTE`: dispatch to a registered chemistry tool or `ResearchExecutor` for temporary code and system commands.
5. `ASSIMILATE`: convert the result into evidence and state-changing events.
6. `REPLAN`: expose the updated state to the model; completion is allowed only when explicit evidence and stopping criteria are present.

The model owns research choices. Deterministic code owns whether an action is valid, permitted, affordable, novel relative to failed actions, and correctly recorded.

## State And Events

`ResearchLedger` stores ordered `ResearchEvent` objects. Each event has a stable id, event type, timestamp, cycle number, actor, payload, and optional causal parent id. JSONL persistence is optional; in-memory operation remains available for tests.

`ResearchState` is rebuilt by replay and contains:

- Goal, constraints, stopping criteria, and budget use.
- Hypotheses and their status, confidence, predictions, and falsification conditions.
- Evidence records linked to hypothesis, reaction, and route ids.
- Route beliefs with status and score dimensions: feasibility, novelty, evidence strength, selectivity risk, availability, safety, cost, and uncertainty.
- Pending approvals, research tasks, execution records, and the decision history.
- Failed action fingerprints used to reject unchanged repeats.

Event types include `goal_set`, `subgoal_set`, `hypothesis_proposed`, `route_registered`, `reaction_registered`, `decision_made`, `decision_rejected`, `approval_requested`, `execution_started`, `execution_finished`, `evidence_added`, `reaction_status_changed`, `route_status_changed`, `action_failed`, `replan_required`, and `research_completed`.

## Failure Propagation

Negative evidence names the affected reaction ids and a failure category. When `reaction_failed` is applied:

- The reaction is marked failed in domain state.
- Every route containing the reaction becomes `failed` unless the evidence explicitly marks the issue as repairable.
- Repairable failures set affected routes to `challenged` and lower feasibility/evidence strength while increasing uncertainty.
- The originating action fingerprint is added to failed fingerprints.
- A `replan_required` condition is exposed in the next observation.

The controller rejects an action whose fingerprint exactly matches a failed action unless the decision identifies new evidence or changed parameters.

## Agent Decision Contract

Each model turn returns one decision:

```json
{
  "research_question": "Which alternative disconnection avoids the failed coupling?",
  "action_type": "tool",
  "action": {"tool": "propose", "parameters": {}},
  "target_ids": ["reaction_1", "route_1"],
  "rationale": "The preferred route is invalidated by selectivity evidence.",
  "expected_observation": "At least one route with a distinct key bond construction.",
  "falsification_condition": "No chemically distinct disconnection is generated.",
  "estimated_cost": {"time_minutes": 2, "compute": "low"},
  "stop": false
}
```

`action_type=state` lets the agent explicitly propose a hypothesis or set a research sub-goal. These actions are auditable research organization, not evidence. Evidence is created only from a tool, computation, command, or feedback result.

Malformed decisions become recorded format failures and are returned for repair. A stop decision must include a conclusion and evidence ids; the controller never treats model confidence alone as completion.

## Execution And Permissions

`PolicyEngine` classifies actions as low or high risk. Registered local chemistry tools and internal state actions are low risk. Until an operating-system sandbox is added, all temporary Python and system-command execution is high risk. `code_execution` means arbitrary Python capability, including transitive filesystem/network effects; `system` means arbitrary shell capability. These scopes are intentionally broad and are not presented as composable fine-grained isolation. Network tools and structured memory writes use separate scopes.

Modes:

- `restricted`: only registered read-only tools.
- `standard`: low-risk actions run automatically; high-risk actions emit `approval_requested` and pause.
- `full`: high-risk actions run without per-action approval only when covered by explicit non-empty scopes for the current run. A grant may also carry an expiration timestamp; `*` must itself be supplied explicitly.

Full mode does not remove audit logging, estimated-time budget enforcement, cycle limits, per-action timeouts, working-directory checks, or timeout process termination. Authorization is user-supplied state; the agent cannot grant itself capabilities.

`ResearchExecutor` creates a per-run directory, writes temporary code there, runs commands with a timeout, captures stdout/stderr and produced files, and emits a complete execution record. Its working-directory checks are operational guardrails, not an operating-system security sandbox. Standard mode therefore pauses before code or commands; full mode must only be used with a trusted model and intentionally broad scopes.

## Integration

The existing `RetroPlanner` remains the implementation for retrosynthesis and design modes. Discovery mode is routed to `DiscoveryController`. Existing tools remain registered in `RetroEnvironment` and are called through an adapter. `RouteGraph` remains the chemical AND-OR graph; v2 research state stores route lifecycle and evidence separately to avoid breaking current route enumeration.

`chemical_critic`, `design_experiments`, `score_routes`, and `research_memory` become optional actions. Final validation checks evidence existence and relevance, rejects selected failed routes, and requires every explicit goal stopping criterion to be declared satisfied instead of enforcing the historical fixed ordering.

## Error Handling

- Invalid decision: record the error and request one corrected decision.
- Unauthorized action: record `approval_requested`; do not execute.
- Tool or command failure, malformed collection, or untrusted state event: record output as execution evidence and allow diagnostic replanning.
- Timeout: terminate the child process and retain partial logs as execution evidence.
- Unparseable tool output: preserve raw output and add low-confidence evidence only when an assimilator can identify a supported fact.
- Ledger corruption: fail closed with the last event index and do not continue from partial state.

## Verification

Unit tests cover event replay, idempotence, route failure propagation, permission decisions, path boundaries, timeouts, action fingerprints, and completion validation.

The deterministic closed-loop scenario begins with route A as the unique highest-feasibility route and route B as a distinct alternative. A route-bound feedback oracle invalidates a reaction that belongs to route A. A scripted model must observe the changed state and choose route B or generate a distinct route C. The benchmark fails if the final selection includes route A, the failure is not linked to its reaction, the replan is not distinct, the failed fingerprint executes again, or completion does not cite the failure evidence.

Full regression tests must preserve all current retrosynthesis and discovery tests.

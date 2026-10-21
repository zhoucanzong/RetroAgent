# State, Evidence, and Continuation

Research state lives under the task's `.retroagent/`, not in this Skill's directory.
Stable resources include `hypothesis:ID`, `reaction:ID`, `route:ID`,
`evidence:ID`, `experiment:ID`, and `molecule:ID`.

Use `research context` for compact state, `view evidence:ID --format json` for a
full observation, and `explain KIND:ID` for linked evidence and causal events.
`replay` reconstructs state; a checkpoint restores digital state, not lab actions.

Record hypotheses or user-supplied information using `batch` JSON. Example:

```json
[
  {"operation":"propose_hypothesis","payload":{
    "id":"H1","claim":"The transient is on the productive pathway.",
    "falsification_condition":"Product forms independently of transient decay."}},
  {"operation":"propose_hypothesis","payload":{
    "id":"H2","claim":"The transient is an off-cycle reservoir.",
    "falsification_condition":"Perturbing the transient changes the product rate directly."}}
]
```

Run `batch INPUT.json --dry-run --project PATH --json`, then apply it. Record
user-reported measurements with their provenance, units and uncertainty; do not
present them as locally executed tool measurements. Do not manually assert
validation success simply to satisfy a completion check.

Call a tool with `--parameters-file INPUT.json` for complex inputs. Use `--target H1`
and `--question TEXT` / `--rationale TEXT` to link the action to its purpose.
Results automatically register returned reactions/routes, evidence, experiment
selections, posterior updates, validation reports, and failure/replan events.
On failure, read the evidence before changing the plan.

For transport retries, supply a unique `--request-id ID` (or MCP `request_id`).
The identical completed action returns its recorded response without executing again.
Changing the action or targets with that ID is rejected. An interrupted action is
not retried automatically because it may already have produced external effects.

When authorization is required, obtain the user's approval through the existing
project/host flow. After `retroagent approve`, retry the exact same tool or execution
command; the backend consumes that approval once. `agent resume` is not needed for
the Skill workflow. Explicitly preauthorized scopes may proceed under their grant.
If you choose to abandon a pending action, use `research withdraw --reason TEXT`.
This records a host-Agent withdrawal. Reserve `reject` for an actual user denial.
Resolve or withdraw pending requests before marking a study complete.

An observation's `confidence` is unset unless separately calibrated. Runtime
versions identify installed software, not a guarantee of chemical correctness.
Condition output labelled `needs-evidence` contains no usable condition evidence;
`heuristic-suggestion` is an unvalidated rule-of-thumb, not an experiment.

## Finish

```json
{
  "conclusion":"The observations favor H1 under the tested conditions.",
  "target_ids":["H1"],
  "evidence_ids":["evidence_0001"],
  "uncertainty":"The intermediate identity remains indirectly assigned.",
  "falsification_condition":"A control uncouples intermediate decay from product formation.",
  "satisfied_criteria":[]
}
```

`research finish CONCLUSION.json --project PATH --json` validates references and
configured goal criteria before recording completion. It does not independently
verify the chemical truth of the prose. Cite actual IDs returned by your run; the
IDs above are illustrative. Leave unresolved studies active with a clear next step.

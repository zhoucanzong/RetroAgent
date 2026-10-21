# Evidence-Seeking Chemistry

For a mechanism question, distinguish structural identity, kinetic ordering,
selectivity, and conditions. An energy ordering alone does not establish product
selectivity; consider competing rates, reversibility, concentrations, and controls
when relevant. Prefer observations that discriminate alternatives.

For route discovery, specify the target, permitted starting materials, stereochemical
requirements, and practical constraints. Seek complementary disconnections, not
merely different reagent names for one strategy. Consider convergency, protecting
groups, chemoselectivity, precursor accessibility, and steps with weak precedent.
Distinguish precedented routes from analog-supported or speculative transformations.

## Selecting a Next Action

Choose among retrieval, structure checks, prediction, calculation, or a request
for experimental data based on the current bottleneck. Missing literature is not
negative evidence of feasibility. A negative result can invalidate a route's critical
step without disproving every mechanism or the target's synthesizability.

`select_experiment` ranks specified experiments under a likelihood model;
`update_hypotheses` performs Bayes updates from supplied likelihoods. They neither
run experiments nor estimate the likelihoods for you. Use quantitative ranking
only with defensible inputs; otherwise record a qualitative comparison and the
assumptions needed for a quantitative one. Do not invent observations to reduce
entropy or repeat the same observation as independent evidence.

`design_experiments` can propose controls and decision points from recorded issues.
An experimental plan remains a plan until an actual measurement is supplied.
Stop when the user's decision can be supported, or when further useful work needs
unavailable evidence. More calls and lower reported entropy are not goals themselves.

## Verification and Reporting

Check whether every decisive conclusion is supported by a recorded observation.
When one source contains multiple relevant findings, retain each needed finding.
For a route, examine mapping, forward-prediction evidence, conditions, availability,
and known failure modes as applicable. Computational agreement is correlated evidence,
especially for models trained on related datasets. Report contradictory evidence.

For a blind evaluation, answer keys and withheld observations must be outside the
Agent's accessible workspace and tools, served by an evaluator when appropriate.
A Skill instruction not to read an accessible answer is not isolation. Distinguish
blind evaluation from normal literature-assisted research.

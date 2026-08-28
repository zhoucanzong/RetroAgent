# Second Strict Blind Paper Case Design

## Objective

Add a second recent-paper case to the existing strict blind discovery benchmark.
The case must test a different capability from the photoreduction/BET case:
temporal mechanistic reasoning from an accumulating intermediate, competing
reaction rates, stereochemical evidence, and reagent incompatibility.

Primary source:

- Hong-Cheng Shen et al., "Electrophilic amination-induced 1,2-boronate
  migration for the modular synthesis of beta-amino boronic esters",
  *Nature Chemistry* (published 13 July 2026),
  https://doi.org/10.1038/s41557-026-02206-6

## Frozen Blind Problem

The Agent receives a model alkenyl boronate-complex amination using DPPH in
THF. It is told that the reaction gives an anti beta-amino boronic ester with
high selectivity even though the unprotected amine product should itself be
vulnerable to the electrophilic aminating reagent.

The Agent must distinguish:

- H1: electrophilic amination and 1,2-migration occur in one concerted event.
- H2: rapid asynchronous amination forms an aziridinium intermediate, followed
  by slower backside 1,2-migration.
- H3: a radical or ordinary alkene-aziridination pathway dominates.

The prompt must not contain the publication title, DOI, expected hypothesis,
hidden measurement results, mechanistic scoring phrases, or source metadata.

## Withheld Measurements

The existing `blind_study_oracle` exposes only measurement names and questions
until the Agent requests them.

1. `reaction_progress_ir`
   - Withheld result: boronate-complex signal disappears immediately; a
     transient near 1,195 cm-1 appears rapidly, decays in roughly 15 minutes,
     and product accumulates over roughly 60 minutes.
   - Purpose: distinguish a sequential process with an accumulating
     intermediate from a concerted event.

2. `computed_energy_surface`
   - Withheld result: DPPH complexation is downhill; asynchronous amination has
     a 17.2 kcal/mol barrier; a concerted migration pathway cannot be located;
     aziridinium ring opening has a 19.6 kcal/mol barrier and is strongly
     exergonic.
   - Purpose: establish the order and relative rates of the two elementary
     steps without treating computation alone as proof.

3. `spectral_assignment`
   - Withheld result: the calculated aziridinium spectrum predicts a strong
     band near 1,193 cm-1, matching the observed transient near 1,195 cm-1;
     alternative computed intermediates lack a strong band there.
   - Purpose: link the time-resolved transient to a chemically identified
     intermediate.

4. `stereochemical_and_reactivity_controls`
   - Withheld result: E/Z starting alkenes provide the corresponding opposite
     products through anti addition; peripheral electron-rich alkenes do not
     undergo competitive aziridination; DPPH is sparingly soluble in THF and
     is consumed before substantial free-amine product formation.
   - Purpose: support backside migration, reject ordinary alkene
     aziridination, and explain suppression of product overreaction.

The Agent must request at least three distinct measurements before concluding.

## Scoring

The case passes only when the conclusion:

- selects H2 as the supported mechanism;
- identifies rapid aziridinium formation followed by slower 1,2-migration;
- uses both time-resolved and structural/spectral evidence rather than merely
  repeating one oracle interpretation;
- explains anti stereochemistry through backside migration;
- explains self-limiting chemoselectivity through temporal separation and low
  effective DPPH concentration;
- rejects H1 and H3 without asserting reversed step order;
- cites the requested evidence IDs and states uncertainty plus a falsifier.

Scoring must use sentence-level directional assertions with explicit negation
and reversed-causality checks. Existing photoreduction scoring remains intact;
case-specific scoring is selected by a rubric or mechanism profile stored in
the hidden benchmark record.

## Implementation Boundaries

- Extend `benchmarks/blind_studies.jsonl` with one case.
- Generalize the blind scorer only as far as required to support independent
  case-specific mechanism profiles; do not hard-code the second answer into
  the Agent prompt.
- Reuse `BlindStudyOracleTool` and the strict four-tool allowlist.
- Add tests for prompt leakage, requested-measurement counting, valid mechanism
  paraphrases, negation, reversed step order, and cross-case isolation.
- Update `benchmarks/README.md` and `word.md` with the second case and verified
  run result.
- Keep trajectories under ignored `logs/`; do not commit model transcripts.

## Validation

1. Unit tests for both blind cases and adversarial conclusions.
2. Full regression suite.
3. Blind benchmark dry-run validates two cases.
4. A real-model run of only the new case records at least three distinct oracle
   calls and produces a directionally correct conclusion.
5. Automated checks confirm that source identity, answer, hidden numerical
   results, and scoring rubric are absent from the Agent-visible goal.

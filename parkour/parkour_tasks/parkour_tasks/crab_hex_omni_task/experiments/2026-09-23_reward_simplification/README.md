# 2026-09-23 — Omni reward simplification (candidate plan)

This is a **planning record**, not a running campaign. Nothing here trains.

The omni task's reward registers 37 names; 17–25 of them are active, depending on the phase. They accumulated one
campaign at a time. On 2026-09-23 the user asked for every term to be reviewed against the literature and the repo's own
evidence. The result is a list of independently plannable simplification candidates: removals, merges, replacements,
phase-outs, and fix-then-decide.

| file | what |
|---|---|
| [CANDIDATES.md](CANDIDATES.md) | Headline findings, the "plan a candidate" protocol, and 38 candidates in waves: W0 cleanup, W0b offline evidence and decision rule, W1 clock-independent screens, T teacher, C with/after the clock, X conditional. Also a per-term verdict for every reward name. |
| [CANDIDATES_DETAIL.md](CANDIDATES_DETAIL.md) | Long form of each candidate: evidence with file:line and ledger pointers, literature, hypothesis and falsifier, decision test, dependencies, risks, cost, and what the adversarial review changed. Also the full per-term table, ordering, dropped candidates, and numbered references. |
| [../lit-review-omni-reward-simplification.md](../lit-review-omni-reward-simplification.md) | The literature review behind the catalogue, in the house style. Citations were verified by opening each source. |

## How to use

Say: **"Plan the `<candidate name>` candidate from the omni reward simplification plan"**. That session then:
1. enters plan mode;
2. reads the candidate's section in both files;
3. checks the status of the candidate's dependencies: other candidates, the clock-deprecation session, M1b, ladder state;
4. re-verifies the cited evidence at the current HEAD;
5. plans only that candidate, with the standing discipline:
   - the offline replay gate first;
   - serial GPU use;
   - same-length fork controls;
   - never stacked with clock removal;
   - decided under `ni-adoption-rule` once it is ratified.

## Provenance

- **Workflow:** `wf_15be1644-c4a`, 2026-09-23, 92 agents, no errors. Its structure:
  - per-cluster repo dossier and literature sweep (opus·high);
  - reference verification (opus·medium, 23 agents);
  - per-cluster design (opus·xhigh);
  - merge (opus·xhigh);
  - refutation (opus·medium, 31 agents);
  - critic and 3 gap sweeps;
  - two writers (opus·xhigh);
  - an independent final check (fable·high).
- **Final check:** its catalogue corrections are applied in CANDIDATES_DETAIL.md and marked "(final check)"; its 18
  lit-review corrections are applied in the lit review.
- **Plan file:** `~/.claude/plans/memoized-puzzling-plum.md`.
- **Evidence base:**
  - config dumps: `parkour/outputs/omni_plan/dumps_motion_knobs_off/cfg_omni_*.json`;
  - per-term income: `Episode_Reward/*`, averaged over the last 10 % of iterations, from X08 t3→c6
    (`crab_hex_omni_flat_walk/2026-09-22_*_seed3_X08_*`) and omni teacher 2a/2b (`crab_hex_omni_teacher/2026-09-18_*`).

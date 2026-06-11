---
xid: STO-SCN-076
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: [STO-SCN-069, STO-SCN-070]
bd-id: krabby-wal
---

# Store schema update: additive structures for instances, run records, scores (non-breaking v3)

## Summary

The scene repository gains the structures Studio needs as **additive**
store-shape v3: where pipeline_instances live (E — settings without
data, today they have no home), the complete run record (073's
output), scores-on-runs (074), and the task-catalog reference — all
without moving, renaming, or breaking anything v2 tracks.

## Context

Store-shape v2 (STO-SCN-062/063) defined tracked = inputs + metadata
+ finals; the A–F taxonomy adds artifact kinds the store has no slot
for. Decision 6: store keeps `transform-NN-*` paths; new schemas say
"task." Every consumer of the store today (runoff scripts,
rate_renders, sync, fleet checkouts, gitignore rules) must keep
working unchanged.

## Problem

E (pipeline_instance) has no on-disk home — settings exist only
inside run dirs after the fact. Run records are scattered
(spec/results/labels). Scores live in scene-level rankings.jsonl
keyed by variant label, not attached to runs. Without defined,
tracked locations for these, 072–075 each invent their own.

## Design

- Proposed (final shapes ratified after the 069 spike picks the
  stack):
  - `scenes/_catalog/` or repo-level task-catalog reference (070's
    output) — single source, T-023
  - `scenes/<scene>/instances/<name>.json` — pipeline_instances (E)
  - `run-<r>/run_record.json` — 073's complete provenance record
  - `run-<r>/scores.json` (or rankings entries gaining run refs) —
    074's scores
- All new files are metadata → tracked under the existing
  `!**/*.json` v2 rule; verify no gitignore change is needed (and the
  pointmaps-style leak precedent: check sizes before assuming).
- v2 → v3 is purely additive: a v2 reader sees nothing new it must
  understand; `schema: 3` markers only in the new files.
- T-007: the schema proposal is presented to the operator before the
  first write lands in the store.

## Definition of Done

- [ ] Schema doc (shapes + locations + tracking rules) reviewed by
      operator before adoption.
- [ ] All existing tools run unchanged against a store containing the
      new files (runoff, rate_renders, render_comparison_matrix,
      sync/gather).
- [ ] Tracked-size delta measured and reported (T-016/T-017) — new
      metadata must not reintroduce a bulk-data leak.
- [ ] RECIPES.md § Storage policy updated to v3 (pointer, not copy).

## Out of scope

- Backfilling historical runs into the new shapes — STO-SCN-077.
- Renaming `transform-NN-*` or any existing path.

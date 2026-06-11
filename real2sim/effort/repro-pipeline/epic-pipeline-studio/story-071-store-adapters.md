---
xid: STO-SCN-071
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: [STO-SCN-069, STO-SCN-070]
bd-id: krabby-3cs
---

# Data-model adapters: A–F over existing store, non-breaking; historical runs browsable

## Summary

Read-side adapters that present the existing scene store
(`scenes/<scene>/pipeline-<p>/run-<r>/transform-NN-*/{specification,results}.json`,
render-variant runs, rankings.jsonl) through the A–F taxonomy, so
every historical run becomes a browsable pipeline_run with task_runs —
without moving or rewriting a single store file.

## Context

The past runoffs (006/007/008 matcha sweeps, DA3 branches, tetra
variants + operator rankings) are the existing experimental corpus —
exactly the data the Purpose says we're mining for the best
reproducible pipeline. Orphaning them would discard the evidence.

## Problem

Store layout encodes C/F implicitly (run dirs) and B/E partially
(spec JSONs) but has no A/D objects and no explicit instance-vs-run
split. Studio needs A–F; the store must not change (operator: don't
break what we have).

## Design

- Adapter maps: run dir → pipeline_run (F); transform dir →
  task_run (C); spec JSON → task_instance (B); the set of specs in a
  run → pipeline_instance (E); catalog (070) supplies A; the recipe
  trunk order supplies D.
- Render-variant runs (`run-<r>-tetra`: renders + run.json{source_run},
  zero transform dirs) map to derived pipeline_runs linked to their
  source run.
- Gaps (e.g. pre-v2 runs missing fields) surface as explicit
  `unknown`, never guessed (T-002).
- Write-side for NEW Studio-defined instances may use new files, but
  only additive ones the existing tools ignore.

## Definition of Done

- [ ] All existing runs across 001–013 enumerate as pipeline_runs
      with task_runs, settings, and attached rankings.
- [ ] Zero modifications to existing store files (verified by
      `git -C /var/krabby/scenes status` clean after a full scan).
- [ ] runoff/rate_renders/render_comparison_matrix still work
      untouched.
- [ ] Round-trip test: adapter output for one known run matches its
      JSONs field-for-field.

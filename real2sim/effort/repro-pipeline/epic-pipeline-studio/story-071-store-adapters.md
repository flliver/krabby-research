---
xid: STO-SCN-071
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: [STO-SCN-069, STO-SCN-070, STO-SCN-076]
bd-id: krabby-3cs
shipped: 2026-06-11
tasks: 4
complete: 4
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
- Write-side targets the v3 structures defined by STO-SCN-076 (this
  story does not invent file shapes — 076 owns the schema).

## Definition of Done

- [x] All existing runs across 001–013 (+dtu-bicycle) enumerate as
      pipeline_runs: **57 total**, incl. 8 render-variants linked to
      source runs and legacy runs surfacing task `unknown` (T-002).
      Leaderboard join reproduces the operator's 006 verdict (tetra
      #1, tetra1m tied #2, TSDF last).
- [x] Zero modifications to existing store files (`git -C
      /var/krabby/scenes status --porcelain` empty after full scan —
      module is read-only by construction).
- [x] runoff/rate_renders/render_comparison_matrix untouched (no
      shared code changed).
- [x] Round-trip test: matcha--8-strong adapter output matches
      spec/results field-for-field (settings, status, host,
      duration, image, renders).

## Implementation Notes

- `real2sim/studio_model.py` — A/D/E loaders (repo-side), C/F
  scanners (store-side, read-only), rankings read-time join +
  `leaderboard` (latest submission per view supersedes), CLI
  (`scan|run|leaderboard`, `--json` for the 072 UI).
- Render-variant runs (no transform dirs) classified by structure,
  linked via run.json `source_run`.
- v3 `run_record.json` is picked up when present (`record` field) —
  ready for 077 backfill output with zero changes.

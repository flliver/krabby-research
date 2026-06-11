---
xid: STO-SCN-072
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-11
depends-on: [STO-SCN-071]
bd-id: krabby-yhs
---

# Experiment UI: browse A–F, edit pipeline_instances (forms), settings diff

## Summary

The "define" surface of the experiment loop: browse tasks (A),
pipelines (D), instances (B/E), runs (C/F) distinctly; **edit a
composed pipeline_instance** via forms (settings constrained to
catalog min/max, defaults pre-filled); diff two instances' settings
side-by-side.

## Context

Operator decision 1: MVP = read-only DAG + A–F + run browsing PLUS
interactive editing of composed definitions. Drag-to-compose is
post-MVP. Purpose filter: the UI exists to shorten
define→run→rank→decide — the settings-diff is how you learn WHY one
configuration won.

## Problem

Today, defining a new experiment means hand-editing spec JSONs or
shell scripts with no range validation; comparing two configurations
means manually diffing JSONs across run dirs.

## Design

- Stack per spike decision (069). Mac-hosted (decision 4: jbeeprz is
  nice-to-have).
- Read-only DAG view of D/E/F (library-rendered, not draggable).
- Instance editor: clone an existing pipeline_instance → form per
  task_instance, fields generated from the catalog def, validated
  against min/max, defaults shown, deviations from default
  highlighted.
- Diff view: two instances → settings table, differing rows
  highlighted — the per-run settings sidecar work (STO-SCN-058) is
  the precedent.
- Saved instances persist via 071's additive write-side.

## Definition of Done

- [x] A/B/C/D/E/F each visibly distinct in the UI: lettered tags on
      every card; pipelines tab renders read-only DAGs (operator
      tasks marked 👤, optional nodes ?); runs tab shows F rows with
      C task_runs inside; legend in the header.
- [x] Edit flow: clone-pipeline-as-instance button prefills catalog
      defaults; non-default values highlighted; save POSTs through
      validation; store untouched (writes only repo-side
      `real2sim/instances/`).
- [x] Out-of-range input rejected: server-side 422 naming the bound
      ("n_images: 999 is greater than the maximum of 300"; da3
      conditional "process_res: 756 exceeds 504 when {'mode':'gs'}")
      + client-side HTML min/max on number fields. Found+fixed: with
      no jsonschema package the server silently skipped validation —
      added a stdlib fallback validator (type/range/enum/allOf
      ceilings) so validation can never silently degrade (T-003).
- [x] Settings diff renders (Diff tab, differing cells highlighted).
- [ ] **OPERATOR (T-020/T-026):** exercise the edit + diff flows at
      **http://localhost:8091/** (server running); story closes only
      on your verification.

## Implementation Notes

- `real2sim/studio/server.py` — stdlib http.server (rate_renders
  mold, no build step): GET tasks/pipelines/instances/runs/
  leaderboard, POST instances (validated). Read-only vs the store.
- `real2sim/studio/index.html` — vanilla JS single file: Pipelines
  (D, SVG topo-layout DAG), Tasks (A, settings tables with
  range/default), Instances (E, form editor w/ defaults + $var
  support), Runs (C/F + per-scene leaderboard joined from
  rankings), Diff (E vs E).

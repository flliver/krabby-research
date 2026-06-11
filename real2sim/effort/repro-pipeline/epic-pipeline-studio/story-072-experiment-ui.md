---
xid: STO-SCN-072
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
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

- [ ] A/B/C/D/E/F each visibly distinct in the UI (operator can point
      at the screen and name which letter they're looking at).
- [ ] Edit flow: clone matcha 8-strong instance, change one setting,
      save → new pipeline_instance, store untouched otherwise.
- [ ] Out-of-range input rejected at the form with the min/max shown.
- [ ] Settings diff of two real historical instances renders.
- [ ] T-020: operator exercises the edit + diff flows; story closes
      only on their verification.

---
xid: STO-SCN-069
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
date: 2026-06-11
depends-on: []
bd-id: krabby-joz
---

# Library spike: verify stack candidates on the experiment loop (define→run→record), pick + decision doc

## Summary

A hard-time-boxed spike that verifies the library-landscape table in
the epic (prior knowledge, T-002-flagged unverified) and picks the
stack, judged ONLY on the experiment loop:
define pipeline_instance → run reproducibly → record → rank.

## Context

EPI-SCN-PIPELINE-STUDIO § "Library landscape". Operator: don't
reinvent; langfuse scoped to scoring/run-capture (decision 5); MVP
ASAP and don't break the existing store/recipes. The Purpose section
is the rubric — generic-workflow features score zero.

## Problem

We have prior-knowledge claims about Dagster / Prefect / Kedro /
langfuse / React Flow but no verified facts (current versions,
licenses, config-schema min/max support, run-record fidelity,
embeddability next to rate_renders). Picking wrong here costs the
whole MVP timeline.

## Design

### Approach

1. Verify current state of each candidate (versions, license, the
   specific features the epic table claims).
2. Demo: ONE real task (e.g. `tetra_condition`) expressed as a task
   def with settings min/max/default, run via the candidate's
   machinery against a real scene store dir, run record captured.
3. Score each candidate against the loop: define / run-on-chosen-host
   / record-reproducibly / rank-attachable / Mac-runnable /
   non-breaking adjacency to the store.
4. Decision doc in this story's Implementation Notes; losers get a
   one-line reason each.

Throwaway code lives in a scratch dir, never in real2sim/ proper.

### Changes

| File | Change |
|------|--------|
| this story § Implementation Notes | decision doc |
| scratch demo (untracked) | discard after decision |

## Definition of Done

- [ ] Each epic-table claim marked verified/refuted with evidence.
- [ ] One real task demoed end-to-end on the chosen candidate.
- [ ] Stack decision recorded (orchestrator, UI, scoring layer) with
      license check (must be usable for contract work — note the DA3
      CC-BY-NC precedent).
- [ ] Time-box respected: if no clear winner inside the box, the
      decision defaults to "thin custom layer over our existing JSONs
      + React-Flow-class UI" and says so explicitly.

## Out of scope

- Any production code. The demo is throwaway by definition.
- Drag-to-compose evaluation beyond "does the UI lib exist and render
  a DAG" (post-MVP feature).

## Implementation Notes

_(Decision doc goes here.)_

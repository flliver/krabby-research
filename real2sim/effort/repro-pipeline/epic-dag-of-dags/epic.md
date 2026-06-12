---
xid: EPI-SCN-DAG-OF-DAGS
parent: ../design.md
kind: epic
effort: scn
status: in-progress
date: 2026-06-11
hugs: [HUG-SCN-005]
tenets: []
bd-id: krabby-d9c
---

# DAG of DAGs — content-addressed store v4, task/graph/job runtime, full migration

## Problem Statement

HUG-SCN-005 (operator direction, 10 locked decisions) replaces the
run-name store with content-addressed identities and the pipeline
model with recursive task/graph/job. This epic implements it: model
core, full-store migration (no legacy residue, locked #9), and the
verification stories the locks created.

## Stories

| XID | Story | Status |
|---|---|---|
| STO-SCN-079 | v4 model core (identity, refs, planner, scan, license ancestry) | shipped |
| STO-SCN-080 | Full-restructure migration of all 14 scenes | shipped (store committed) |
| STO-SCN-081 | Matcha monolith split verification | shipped — stage flags at pinned SHA; matcha@1 = executor change |
| STO-SCN-082 | orient-cameras method verification | shipped — sparse RANSAC REJECTED on measurement; bootstrap-mesh adopted |

## Outcomes

- Store-shape v4 live: 14 scenes content-addressed; identities
  computed from real specs (algo@0); operator rankings carried into
  scores.jsonl; v2 leaderboard verdicts reproduce.
- Consumers ported: Studio (:8091), rate_renders (:8090, URL contract
  preserved), studio_model, repro_check (metadata-as-record;
  migrated artifacts rank but cannot gate M11).
- Follow-on (first v4-native job drives them): the v4 executor
  (plan→dispatch), matcha@1 per-stage invocation,
  tetra_downsample_ratio promotion, fleet re-clone of the migrated
  store.

## Non-Goals (Out of Scope)

- Draggable graph composer (post-MVP per the Studio epic).
- Photo-spine completion (parked epic; spine byproducts intentionally
  remain under 005's input/spine-01).

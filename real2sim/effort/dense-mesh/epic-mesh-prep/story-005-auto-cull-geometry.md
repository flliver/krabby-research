---
xid: STO-SCN-005
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-roo
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.B2 — Auto-cull Out-of-bounds Geometry
---

# T1.B2 — Auto-cull Out-of-bounds Geometry

## Summary

Auto-remove distant, irrelevant geometry captured by the camera frustum. Solves the "background noise" problem from Phase A.

## Context

Auto-remove distant, irrelevant geometry captured by the camera frustum. Solves the "background noise" problem from Phase A.

Evidence: commit `cda1c29` (Phase B2: auto-cull out-of-scope geometry).

## Definition of Done

- [x] Out-of-frustum geometry removed
- [x] In-frustum scene geometry preserved
- [x] Tooling integrated into the post-processing pipeline


## Journal Notes

Implemented in `workspace/cull_mesh.py`: z-threshold + radial-distance cull removing below-floor outliers and unbounded distant-background pollution, currently per-scene-tuned. Retention observed: baseline 78%/65% vs lowres-15 90.5%/87.3%. A density-based or learned cull is noted as a generalization improvement (non-blocking).
_Sources: post-processing/entries 2026-05-01T144327-phase-b-pipeline-stood-up; entry 2026-05-01T144205-b6a-…._

---
_Imported from legacy beads `m11-1oc` (M11 DAG re-import, 2026-06-03)._

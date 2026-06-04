---
xid: STO-SCN-003
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-alk
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.A2 — Initial Tetrahedral Meshes for 3 Scenes
assignee: krabby
---

# T1.A2 — Initial Tetrahedral Meshes for 3 Scenes

## Summary

First end-to-end run of selected MAtCha pipeline on three real-world scenes: 001-patio, 003-firepit, 004-sky-house.

## Context

First end-to-end run of selected MAtCha pipeline on three real-world scenes: 001-patio, 003-firepit, 004-sky-house.

All 3 produced watertight tetrahedral meshes. The exit-criterion observation that motivated Phase B: every scene exhibited the same post-processing gaps (no ground plane, incorrect orientation, background noise from camera frustum). This drove the design of the Phase B post-processing pipeline.

Note: tetrahedral meshes were later superseded by TSDF for visual quality — the watertight requirement was met in both cases.

Evidence: commit `6ed424a` (Phase A complete: 3 watertight MAtCha meshes + post-processing pivot).

## Definition of Done

- [x] 3 scenes processed end-to-end
- [x] All meshes watertight (verified visually in MeshLab/Blender)
- [x] Post-processing gap inventory captured for Phase B


## Journal Notes

Three Phase-A tetra meshes (001 patio, 003 firepit, 004 sky-house-dining) from one recipe: 12 evenly-time-spaced keyframes @1024×576, `vitl` encoder, unposed SfM, ~11 min on RTX 5080. Jeremy's 2026-04-30 verdicts: 001 (4K hyperlapse, 155° fisheye) and 003 (4K@60fps fisheye) "chaotic but obviously the filmed scene… too much background noise (far things)"; 004 (2.7K@30fps semi-indoor) "dense in many areas, but obvious gaps." Per-scene records in `experiments/001-matcha-patio-fisheye/`, `003-matcha-firepit/`, `004-matcha-sky-house/`.
_Source: entry 2026-05-01T144135-phase-a-three-meshes-and-the-post-processing-pivot._

---
_Imported from legacy beads `m11-23n` (M11 DAG re-import, 2026-06-03)._

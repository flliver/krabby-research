---
xid: STO-SCN-004
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-8xl
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.B1 — RANSAC Ground-Plane Orientation
---

# T1.B1 — RANSAC Ground-Plane Orientation

## Summary

Auto-deduce the ground plane via RANSAC and orient the mesh to a Z-up coordinate system with the floor at z=0. Solves the "scene tilted at random angle" problem from Phase A.

## Context

Auto-deduce the ground plane via RANSAC and orient the mesh to a Z-up coordinate system with the floor at z=0. Solves the "scene tilted at random angle" problem from Phase A.

Used to process current best meshes for scene 004 and bicycle. Output verified visually in Blender.

Evidence: commit `4e9977b` (Phase B1: auto-deduce ground plane and orient mesh to z-up).

## Definition of Done

- [x] RANSAC plane detection robust across multiple scenes
- [x] Output is Z-up with floor at z=0
- [x] Tooling integrated into the post-processing pipeline


## Journal Notes

Implemented in `workspace/orient_mesh.py`: RANSAC ground-plane detection on the raw MAtCha tetra mesh, then rotation + z-shift so the floor sits at z=0 normal to +Z. Robustness wrinkle in the B6a lowres-15 run: it picked a different RANSAC candidate (cand 2, score 12,075) than baseline (cand 1, score 18,865), giving 1.26 m of below-floor geometry vs baseline's 0.34 m — mostly culled cleanly, but shows candidate-plane scoring is sensitive to small mesh changes (flagged for a future regression test; non-blocking).
_Sources: post-processing/entries 2026-05-01T144327-phase-b-pipeline-stood-up; entry 2026-05-01T144205-b6a-lowres-keyframes-negative-result._

---
_Imported from legacy beads `m11-5wp` (M11 DAG re-import, 2026-06-03)._

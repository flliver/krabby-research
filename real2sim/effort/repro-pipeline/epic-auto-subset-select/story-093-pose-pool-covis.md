---
xid: STO-SCN-093
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-13
depends-on: [STO-SCN-091, STO-SCN-092]
bd-id: krabby-0dk
---

# Pose the pre-culled pool → poses + co-visibility graph

## Summary

Solve camera poses for the pre-culled candidate set with a scalable, camera-model-correct
solver, and emit the **co-visibility / track graph** (which images see which 3D points,
pairwise overlap) that the selector consumes.

## Context

Step 2 of the pipeline. Depends on the camera model (STO-SCN-091) and the pre-cull
(STO-SCN-092). The co-visibility graph is the key output — it's what makes "best N"
a principled, automatable choice (design story conclusion #2).

## Problem

The solver must (a) use the right camera model, (b) scale to a few-hundred-frame
candidate set without the drift that killed the 300-frame MASt3R run, and (c) expose the
track graph. Solver choice depends on modality/overlap (conclusion #4): sequential/SfM
for ordered video, feed-forward for sparse — see the corpus solver landscape.

## Design

### Approach

Pick the solver by modality + the camera model from STO-SCN-091, run it on the candidate
set, and persist `poses + co-visibility graph`. Gate the result with the planarity /
sanity check (conclusion #5) before passing downstream. Candidate solvers: COLMAP-sequential
(video, needs deploy), GLOMAP/FastMap (GPU SfM), feed-forward (sparse). Exact solver
selection is the main open decision of this story.

### Changes

| File | Change |
|------|--------|
| pose stage | solver dispatch (modality + camera model) → poses + track graph |
| solve-validity gate | planarity / sanity ratio on solved cameras |

## Definition of Done

- [ ] Pre-culled pool → poses + co-visibility graph, using the profile's camera model.
- [ ] Passes the solve-validity gate (no nebula); fails loud otherwise.
- [ ] Track-graph output in a form the selector (STO-SCN-094) consumes.

## Spine note (longer-term — see STO-SCN-096 conclusion #7)

A full video is too big to pose at once. This story poses *one segment* (a tractable
window of the spine). The **global registration** of M segments into one cohesive
gauge — per-segment local solves + pose-graph optimization / loop closure / global BA —
lives in the sibling spine-assembly epic (EPI-SCN-SPINE-ASSEMBLY). Keep the per-segment
solver's output (poses + track graph) in a form that the global registration can consume
across segment boundaries.

## Out of scope

- The selection itself (STO-SCN-094).
- Global cross-segment registration / drift management (sibling epic).
- Deploying a specific solver binary is an implementation detail decided in this story.

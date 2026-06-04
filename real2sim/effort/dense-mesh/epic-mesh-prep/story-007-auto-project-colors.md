---
xid: STO-SCN-007
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-03
depends-on: []
bd-id: krabby-azm
priority: 2
shipped: 2026-06-03
tasks: 3
complete: 3
title: T1.B4 — Auto-project Vertex Colors
assignee: krabby
---

# T1.B4 — Auto-project Vertex Colors

## Summary

Project vertex colors from source frames onto the mesh. Critical for visual quality and required for any kind of meaningful A/B comparison rendering.

## Context

Project vertex colors from source frames onto the mesh. Critical for visual quality and required for any kind of meaningful A/B comparison rendering.

Evidence: commit `0467a10` (Phase B4: project source frames onto mesh as vertex colors).

## Definition of Done

- [x] Vertex colors visible and correctly registered to mesh
- [x] Multiple source frames considered per vertex (not single-view)
- [x] Tooling integrated into the post-processing pipeline


## Journal Notes

Implemented in `workspace/project_color.py`: projects vertex colors by sampling source frames at each vertex's projection in each visible camera, weighted-averaged by view confidence. Coverage: baseline 89.4% vs lowres-15 97.7% (median 6 vs 4 views/vertex). Known limit: averaging muddies specular surfaces; view-dependent shading or per-frame proxy textures could help (non-blocking).
_Sources: post-processing/entries 2026-05-01T144327-…; entry 2026-05-01T144205-b6a-…._

---
_Imported from legacy beads `m11-pbo` (M11 DAG re-import, 2026-06-03)._

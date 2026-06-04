---
xid: STO-SCN-013
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-19o
priority: 2
title: T2.D1 — Merge & Gap-Fill Mesh Surfaces
---

# T2.D1 — Merge & Gap-Fill Mesh Surfaces

## Summary

Use surface reconstruction (continuing TSDF fusion or Poisson on output) to resolve conflicts, merge nearby surfaces, fill remaining holes/gaps. Ensures the mesh is manifold prior to physics simulation.

## Context

Use surface reconstruction (continuing TSDF fusion or Poisson on output) to resolve conflicts, merge nearby surfaces, fill remaining holes/gaps. Ensures the mesh is manifold prior to physics simulation.

## Definition of Done

- [ ] Mesh manifold (no non-manifold edges/vertices)
- [ ] No visible holes in walkable surfaces
- [ ] Volume preserved (no significant shrinkage)


## Journal Notes

No M11 implementation (each M11 room fits a single MAtCha run), but the approach is specified in the M12+ submap-fusion design: after positioning N overlapping sub-scenes via the camera "spine," merge/gap-fill conflicting surfaces, then TSDF-fuse to make ground/obvious surfaces watertight. MAtCha's `extract_tsdf_mesh.py` already does multi-resolution TSDF fusion (reusable); Open3D `ScalableTSDFVolume` is the off-the-shelf driver; boundary artifacts handled by confidence-weighted depth integration (a confidence problem, not averaging).
_Sources: notes 2026-05-01T174650-submap-based-mesh-fusion, 2026-05-04T120000-submap-fusion-strategy-detailed._

---
_Imported from legacy beads `m11-21m` (M11 DAG re-import, 2026-06-03)._

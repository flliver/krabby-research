---
xid: STO-SCN-013
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: [STO-SCN-099]
bd-id: krabby-19o
priority: 2
title: T2.D1 — Merge & Gap-Fill Mesh Surfaces
assignee: krabby
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

No M11 implementation (each M11 room fits a single MAtCha run). This story is the
**post-fusion conditioning** step: it consumes the single cohesive mesh and makes it
manifold + watertight + gap-filled, ready for physics/USD export.

**Boundary with the spine epic (reconciled 2026-06-13):** the *inter-segment seam
fusion* — dedup of doubled walls / blending at segment overlaps after global
registration — is **owned upstream by STO-SCN-099** (EPI-SCN-SPINE-ASSEMBLY), whose DoD
explicitly emits geometry "consumable by downstream condition/export." Hence
`depends-on: STO-SCN-099`. This story does **not** re-do seam fusion; it conditions the
already-fused result. For a single space (M=1) STO-SCN-099 is a pass-through and this
story conditions the lone reconstruction directly.

Reusable tooling for the conditioning itself: MAtCha's `extract_tsdf_mesh.py`
(multi-resolution TSDF fusion); Open3D `ScalableTSDFVolume` as the off-the-shelf driver;
boundary artifacts handled by confidence-weighted depth integration (a confidence
problem, not averaging).
_Sources: notes 2026-05-01T174650-submap-based-mesh-fusion, 2026-05-04T120000-submap-fusion-strategy-detailed._

---
_Imported from legacy beads `m11-21m` (M11 DAG re-import, 2026-06-03)._

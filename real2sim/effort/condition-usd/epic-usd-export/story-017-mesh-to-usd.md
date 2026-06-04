---
xid: STO-SCN-017
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-1qp
priority: 0
title: T2.E2 — Mesh-to-USD via Isaac Lab MeshConverter
assignee: krabby
---

# T2.E2 — Mesh-to-USD via Isaac Lab MeshConverter

## Summary

Convert validated, watertight TSDF mesh to USD format. Likely two separate exports per scene:

## Context

Convert validated, watertight TSDF mesh to USD format. Likely two separate exports per scene:
- High-quality visual mesh (the TSDF)
- Simplified watertight collision proxy (V-HACD convex decomposition)

USD must carry: rigid body schema, mesh collider (or convex collider on the proxy), correct scale (from Phase E1), Z-up orientation, physics properties.

Tool: Isaac Lab's `MeshConverter`.

## Definition of Done

- [ ] USD file generated for ≥1 scene
- [ ] Loads in IsaacSim without errors
- [ ] Visual + collision meshes both present
- [ ] Scale + orientation + physics properties correct

---
_Imported from legacy beads `m11-uy5` (M11 DAG re-import, 2026-06-03)._

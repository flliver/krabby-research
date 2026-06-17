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

## Pipeline integration (v4 content-addressed store)

USD export is a **terminal task node** consuming the materialized, conditioned + scaled mesh — it
reads upstream nodes and writes a USD artifact; it mutates nothing upstream.

- **Task:** `tasks/export-usd.json` (**new**, `algo: usd-export@0`). Inputs: `visual_mesh`
  `from: condition` (the conditioned TSDF) plus a derived `collision_mesh` (V-HACD convex
  decomposition). Knobs (`vhacd_resolution`, `convex_hulls`, …) are `class: tunable` → distinct
  nodes.
- **Placement:** `{up_meshify_dir}/export/usd/{identity}/scene.usd` — a new export subtree off the
  conditioning chain. Not a renderable mesh, so `v4job.mesh_targets` ignores it.
- **Reuse materialized outputs:** consumes the already-conditioned + scaled `mesh.ply` (canonical
  gauge, Z-up, metric from STO-SCN-016); Isaac Lab `MeshConverter`; **NOOP** when the export
  identity exists.
- **Backwards-compat:** new additive terminal task + algo; no change to any upstream taskdef.
  Canonical rule: **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done

- [ ] USD file generated for ≥1 scene
- [ ] Loads in IsaacSim without errors
- [ ] Visual + collision meshes both present
- [ ] Scale + orientation + physics properties correct
- [ ] Implemented as a `usd-export@0` node consuming a materialized conditioned mesh (NOOP re-run;
      no upstream mutation); visual + V-HACD collision both emitted.

---
_Imported from legacy beads `m11-uy5` (M11 DAG re-import, 2026-06-03)._

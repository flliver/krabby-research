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

> **Split into two approaches (operator, 2026-06-15):** this goal is realized via
> **STO-SCN-142 — (A) screened Poisson** (★ PRIORITIZED, the chosen first approach) and
> **STO-SCN-143 — (B) TSDF re-fusion** (⏸ DEFERRED until A's results). This story is the umbrella
> intent; the implementation lives in 142 / 143.

## Context

Use surface reconstruction (continuing TSDF fusion or Poisson on output) to resolve conflicts, merge nearby surfaces, fill remaining holes/gaps. Ensures the mesh is manifold prior to physics simulation.

## Pipeline integration (v4 content-addressed store)

This is a **conditioning task** — it runs *downstream* of an already-materialized mesh and emits
a new content-addressed node; it never re-runs reconstruction or the GPU.

- **Task:** `tasks/merge-gapfill.json` (**new**, `algo: merge-gapfill@0`). Input `mesh`
  `from: meshify` — and equally an upstream `condition/*` node (e.g. a STO-SCN-136 cull), since
  condition nodes compose. Knobs (`poisson_depth`, `hole_max_edges`, …) are `class: tunable`, so
  each parameterization is a distinct store node.
- **Placement:** `{up_meshify_dir}/condition/{identity}` — the same conditioning subtree the
  existing `condition` task (`tetra-condition@0`) and the cull node (STO-SCN-136) use. It is
  auto-discovered + rendered by `v4job.mesh_targets` (which yields `meshify/*/*/condition/*/`), so
  the merged mesh becomes a rankable variant with **no renderer change**.
- **Reuse materialized outputs (T-013/T-016):** consumes the already-grounded `mesh.ply` from the
  upstream meshify/condition node (canonical gauge preserved); pure CPU on the gather host
  (Open3D `ScalableTSDFVolume` re-fusion / Poisson). `identity_hash({"mesh": <upstream id>},
  settings, "merge-gapfill@0")` → **re-run is NOOP** when the node exists; the raw + culled meshes
  stay alongside for comparison.
- **Seam fusion stays upstream:** inter-segment dedup/blending is the `spine-fuse` task
  (STO-SCN-099, shipped + materialized). For M=1 it is a pass-through; this task conditions the
  lone fused result (the existing `depends-on: STO-SCN-099`).
- **Backwards-compat:** a **new additive task + algo**; **never append keys to
  `meshify-via-tsdf`/`meshify-via-tetra`** (that re-keys every historical mesh). Canonical rule +
  mechanism: **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done

- [ ] Mesh manifold (no non-manifold edges/vertices)
- [ ] No visible holes in walkable surfaces
- [ ] Volume preserved (no significant shrinkage)
- [ ] Implemented as an additive `merge-gapfill@0` condition node consuming a materialized
      meshify/condition mesh (NOOP re-run; no GPU; canonical gauge preserved).
- [ ] Backwards-compat proven: the identity of an existing meshify node is unchanged after the
      task lands (additive node, not a re-key) — per the STO-SCN-136 store-identity rule.


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

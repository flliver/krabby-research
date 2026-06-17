---
xid: STO-SCN-014
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-25p
priority: 2
title: T2.D2 — Verify Watertightness (genus/manifold report)
assignee: krabby
---

# T2.D2 — Verify Watertightness (genus/manifold report)

## Summary

Genus/manifold report on processed TSDF mesh. Required for physics simulators (rigid body collision needs closed surfaces).

## Context

Genus/manifold report on processed TSDF mesh. Required for physics simulators (rigid body collision needs closed surfaces).

Concrete check: load mesh in Open3D or trimesh; verify `is_watertight() == True` and Euler characteristic is consistent with expected genus.

## Pipeline integration (v4 content-addressed store)

A **verification task** — it reads a materialized mesh and emits a *report* node; it produces no
new geometry and mutates nothing.

- **Task:** `tasks/verify-watertight.json` (**new**, `algo: verify-watertight@0`). Input `mesh`
  `from: meshify` (or any `condition/*` node). No `tunable` knobs (the check is deterministic);
  any thresholds are `class: frozen` so they enter the identity but stay constant.
- **Placement:** `{mesh_dir}/verify/{identity}/report.json` — a sibling report node. It is **not**
  a renderable mesh, so `v4job.mesh_targets` correctly ignores it.
- **Reuse materialized outputs:** loads the existing `mesh.ply` (Open3D/trimesh `is_watertight()`,
  Euler characteristic / genus, non-manifold edge + vertex counts); pure CPU.
  `identity_hash({"mesh": <id>}, {}, "verify-watertight@0")` → **NOOP** when the report exists.
- **Feedback loop (the "feeds back into D1" item):** a failing report is the signal to run
  STO-SCN-013 merge/gap-fill on the *same* upstream node — both are additive condition/verify
  nodes over one materialized mesh, so the loop never rebuilds geometry it already has.
- **Backwards-compat:** new additive verify task; never appended to a meshify taskdef. Canonical
  rule: **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done

- [ ] Watertightness check passes for ≥1 scene's final TSDF
- [ ] Report committed to journal
- [ ] If fails: feeds back into D1
- [ ] Implemented as a `verify-watertight@0` report node over a materialized mesh (NOOP re-run; no
      mutation of the mesh node; report is a sibling, not a renderable).


## Journal Notes

No dedicated verification tool in the journal. Grounding: MAtCha produces watertight meshes natively (the primary reason it beat the other five pipelines), so R1 is satisfied at the source; the M12+ submap-fusion plan uses TSDF fusion specifically to restore watertightness after merging sub-scenes. A standalone genus/manifold check still needs to be authored for this story.
_Sources: thread matcha-quality/thread.md; entry 2026-05-01T144135-phase-a-…; note 2026-05-04T120000-submap-fusion-strategy-detailed._

---
_Imported from legacy beads `m11-400` (M11 DAG re-import, 2026-06-03)._

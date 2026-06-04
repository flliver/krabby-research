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
---

# T2.D2 — Verify Watertightness (genus/manifold report)

## Summary

Genus/manifold report on processed TSDF mesh. Required for physics simulators (rigid body collision needs closed surfaces).

## Context

Genus/manifold report on processed TSDF mesh. Required for physics simulators (rigid body collision needs closed surfaces).

Concrete check: load mesh in Open3D or trimesh; verify `is_watertight() == True` and Euler characteristic is consistent with expected genus.

## Definition of Done

- [ ] Watertightness check passes for ≥1 scene's final TSDF
- [ ] Report committed to journal
- [ ] If fails: feeds back into D1


## Journal Notes

No dedicated verification tool in the journal. Grounding: MAtCha produces watertight meshes natively (the primary reason it beat the other five pipelines), so R1 is satisfied at the source; the M12+ submap-fusion plan uses TSDF fusion specifically to restore watertightness after merging sub-scenes. A standalone genus/manifold check still needs to be authored for this story.
_Sources: thread matcha-quality/thread.md; entry 2026-05-01T144135-phase-a-…; note 2026-05-04T120000-submap-fusion-strategy-detailed._

---
_Imported from legacy beads `m11-400` (M11 DAG re-import, 2026-06-03)._

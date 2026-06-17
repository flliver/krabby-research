---
xid: STO-SCN-018
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-7mh
priority: 0
title: T2.E3 — IsaacSim Load + Robot Spawn + Depth Sensor Returns
assignee: krabby
---

# T2.E3 — IsaacSim Load + Robot Spawn + Depth Sensor Returns

## Summary

End-to-end T2 closure: spawn a robot on each scene's mesh floor; verify depth-sensor returns are consistent with scene geometry.

## Context

End-to-end T2 closure: spawn a robot on each scene's mesh floor; verify depth-sensor returns are consistent with scene geometry.

Depends on E2.

## Pipeline integration (v4 content-addressed store)

End-to-end closure is a **verification node** over the materialized USD export — spawn + depth are
checked against a fixed USD artifact, and the verdict is recorded as a report node.

- **Task:** `tasks/verify-isaacsim.json` (**new**, `algo: verify-isaacsim@0`). Input `usd`
  `from: export-usd` (STO-SCN-017). Emits `report.json` (spawn-stability + depth-consistency).
- **Placement:** `{export_dir}/verify/{identity}/report.json` — a sibling report; not a renderable
  mesh, so `v4job.mesh_targets` ignores it.
- **Reuse materialized outputs:** loads the existing `scene.usd`; **NOOP** when the report exists.
  No re-export, no re-reconstruction.
- **Backwards-compat:** new additive verify task; reads, never mutates. Canonical rule:
  **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done

- [ ] Robot spawns on floor without falling through (collision works)
- [ ] Depth sensor returns match expected scene geometry (visual sanity check)
- [ ] Reproducible on ≥1 scene; ideally 2
- [ ] Implemented as a `verify-isaacsim@0` report node over a materialized USD export (NOOP re-run;
      no mutation).

---
_Imported from legacy beads `m11-dt8` (M11 DAG re-import, 2026-06-03)._

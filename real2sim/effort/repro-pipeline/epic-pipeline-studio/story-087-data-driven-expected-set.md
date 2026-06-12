---
xid: STO-SCN-087
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: []
bd-id: krabby-zh2
shipped: 2026-06-11
tasks: 3
complete: 3
---

# Data-driven expected set: gaps from the GRAPHS (planner), not just existing artifacts — da3/mesh-branch gaps visible with NC badges

## Summary

The expected set comes from the GRAPHS, not from what happens to
exist (operator, 2026-06-11: "should be DATA driven — buttons for
da3"). Gaps the graphs imply are now visible even when the artifact
tier is entirely absent.

## Shipped

- `v4core.expected_task_gaps(scene)`: whole-branch gaps (e.g.
  reconstruct-da3 never ran — settings prefilled from task-def
  defaults) + mesh-branch gaps on existing representations
  (matcha→tetra/tsdf, da3→tsdf).
- Payload: `task_gaps` beside `missing` (render tier).
- UI: purple dashed 🧬 tiles — task name, label, **NC badge derived
  from the task def BEFORE the artifact exists** (locked #10),
  honest-but-disabled: "GPU job — needs executor (STO-SCN-088) +
  operator host choice".
- Verified: 009 → da3 gap (NC); 006 → no gaps (complete); dtu → da3 +
  matcha--legacy tetra/tsdf branches.

## Definition of Done

- [x] Gaps computed from graphs/task defs, not artifact presence.
- [x] NC eligibility shown pre-materialization.
- [x] GPU gaps unclickable with the reason named (no pretending).

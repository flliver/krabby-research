---
xid: STO-SCN-015
parent: ./epic.md
kind: story
effort: scn
size: S
status: open
date: 2026-06-03
depends-on: []
bd-id: krabby-38w
priority: 3
title: T2.D3 — Final Taubin Smoothing Pass
assignee: krabby
---

# T2.D3 — Final Taubin Smoothing Pass

## Summary

Final smoothing pass on geometry. Taubin smoothing preferred over Laplacian to minimize surface shrinkage.

## Context

Final smoothing pass on geometry. Taubin smoothing preferred over Laplacian to minimize surface shrinkage.

## Pipeline integration (v4 content-addressed store)

The **`condition` task already exists** — `tasks/condition.json` (`algo: tetra-condition@0`)
already carries `taubin_iters` (tunable, default 10) alongside `target_tris`, and its executor is
`real2sim/tetra_condition.py` (decimate → volume-preserving Taubin → color transfer). This story
**wires that task as a first-class v4 condition node** (today `tetra_condition.py` is run by hand)
and validates the smoothing pass.

- **Task:** existing `condition` (`tetra-condition@0`); `taubin_iters` / `target_tris` are
  `class: tunable`, so each setting is a distinct node. If a *smoothing-only* pass (no decimation)
  is wanted, add it as a **new algo version** (`taubin-smooth@0`) — do **not** mutate
  `tetra-condition@0`.
- **Placement:** `{up_meshify_dir}/condition/{identity}` (already declared in `condition.json`);
  auto-discovered + rendered by `v4job.mesh_targets`.
- **Reuse materialized outputs:** consumes the already-grounded `mesh.ply` from the upstream
  meshify/condition node (canonical gauge preserved); CPU only. `identity_hash({"mesh": <up id>},
  {taubin_iters,…}, "tetra-condition@0")` → **NOOP** when present.
- **Chains after cull/merge:** smoothing typically runs on the output of STO-SCN-136 (cull) or
  STO-SCN-013 (merge) — a condition node may take another condition node as its input.
- **Backwards-compat:** the `condition` taskdef + its tunables already exist (no schema change);
  any *new* knob lives on a **new algo version**, never appended to `tetra-condition@0`. Canonical
  rule: **STO-SCN-136 § "Backwards compatibility — store identity"**.

## Definition of Done

- [ ] Smoothing applied without significant volume loss
- [ ] Visual quality preserved or improved
- [ ] Mesh remains watertight after smoothing
- [ ] Wired as the `tetra-condition@0` (or a new `taubin-smooth@0`) condition node consuming a
      materialized mesh (NOOP re-run; no GPU; gauge preserved); volume-loss measured.


## Journal Notes

Only forward-looking: the M12+ submap-fusion workflow ends with "a final smoothing pass using either Laplacian or Taubin smoothing… with a preference for Taubin to minimize shrinkage." No M11 implementation or parameters yet.
_Source: note 2026-05-04T120000-submap-fusion-strategy-detailed._

---
_Imported from legacy beads `m11-87v` (M11 DAG re-import, 2026-06-03)._

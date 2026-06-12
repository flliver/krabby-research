---
xid: STO-SCN-082
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
date: 2026-06-11
depends-on: []
hugs: [HUG-SCN-005]
bd-id: krabby-w1t
shipped: 2026-06-11
tasks: 3
complete: 3
---

# orient-cameras method verification: sparse-RANSAC vs bootstrap vs operator-assisted floor fit (locked #2 open item)

## Summary

Measured verification of the orient-cameras method (HUG-SCN-005
locked #2 open item): can the gauge be fixed from SPARSE data at
ingest time, before any mesh exists?

## Verdict (measured, 2026-06-11)

**floor-ransac-sparse REJECTED.** Against mesh-era ground truth
(the migrated bootstrap transforms):

| Scene | z-axis error | notes |
|---|---|---|
| 006-kubota | 58.1° (108° unconstrained) | locked onto the tractor side (normal = +X) |
| 003-firepit | 56.4° | |
| 004-sky-house | 166.5° | near-inverted |
| 008-kubota | 88.3° | |

Camera-up prior doesn't rescue it: the prior itself is unreliable —
portrait vs landscape captures flip which camera axis is world-up
(006 favors c2w col-z at 24.9°, 004 favors row-z at 4.8° — no single
convention fits).

**Adopted: bootstrap-mesh** (the validated STO-SCN-004 dense floor
fit from the first reconstruction, baked back onto primary's
cameras, once per solve) with **operator pick** as manual fallback.
Task def + HUG updated; `real2sim/orient_sparse.py` kept as the
rejected-method experiment record.

## Definition of Done

- [x] Method candidates tested against ground truth on real scenes.
- [x] Verdict recorded in the task def (`orient-cameras` default =
      bootstrap-mesh) and HUG-SCN-005 locked #2.
- [x] Experiment code preserved (orient_sparse.py verify/run modes).

- 2026-06-12 (STO-SCN-089 fallout): the bootstrap-mesh verdict stands,
  but @0's largest-plane floor fit rolled 009's corridor gauge 90 deg.
  Superseded by orient-floor@2 (horizon up prior — the camera-up idea
  this story rejected, reformulated pitch-immune as
  eigvec_min(sum X_i X_i^T) — + solve-framed bootstrap mesh as a
  resolved input). The 082 rejection of the *sparse* floor source
  remains valid; the rejection of the *up prior* was over-broad.

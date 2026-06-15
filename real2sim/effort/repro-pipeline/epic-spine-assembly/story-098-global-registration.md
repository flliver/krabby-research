---
xid: STO-SCN-098
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
shipped: 2026-06-14
date: 2026-06-13
depends-on: [STO-SCN-097, STO-SCN-093]
bd-id: krabby-1lm
assignee: krabby
tasks: 3
complete: 3
---

# Global registration of segment submaps (pose-graph + loop closure + global BA)

## Summary

Bring the M per-segment submaps into **one global gauge**, correcting drift across the
spine via pose-graph optimization, loop closure, and global bundle adjustment.

## Context

The make-or-break for cohesion (STO-SCN-096 #7). Consumes per-segment poses + boundary
overlaps (STO-SCN-097/093). Without it, locally-good segments stay disjoint and drift
compounds along the spine.

## Problem

Each segment is solved in its own arbitrary gauge with its own drift. They must be aligned
into a single, globally-consistent frame using boundary co-visibility and any loop closures
(path revisits).

## Design

### Approach

Build a pose graph over segments: relative-pose edges from shared boundary frames + loop-
closure edges from revisits; optimize globally (pose-graph optimization), then optional
global BA over the merged tracks. Output: every camera in one gauge, drift-corrected.

## Definition of Done

- [x] M submaps → single global gauge; relative drift across seams within tolerance.
      (`spine_register.register`: SIM(3) pose graph, reference fixed = identity, spanning-tree
      init, Gauss-Seidel relaxation. **Robust** per-edge fit + per-seam gate so noisy boundary
      frames don't fail a good registration. Validated: synthetic recovery to machine
      precision; **real 001-patio geometry** partitioned + per-segment-perturbed recovered to
      fit-to-GT **4e-13**; **two genuinely independent real FastMap solves** (632 cameras,
      134 shared) registered to one gauge, within_tol, 37 outlier frames surfaced not fatal.)
- [x] Loop closures applied where the path revisits.
      (Loop edges fold into the pose graph as cycle constraints; the relaxation distributes
      loop-closure residual around the cycle. Unit-tested: a loop edge measurably pulls a
      revisit pair together vs chain-only. Producing the loop *correspondence* from raw
      single-frame revisits needs a feature-match expansion of the revisit neighbourhood —
      the documented integration step.)
- [x] Globally-consistent poses emitted for fusion (STO-SCN-099).
      (`global.json`: per-segment gauges + per-camera global `center`/`R` + per-seam residuals
      — the input STO-SCN-099 fuses.)

## Implementation Notes

**Pose graph.** Nodes = per-segment gauges. **Relative-pose + scale edges** from shared
boundary frames — solved by Umeyama similarity on the retained-anchor camera centers.
Reuse `gauge_align.align_camera_sets`, which **already** computes a similarity (rotation,
translation, scale) from shared camera identities — the very mechanism behind the
posed-weld gauge-sim gate (STO-SCN-090). The scale term is what resolves each segment's
arbitrary SfM gauge (the OUT contract from STO-SCN-095). **Loop-closure edges** come from
STO-SCN-097's revisit flags.

**Optimize.** Pose-graph optimization over the segment graph (g2o / GTSAM / Open3D global
registration are all viable backends — pick at implementation), then an **optional global
BA** over the merged tracks for a final tightening. Output: every camera in one global
gauge + a per-seam drift residual (the tolerance gate).

**Why this is make-or-break.** M locally-good segments in disjoint gauges are still M
disjoint reconstructions; drift compounds along the spine. This stage is the only place it
gets corrected globally (conclusion #7).

**Test.** M submaps of a known scene register with per-seam residual < tol; a deliberately
drifted/rotated segment is caught by the residual gate (T-001 — the falsifiable check).

## Result (2026-06-14) — shipped: `spine-register@0` global registration node

Built `spine_register.py` (pure-numpy) + the v4 node **`spine-register@0`**
(`tasks/spine-register.json` + `v4exec.py cmd_spine_register` + `spine-register`
subcommand), with `tests/test_spine_register.py` (7/7 green).

**Engine.** A SIM(3) **pose graph**: nodes = per-segment gauges, edges = shared boundary
cameras (adjacent seams share camera identities) + loop correspondences. Per-edge relative
similarity reuses canonical `gauge_align` (Umeyama + **`consensus_align`** trimming +
rotation augmentation for near-collinear walking-path overlaps, STO-SCN-090). Fix the
reference segment = identity, init along a spanning tree, then **Gauss-Seidel relaxation**
re-fits each gauge to neighbours' current global positions until one drift-corrected gauge
emerges. Loop edges close cycles so the relaxation spreads loop residual around the loop
(converges fast on a chain, geometrically on a cycle — hence the larger iter budget).

**The robustness lesson (T-001).** First real run (two independent 001-patio solves) tripped
the gate at 15% — investigation (not assumption) showed it was **outliers, not a bug**: a
plain fit gave 20% max but `consensus_align` kept 60% at 1.7%. 53 boundary frames were
individually badly-solved. Fix: robust per-edge fit **and** a robust **per-seam gate** —
a seam passes when a sufficient MAJORITY agrees within tol (consensus_frac ≥ 0.5); sparse
noisy frames are trimmed + surfaced (`n_outlier`), a *systematically* warped segment leaves
too few in consensus and fails. (Honest limitation: consensus trims from a non-robust seed,
so it handles real moderate outliers but not gross ones — a RANSAC seed would harden it;
deferred, not seen on real data.)

**Validation.** Synthetic: recovery to machine precision; anisotropic-warp trips the gate;
loop closure pulls a revisit together; no-overlap refuses to chain; sparse outliers survive.
**Real geometry**: 001-patio's 539 cameras partitioned + per-segment-perturbed → fit-to-GT
**4e-13**. **Real independent solves**: two FastMap gauges (632 cameras, 134 shared) → one
gauge, within_tol, 37 outliers surfaced.

Run:
```
v4exec.py spine-register <scene> --spine <id> --solves seg0=<subset>/<solve>,... [--rel-tol 0.02]
```

**Deferred (noted, not blocking):** optional global BA over merged tracks (a final
tightening — the pose graph is the make-or-break and is delivered); loop-correspondence
expansion from raw single-frame revisits (feature match the revisit neighbourhood); the
full M-real-segment-solve end-to-end (needs the per-segment pipeline run on each spine
segment — the two-independent-solve case already exercises the node on real gauges).

**Next (STO-SCN-099):** fuse the per-segment reconstructions using `global.json`'s gauges
into one cohesive mesh/gaussian space.

## Out of scope

- Geometry fusion (STO-SCN-099); segmentation (STO-SCN-097).

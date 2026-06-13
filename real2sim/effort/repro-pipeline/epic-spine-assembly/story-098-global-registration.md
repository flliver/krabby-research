---
xid: STO-SCN-098
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-13
depends-on: [STO-SCN-097, STO-SCN-093]
bd-id: krabby-1lm
assignee: krabby
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

- [ ] M submaps → single global gauge; relative drift across seams within tolerance.
- [ ] Loop closures applied where the path revisits.
- [ ] Globally-consistent poses emitted for fusion (STO-SCN-099).

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

## Out of scope

- Geometry fusion (STO-SCN-099); segmentation (STO-SCN-097).

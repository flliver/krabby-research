---
xid: STO-SCN-048
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-qec
assignee: krabby
shipped: 2026-06-10
tasks: 7
complete: 7
---

# gauge_align — shared Umeyama/Procrustes module (extract from build_blender_scene/viewer)

## Summary

`real2sim/gauge_align.py` is the canonical similarity-transform solver
(scale + rotation + translation between two camera sets) with an
orientation-augmented two-pass solve and a residual hard gate —
correct even on coplanar camera layouts where plain Umeyama is
rotation-ambiguous.

## Context

Parent: [EPI-SCN-PHOTO-SPINE-PIPELINE](./epic.md). Stitching chunk
k+1 into the spine gauge is exactly the Umeyama alignment already
inlined twice (build_blender_scene.py anchor alignment,
camera_viewer/viewer.py). Per T-022/T-023 the extraction creates the
NEW canonical first; call-site consolidation of the two legacy inline
copies follows as a separate cleanup.

## Problem

The stitcher needs `align(src_cams, dst_cams) → (s, R, t)` with a
loud failure when the overlap cameras don't agree. Neither inline
copy is importable, and neither handles the degenerate geometry a
meadow capture actually produces: camera centers that are nearly
COPLANAR (a person walking a field holds the camera at one height),
for which position-only Umeyama leaves one rotation DOF unconstrained.

## Design

### Approach

- `umeyama(P, Q)` — closed-form similarity solve (SVD, det-corrected).
- `align_camera_sets(src, dst, max_residual, src_rotations,
  dst_rotations)` — two-pass solve: pass 1 positions-only to get the
  scale; pass 2 augments each camera center with synthetic points
  along its optical and up axes (offset `d_dst` = mean center
  spread in dst; `d_src = d_dst / s`) so camera ORIENTATIONS pin the
  rotation that coplanar centers cannot. RuntimeError when max
  residual exceeds the gate.
- `apply_to_cams2world(c2w, s, R, t)` — R applied to rotation blocks,
  full similarity to translations (scale must NOT touch rotations).
- `residuals()` — per-point distances for the report.

### Changes

| File | Change |
|------|--------|
| `real2sim/gauge_align.py` | add (canonical module) |
| synthetic split-solve test | exercised via batched_sfm stitch on a known-good pool split |

## Definition of Done

- [x] Coplanar-layout alignment recovers the true rotation (the case
      that motivated the two-pass design).
- [x] Synthetic round-trip: known-good solved pool split into
      overlapping halves, re-aligned — positions match to 4.6e-15 m,
      rotation element-diff 6.1e-16 (exact to float precision).
- [x] Residual gate raises RuntimeError, never returns silently bad
      transforms.
- [x] Self-reviewed.

## Testing

### Unit / fixture tests

- [x] Synthetic similarity round-trip (random s/R/t) — exact recovery.
- [x] Coplanar centers + orientation augmentation — rotation pinned.

### Integration

- [x] Used by `batched_sfm.py stitch` (STO-SCN-050) on real chunk
      overlaps.

## Out of scope

- Call-site consolidation in build_blender_scene.py /
  camera_viewer/viewer.py (T-022 round 2 / T-023 follow-up; behavior
  there unchanged).

## Implementation Notes

### What Changed

As designed, after one real discovery forced the two-pass shape (see
Gotchas).

### Files Modified

- `real2sim/gauge_align.py` — new module.

### Gotchas

- **Position-only Umeyama is rotation-ambiguous on coplanar/collinear
  camera centers.** The first synthetic stitch test showed a "2.55°
  rotation error" that was NOT a bug in the solve — it was the
  unconstrained DOF. Orientation augmentation (synthetic axis points)
  closes it.
- **MASt3R rotation matrices are ~1.16e-3 non-orthonormal.** Trace-angle
  rotation-error formulas fabricate ~2.5° of phantom error on them; use
  element-wise matrix diffs when validating against MASt3R output.

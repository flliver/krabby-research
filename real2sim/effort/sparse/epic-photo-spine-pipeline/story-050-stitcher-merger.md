---
xid: STO-SCN-050
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-10
depends-on: [STO-SCN-048, STO-SCN-049]
bd-id: krabby-kou
assignee: krabby
---

# Overlap stitcher + gauge merger with residual hard gates → unified pool cameras.json

## Summary

`batched_sfm.py stitch` chains every solved chunk into chunk-01's
gauge through the overlap cameras (basename-matched), hard-gates each
stitch on alignment residual, and emits one unified
`spine_cameras.json` (schema-5 pool shape) + `stitch_report.json` —
the single posed pool the rest of the toolchain already consumes.

## Context

Parent: [EPI-SCN-PHOTO-SPINE-PIPELINE](./epic.md). Consumes
STO-SCN-048 (gauge_align) and STO-SCN-049's per-chunk solves. The
output feeds the existing camera_viewer/curation path unchanged —
the spine is just a big pool.

## Problem

Eight independent solves live in eight arbitrary gauges (each chunk's
own scale/orientation/origin). They must be merged into ONE frame
without silently absorbing a bad alignment — a single corrupted
stitch poisons every downstream chunk in the chain.

## Design

### Approach

- Chunk 1 = reference gauge. For chunk k (2…N): find shared frames by
  BASENAME (the overlap contract from the chunker), require ≥3,
  align via `gauge_align.align_camera_sets` (orientation-augmented —
  positions + rotations of the shared cameras), reject if max
  residual > gate (default 0.10 m → `--max-residual`).
- Overlap frames keep their existing spine pose (first-solve wins);
  only new frames are added with mapped poses.
- Refuses to stitch a spine with unsolved holes (no partial spines).
- Outputs: `spine_cameras.json` `{filepaths, focals, cams2world}` +
  `stitch_report.json` (per-stitch shared count, scale, max/mean
  residual).

### Changes

| File | Change |
|------|--------|
| `real2sim/batched_sfm.py` | add `stitch` subcommand |

## Definition of Done

- [x] Synthetic acceptance (epic criterion 1): known-good solved pool
      split into overlapping halves, solved separately, stitched —
      merged poses match the single-solve poses to 4.6e-15 m /
      6.1e-16 rotation element-diff (exact).
- [x] Residual gate aborts the whole stitch loudly (RuntimeError →
      sys.exit with chunk id), never emits a partial spine.
- [x] Unsolved chunk → refusal with explicit chunk id.
- [x] Self-reviewed.
- [ ] First REAL 8-chunk stitch (005-meadow) — tracked in STO-SCN-051;
      the per-stitch residuals there are the production validation.

## Testing

### Unit / fixture tests

- [x] Synthetic split-solve-stitch round trip (exact match).
- [x] <3 shared frames → hard error.

### Integration

- [ ] 005-meadow 8-chunk stitch (STO-SCN-051, in flight).

## Out of scope

- Curation/selection over the spine (existing viewer does this).
- Loop closure / global bundle adjustment across chunks — the chain
  is linear by design; if chain drift shows up in practice, that's a
  new story.

## Implementation Notes

### What Changed

As designed. The "merger" of the original title is the
spine_cameras.json emission — no separate merge step was needed once
first-solve-wins handled overlap frames.

### Files Modified

- `real2sim/batched_sfm.py` — `stitch` subcommand (numpy +
  gauge_align imported lazily so chunk/solve still run on
  numpy-less fleet hosts).

### Gotchas

- The apparent "2.55° stitch error" during validation was two
  instrument errors, not a stitch error: position-only Umeyama's
  rotation ambiguity on coplanar centers (fixed in gauge_align) and
  trace-angle metrics lying about MASt3R's ~1.16e-3 non-orthonormal
  rotations (fixed by element-wise diffs). The stitch math is exact.

### Production redesign (2026-06-10, first real stitch)

The 005 8-chunk stitch broke the v1 design two ways and forced a
rewrite (commit 1bee8c8):

1. **Absolute residual gates are a unit error.** Each chunk's gauge
   has arbitrary scale (observed inter-chunk scale ratios 0.017–2.4);
   "0.10 m" means nothing. Gate is now `rel_tol × overlap spread`
   (default 2%).
2. **Real chunks contain badly-registered frames** (blurry/featureless
   1024×768 meadow shots) whose poses are simply wrong; full-overlap
   least squares is poisoned by them (01↔02 full mean residual ≈ 10%
   of scene span). `gauge_align.consensus_align` iteratively trims the
   worst frame until survivors pass the gate, with a hard consensus
   floor (≥6 frames AND ≥25% of overlap) below which the link is
   declared BROKEN. Trimmed frames are carried in the spine as
   `low_confidence`, not dropped.
3. **`--order`** sets explicit chain order + reference gauge — needed
   because 005's chunk-01 tail is internally bad and the spine must
   chain from chunk-02, attaching chunk-01 last via a bridge chunk.

Validated: chunks 02–08 stitched at consensus 40–90%, every gate
passed, 1,778 poses, 146 flagged low-confidence.

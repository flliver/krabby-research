---
xid: EPI-SCN-PHOTO-SPINE-PIPELINE
parent: ../design.md
kind: epic
effort: scn
status: deferred
date: 2026-06-10
hugs: []
tenets: []
bd-id: krabby-q9y
---

# Photo Spine Pipeline

## Problem Statement

MASt3R-SfM positions photos by RETRIEVAL-driven pairing in a single
global solve — no temporal prior, hard ceiling ≈300 frames/solve
(measured: 13.4 GB @300, OOM @400–500 on RTX 5080). Large self-similar
captures break it twice over: retrieval mismatches (grass pairs with
the wrong grass) corrupt the graph, and pool size exceeds the solve
ceiling. 005-meadow (2,028 photos) produced garbage poses
(2026-06-10). Operator direction: process temporally-close batches and
stitch them — a "photo spine" of chained gauges covering arbitrarily
large captures.

## Goals

- One unified pool `cameras.json` (schema-5 pool shape) for captures of
  ANY size, built from ≤300-frame temporal chunks with overlap
  stitching (Umeyama on shared cameras), per-stitch residual hard gates.
- Chunks solve independently → farmable across the fleet.
- 005-meadow's 2,028 photos fully posed; curation viewer over the
  unified spine.

## Stories

| # | XID | Story | Size | Status |
|---|-----|-------|------|--------|
| 1 | `STO-SCN-048` | gauge_align shared module | S | shipped 2026-06-10 |
| 2 | `STO-SCN-049` | chunker + per-chunk solve driver | M | shipped 2026-06-10 |
| 3 | `STO-SCN-050` | stitcher + merger w/ residual gates | M | shipped 2026-06-10 |
| 4 | `STO-SCN-051` | 005-meadow full spine + curation handoff | M | **deferred** 2026-06-10 (operator sidelined; pickup doc in scene) |

**EPIC DEFERRED 2026-06-10**: pipeline (048–050) fully shipped and
production-validated; the 005 application (051) is parked at 1,878/2,028
poses by operator direction. Pickup point:
`scenes/005-meadow/FINDINGS-photo-spine-2026-06-10.md` (session split
A/B/C + per-session-spine + retrieval-matched merge plan).

## Design

### Approach

`batched_sfm.py` (chunk/stitch/merge CLI) + `gauge_align.py`
(canonical Umeyama; T-023 consolidation target for the two existing
inline copies in build_blender_scene.py / camera_viewer/viewer.py —
extracted as the NEW canonical first, call-site consolidation follows
per T-022). Per-chunk solves reuse the existing container `--sfm_only`
invocation (same path as the 005 pool SfM run). Stitch chain: chunk 1
is the reference gauge; chunk k+1 maps into it via the overlap
cameras; max residual above threshold = loud failure, never silent
propagation.

### Key numbers

- chunk size ≤300 (measured solve ceiling), overlap default 50 (~17%
  — thin 20-frame overlap rejected: too few shared poses to survive
  outliers on self-similar terrain)
- 2,028 photos → **8 chunks** (right-aligned chunk math; early
  estimate said 9) → ~4.5 GPU-h serial, ~1.5 h farmed across t/b/d/s

## Success Criteria

- [x] Synthetic test: split a KNOWN-good solved pool into overlapping
      halves, solve separately, stitch — merged poses match the
      single-solve poses to 4.6e-15 m / 6.1e-16 rotation element-diff
      (exact; 2026-06-10).
- [ ] 005 spine: 8/8 chunks solved, 7/7 stitches under residual gate,
      2,028 poses in one cameras.json. _(7/8 solved as of 2026-06-10
      12:00; chunk-04 in flight on b.)_
- [ ] Operator curates from the unified spine in camera_viewer (T-020).

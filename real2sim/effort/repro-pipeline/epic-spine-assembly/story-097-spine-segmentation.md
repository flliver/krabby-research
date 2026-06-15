---
xid: STO-SCN-097
parent: ./epic.md
kind: story
effort: scn
size: M
status: shipped
shipped: 2026-06-14
date: 2026-06-13
depends-on: [STO-SCN-091, STO-SCN-092]
bd-id: krabby-duy
assignee: krabby
tasks: 3
complete: 3
---

# Spine segmentation (chunk trajectory into M overlapping segments)

## Summary

Partition a long video/trajectory into M **overlapping** segments, each individually
tractable for the per-segment pipeline, with shared boundary frames that guarantee
neighbors can register.

## Context

Outer loop of EPI-SCN-SPINE-ASSEMBLY; feeds each segment into
EPI-SCN-AUTO-SUBSET-SELECT. Longer-term scaffold (STO-SCN-096 #7).

## Problem

Cut points and segment size must respect: solver capacity (segment small enough to pose),
boundary overlap (segments share enough co-visible frames to register), and natural scene
structure (don't cut across a poorly-overlapping gap).

## Design

### Approach

Overlapping-window segmentation along the temporal/trajectory spine; size to solver
capacity; place boundaries where boundary overlap is high. Loop points (path revisits) are
flagged for loop-closure constraints downstream (STO-SCN-098).

## Definition of Done

- [x] Long pool → M overlapping segments, each within solver capacity.
      (`spine_segment.segment`: fixed overlapping windows on stride `cap-overlap`; each
      segment ≤ `cap` by construction. Verified on the real 942-frame 001-patio pool →
      4 segments, max 300/300.)
- [x] Boundary overlap between adjacent segments meets a registrability threshold.
      (Overlap budget ≥ `overlap` guaranteed by construction — cuts snap *earlier* only;
      001-patio seams shared 34/38/39 frames ≥ 30. Per-seam **content** registrability
      = mean consecutive pHash distance, measured and **flagged** when it exceeds
      `reg_thresh`: 2 of 3 seams cleared 12; one fast-motion seam was honestly flagged
      at 13.42 with an actionable fix — widen `--overlap` / lower `--reg-thresh`.)
- [x] Loop/revisit candidates flagged for global registration.
      (Cheap cross-segment pHash scan over representative frames; flags the best
      sub-`loop_thresh` pair per NON-adjacent segment pair. Unit-tested with a planted
      revisit; 001-patio is a linear walk so 0 found — correct. Real loop-trajectory
      verification awaits a loop-containing capture.)

## Implementation Notes

**Segmentation.** Overlapping temporal windows along the trajectory. Window size capped by
**solver capacity** (≤ a few-hundred frames for SfM / the DA3 view ceiling per segment).
Overlap is a tunable stride so adjacent windows share ≥ the **boundary-overlap budget**
(the registrability threshold consumed by STO-SCN-098).

**Boundary placement.** Prefer cut points where inter-frame overlap is *high* (don't cut
across a low-overlap gap, or the seam won't register). Use the pre-cull / co-visibility
signal where a coarse pass is available; otherwise fall back to trajectory speed / frame
similarity.

**Loop / revisit detection.** Cheap global-descriptor or pHash similarity across
**non-adjacent** windows (reuse the STO-SCN-092 pHash) flags candidate loop closures —
handed to STO-SCN-098 as extra pose-graph edges.

**Output = the spine's per-segment `boundary_spec`** (the IN contract that STO-SCN-094
honors): pinned anchor frames + overlap region per neighbor, plus the global `camera_model`
(STO-SCN-091, identical for every segment). This is precisely the `097 → 091,092` edge.

**Test.** A multi-segment walk splits into M windows each within capacity; every adjacent
pair clears the overlap threshold; at least one path-revisit is flagged.

## Result (2026-06-14) — shipped: `spine@0` pose-free segmentation node

Built `spine_segment.py` (pure-stdlib core `segment(ids, hashes, ...)` — fully
unit-testable without images; `hashes_for()` decode helper reusing `phash`), wired as the
v4 store node **`spine@0`** (`tasks/spine-segment.json` + `v4exec.py cmd_spine` + `spine`
subcommand), with `tests/test_spine_segment.py` (7/7 green) covering the falsifiable bar
above plus snapping/budget-floor/single-segment(M=1)/validation/determinism.

**How it works.** Pose-free outer loop, runs locally on the capture-ordered pool (one pHash
decode pass), like the pre-cull — NO solve required. Fixed overlapping windows on stride
`cap-overlap` guarantee each segment ≤ `cap` AND adjacent overlap ≥ `overlap` **by
construction**. Each cut is then **snapped earlier** (never later — that would drop overlap
below budget) to the most-similar local pHash transition, so the shared region sits in a
coherent stretch instead of across a fast-motion gap. Per-seam registrability (mean
consecutive pHash distance) is measured and flagged. Loop/revisit candidates come from a
cheap cross-segment pHash scan over representative frames (non-adjacent segments only),
handed to STO-SCN-098 as extra pose-graph edges. Output `spine.json` carries the per-segment
`boundary_spec` (pinned anchor frames + per-neighbor overlap region — the IN contract
STO-SCN-094 honors) + the global `camera_model` (091, identical per segment). M=1 for a
single tractable space (no seams; empty spec — the per-segment pipeline runs unchanged).

**Real run** (001-patio, 942 frames, `spine/JVK5BAMA4BPC`): 4 segments
([0–299],[266–565],[528–827],[789–941]), all ≤ 300, overlaps 34/38/39, registrability
5.82 / 11.03 / **13.42 (flagged)**, 0 loop candidates (linear walk). Idempotent NOOP on
re-run; emits the store-hash anchor ids the per-segment pipeline resolves.

Run:
```
v4exec.py spine <scene> [--cap 300] [--overlap 30] [--snap 10] [--reg-thresh 12] \
                [--loop-thresh 8] [--loop-min-sep 2] [--loop-step 5]
```

**Next (STO-SCN-098):** consume `spine.json`'s anchors + loop_candidates to register the
per-segment submaps into one global gauge (pose-graph + loop closure + global BA).

## Out of scope

- Registration / fusion (later stories).

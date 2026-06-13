---
xid: STO-SCN-092
parent: ./epic.md
kind: story
effort: scn
size: M
status: in-progress
date: 2026-06-13
depends-on: []
bd-id: krabby-e28
assignee: krabby
---

# Pose-free pre-cull (sharpness + perceptual-dedup) for large pools

## Summary

Before paying for a pose solve, cheaply shrink a massive pool (thousands of frames) to a
tractable candidate set by dropping motion-blurred frames and near-duplicates — no poses
required.

## Context

Conclusion (design story): solving the full pool is the expensive/unstable step; cull
first. We already have `select_sharp_frames.py` (Laplacian-variance sharpness) and
perceptual-hash dedup in `camera_viewer`.

## Problem

A 30 fps hyperlapse has hundreds of redundant, sometimes-blurry frames. Feeding them all
to the solver is wasteful and harms quality. We need a fast, pose-free filter: 5000 → a
few hundred good candidates.

## Design

### Approach

Two pose-free filters composed: (1) sharpness gate (variance-of-Laplacian; drop blurred),
(2) near-duplicate removal (perceptual hash / frame similarity). Tunable target count.
Reuse `select_sharp_frames.py` + the `camera_viewer` pHash.

### Changes

| File | Change |
|------|--------|
| pre-cull stage | compose sharpness + pHash dedup → candidate set |
| `select_sharp_frames.py` | reuse/extend |

## Definition of Done

- [x] Large pool → tractable candidate set, pose-free, in seconds. (`precull_frames.py`
      engine + CLI; CPU-only, numpy+PIL.)
- [x] Blurred + near-duplicate frames demonstrably removed; tunable target N. (median-
      relative blur gate + local pHash dedup; `--target`, default 300 = solve ceiling.)
- [x] Sharp, well-distributed candidates retained (no big temporal gaps). (`--max-gap`
      guard re-inserts the sharpest frame in oversized gaps.)
- [x] Verified: 22 tests (venv) / 11+3-skip (system); CLI smoke 68→6. ⏳ Real large-pool
      ingest is the operator-verification gate (T-020).

## Implementation Notes (as built, 2026-06-13)

**Shared dependency-free pHash** (`real2sim/phash.py`). DCT pHash (32x32 → DCT → 8x8 →
median threshold → 64 bits) using only numpy+PIL — replaces the `imagehash` dependency.
`camera_viewer/clustering.py` was refactored to use it too (single source; `imagehash`
removed from its `requirements.txt`). The "shared-phash.py" extraction was done as part of
this story (operator direction).

**Sharpness** reuses `select_sharp_frames.sharpness_of_gray()` (Laplacian variance), split
out so the precull scores sharpness + pHash in a **single image decode**.

**Engine** (`precull_frames.py`): score → **local** pHash dedup (consecutive runs within a
`dup_window`, keep sharpest) → **median-relative blur gate** (`blur_rel`×median; not a
percentile) → optional windowed-sharpest **target** thin → **gap guard** (bound max
temporal spacing). Pure `precull(items=(id,path))` core serves both the CLI (ids =
filenames) and the v4 wiring (ids = image hashes).

**Two corrections earned (T-001):**
- **Local (windowed) dedup, not global** — global pHash dedup would delete path *revisits*,
  which are loop-closure / co-visibility signal for STO-SCN-098, not redundancy.
- **Median-relative blur gate, not percentile** — a percentile lands *inside* a minority
  blur cluster and lets blur through (a test caught it).

**v4 wiring (option i — opt-in, non-breaking).** `tasks/precull-subset.json` + `cmd_precull`
+ an **optional** `precull` node in `ingest-scene.json` as a side-branch off `pool` (not in
the default solve path). `v4exec.py precull <scene> [params]` writes a curated subset
(`mechanism: precull`) set-if-unset; **leaves `primary` untouched** unless `--set-primary`
(a deliberate operator act — locked #1 forbids silent ref moves). STO-SCN-093 poses this
subset.

**Test.** Synthetic pools cover near-dup collapse, blur rejection, **revisit preservation**,
gap guard, small-pool passthrough, and the store wiring (subset written, primary respected).

## Out of scope

- Coverage-aware selection — that needs poses (STO-SCN-094).

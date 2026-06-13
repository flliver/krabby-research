---
xid: STO-SCN-092
parent: ./epic.md
kind: story
effort: scn
size: M
status: open
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

- [ ] Large pool → tractable candidate set, pose-free, in seconds.
- [ ] Blurred + near-duplicate frames demonstrably removed; tunable target N.
- [ ] Sharp, well-distributed candidates retained (no big temporal gaps).

## Implementation Notes

**Compose two existing filters.** (1) **Sharpness gate** — reuse `select_sharp_frames.py`
(variance-of-Laplacian); keep frames above a percentile threshold rather than an absolute
cutoff so it adapts to a scene's overall focus. (2) **Near-duplicate removal** — reuse the
`camera_viewer` perceptual hash; cluster by Hamming distance on the 64-bit pHash and keep
the **sharpest** frame per cluster (the two filters cooperate — dedup picks the survivor by
the sharpness score).

**Order.** pHash-cluster first to collapse redundancy, then apply the sharpness percentile
within/across clusters — cheaper than scoring every frame's sharpness when many are
near-identical.

**Coverage guard.** Enforce a **max temporal-index spacing** between retained frames so
aggressive dedup can't open a coverage hole — this preserves the sequential overlap that
the pose solve (093) and its connectivity needs. Tunable target N (~300–500 from a
multi-thousand pool, matching the historical 5000→300 step).

**Cost.** Pure CPU, seconds, deterministic given thresholds — runs *before* any GPU solve,
which is the whole point (don't pay solve cost on blurred/duplicate frames).

**Test.** A real hyperlapse pool: assert blurred frames dropped, near-duplicate runs
collapsed to one, no retained gap larger than the spacing bound, and target-N honored.

## Out of scope

- Coverage-aware selection — that needs poses (STO-SCN-094).

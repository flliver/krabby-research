---
xid: STO-SCN-092
parent: ./epic.md
kind: story
effort: scn
size: M
status: draft
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

## Out of scope

- Coverage-aware selection — that needs poses (STO-SCN-094).

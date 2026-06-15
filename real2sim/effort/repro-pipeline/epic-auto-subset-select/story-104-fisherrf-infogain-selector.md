---
xid: STO-SCN-104
parent: ./epic.md
kind: story
effort: scn
size: L
status: deferred
date: 2026-06-14
depends-on: [STO-SCN-103]
bd-id: krabby-0c1
assignee: krabby
---

# FisherRF information-gain view selector (GPU, model-aware) — RESERVE

> **DEFERRED / reserve (operator decision 2026-06-14).** Ship the geometric
> voxel-coverage selector (STO-SCN-103) first; graduate to this only if
> coverage-greedy underperforms or we want task-aligned selection + uncertainty maps.

## Summary

A GPU view selector that ranks views by **Expected Information Gain on the 3DGS
parameters** (Fisher information) — i.e. "which view most reduces uncertainty in the
reconstruction we'll actually build" — rather than a geometric coverage proxy. Reuses the
DA3 scout as the radiance field. Produces per-region **uncertainty maps** as a byproduct.

## Context

The deeper answer to the epic's selection problem. Where STO-SCN-103 optimizes a geometric
proxy (surface coverage from good angles), FisherRF optimizes the **downstream objective
directly** (final splat quality) — model-aware, occlusion-handled-for-free (the field's
differentiable rendering encodes visibility), and appearance-sensitive (it values
information-rich views, not just well-faced geometry).

- *FisherRF: Active View Selection and Uncertainty Quantification for Radiance Fields using
  Fisher Information* — ECCV 2024 (oral). [paper](https://arxiv.org/abs/2311.17874) ·
  [project + code](https://jiangwenpl.github.io/FisherRF/). ~7 s/selection, ~4 GB on an
  L40-class GPU; runs on a 3DGS backend.
- Sibling: *GauSS-MI* (Shannon mutual information for active 3DGS, arXiv 2504.21067, 2025).

## Problem

Rank / select a subset of an **already-captured posed pool** by expected information gain on
the scene's 3DGS, and emit an uncertainty map showing where the reconstruction is weak.

## Design

### Approach

Feed the posed pool + the DA3 scout field into FisherRF's Fisher-information EIG; rank the
pool by per-view information gain; select greedily (EIG is submodular-like → redundant views
yield ~0 gain by construction). Surface the per-region uncertainty as an operator-facing map
in the verify viewer.

### Known integration cost (why this is the reserve, not the default)

- **Needs a trained 3DGS field** as input. We produce the DA3 scout, but FisherRF expects an
  optimizing 3DGS; wiring the scout in (or a quick gsplat fit) is required.
- **Built for sequential next-best-view**, not batch subset-ranking of an existing pool — the
  NBV→batch-rank adaptation is the main unknown effort (estimate: days, not hours).
- **New GPU stack**: nerfstudio/gsplat + the FisherRF repo, CUDA build, a new container,
  added GPU contention on a step that is currently free CPU.
- Less deterministic / harder to debug than a ~200-LOC numpy module.

### Changes

| File | Change |
|------|--------|
| `images/fisherrf/` | new GPU container (nerfstudio/gsplat + FisherRF) |
| `real2sim/v4exec.py` | a `select-fisherrf` path (host orchestrator over SSH, MQTT progress) |
| verify viewer | render the EIG uncertainty map |

## Definition of Done

- [ ] Posed pool + scout field → EIG-ranked proposed-N on GPU.
- [ ] Per-region uncertainty map rendered in the verify surface.
- [ ] Head-to-head vs STO-SCN-103 on a real scene: does information-gain selection
      reconstruct measurably better than geometric coverage? (the falsifiable bar, T-001)

## Out of scope

- Everything STO-SCN-103 covers (geometric coverage selection) — that ships first.

## Implementation Notes

_(Deferred — do not start until STO-SCN-103 is shipped and the operator promotes this.)_

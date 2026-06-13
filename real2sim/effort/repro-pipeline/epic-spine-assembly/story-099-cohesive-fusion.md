---
xid: STO-SCN-099
parent: ./epic.md
kind: story
effort: scn
size: L
status: draft
date: 2026-06-13
depends-on: [STO-SCN-098, STO-SCN-095]
bd-id: krabby-88l
---

# Cohesive fusion of per-segment reconstructions into one gauge

## Summary

Fuse the M per-segment meshes/gaussians — now in one globally-registered gauge — into a
single cohesive space, without seams, double-surfaces, or gaps at the overlaps.

## Context

Final geometry step of EPI-SCN-SPINE-ASSEMBLY. Consumes globally-registered poses
(STO-SCN-098) and the per-segment reconstructions from the per-segment pipeline.

## Problem

Overlapping segments produce overlapping geometry. Naive union gives doubled walls and
seam artifacts. Fusion must blend/dedup overlap regions into one consistent surface.

## Design

### Approach

With everything in one gauge: blend overlapping geometry (e.g., TSDF/volumetric fusion or
overlap-aware mesh merge), dedup boundary surfaces, and produce a single mesh/gaussian.
Reuse the existing reconstruct/condition tooling where possible.

## Definition of Done

- [ ] M registered sub-reconstructions → one mesh/gaussian, no double-surfaces at seams.
- [ ] Overlap regions blended/deduped consistently.
- [ ] Output consumable by downstream condition/export.

## Out of scope

- Registration (STO-SCN-098); verification (STO-SCN-100).

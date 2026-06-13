---
xid: STO-SCN-099
parent: ./epic.md
kind: story
effort: scn
size: L
status: open
date: 2026-06-13
depends-on: [STO-SCN-098, STO-SCN-095]
bd-id: krabby-88l
assignee: krabby
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

## Implementation Notes

**Fuse in the single global gauge** (from STO-SCN-098). Two viable drivers:
- **Volumetric TSDF fusion** — reuse MAtCha's `extract_tsdf_mesh.py` (multi-resolution
  TSDF) or Open3D `ScalableTSDFVolume` as the off-the-shelf integrator. This is the same
  tooling STO-SCN-013 names, used here for *inter-segment* fusion.
- **Overlap-aware mesh merge** — where meshes already exist per segment, dedup/blend the
  overlap regions directly.

**Seams are a confidence problem, not averaging.** Integrate overlap with
**confidence-weighted depth** (per the submap-fusion notes) so the higher-confidence
segment wins at a seam rather than smearing both — this is what prevents doubled walls and
ghost surfaces at overlaps.

**Output → the downstream boundary.** One mesh/gaussian in the global gauge, consumed by
**STO-SCN-013** (mesh-condition) — the producer→consumer edge reconciled 2026-06-13 (see
STO-SCN-096 "Downstream boundary"). This story owns the **seam fusion**; 013 owns the
subsequent manifold/watertight conditioning.

**M=1 degenerate.** Nothing to register or fuse — pass the lone reconstruction straight
through to 013 unchanged. The stage must no-op cleanly so the single-space path doesn't pay
for spine machinery.

**Test.** M registered submaps → a single mesh with no double-surfaces at seams; overlap
regions blended/deduped; output reconstructs end-to-end into STO-SCN-013.

## Out of scope

- Registration (STO-SCN-098); verification (STO-SCN-100).

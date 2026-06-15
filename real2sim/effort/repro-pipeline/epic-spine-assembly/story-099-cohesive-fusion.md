---
xid: STO-SCN-099
parent: ./epic.md
kind: story
effort: scn
size: L
status: shipped
shipped: 2026-06-14
date: 2026-06-13
depends-on: [STO-SCN-098, STO-SCN-095]
bd-id: krabby-88l
assignee: krabby
tasks: 3
complete: 3
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

- [x] M registered sub-reconstructions → one mesh/gaussian, no double-surfaces at seams.
      (`spine_fuse.fuse`: confidence-weighted opacity feather — overlap density collapses to
      single coverage, not double. Unit-tested + validated on the real 3.25M scout: overlap
      mean-alpha ratio 0.55 ≈ 0.5 = cross-faded, not doubled.)
- [x] Overlap regions blended/deduped consistently.
      (Per-gaussian camera-coverage confidence `w_k = score_k/Σ score_j` from the 098 global
      poses — interior untouched `w≈1`, overlap halved `w≈0.5` so the two segments sum to ~1.
      The higher-confidence segment wins smoothly at the seam, no hard cut, no smear.)
- [x] Output consumable by downstream condition/export.
      (One global-gauge `fused.gs.ply` in the canonical 17-float 3DGS layout — exact
      write+read round-trip at 3.68M gaussians; the input STO-SCN-013 mesh-condition consumes.)

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

## Result (2026-06-14) — shipped: `spine-fuse@0` gaussian fusion node

Built `spine_fuse.py` (pure-numpy + scipy `cKDTree`) + the v4 node **`spine-fuse@0`**
(`tasks/spine-fuse.json` + `v4exec.py cmd_spine_fuse` + `spine-fuse` subcommand), with
`tests/test_spine_fuse.py` (8/8 green).

**Chose gaussian fusion over TSDF** for v1 — the pipeline is gaussian-centric (scout/verify/
da3 are 3DGS) and TSDF needs per-camera depth + Open3D; gaussian feather is lighter and
directly expresses the design's confidence-weighting. The pipeline's gaussians are the
**17-float DC-only 3DGS** (STO-SCN-095's "17×float32" note) — DC SH ⇒ colour is
rotation-invariant, so the SIM(3) transform is the easy case (position s·R·p+t, log-scale
+log s, quaternion ∘ R, f_dc/opacity carried). The 095 "never naively rewrite the .ply"
gotcha is handled in `write_ply` (header offset = the single `\n` after `end_header`, then
the raw float32 block).

**Seams are a confidence problem, not averaging.** Per-gaussian camera-coverage confidence
`w_k(p) = score_k/Σ_j score_j`, `score_j = exp(-(d_j/r)²)`, `d_j` = distance to segment j's
nearest 098-global camera (cKDTree). A segment interior keeps `w≈1` (untouched); an overlap
gets `w≈0.5` per segment so the two contributions SUM to ~1 — the doubled wall collapses to
one cross-faded surface, the higher-confidence segment winning smoothly. Applied as an
opacity feather (logit-space). M=1 is a clean pass-through (no feather). Output `fused.gs.ply`
in the global gauge → STO-SCN-013.

**Validation.** Unit: overlap cross-faded not doubled; coverage weights ≈1 interior / ≈0.5
overlap; transform positions+scale + inverse round-trip (incl. quaternion); M=1 passthrough;
exact PLY round-trip; format guard. **Real scale**: the 3.25M-gaussian 001-patio scout split
into two overlapping halves → read 0.3 s, fused 3.68 M in 4 s, overlap mean-alpha ratio
**0.55 ≈ 0.5** (cross-faded), exact write+read round-trip at 3.68 M.

Run:
```
v4exec.py spine-fuse <scene> --spine <id> --register <id> \
          --solves seg0=<subset>/<solve>,... --gaussians seg0=<ply>,... [--radius 0]
```

**Deferred (noted, not blocking):** the full M-real-segment end-to-end (needs the per-segment
pipeline reconstruction on each spine segment — the real-scale split-scout case already
exercises the engine + IO on real gaussian data); a TSDF/mesh fusion driver (gaussian fusion
is the v1; mesh merge is an alternative if a downstream wants meshes before 013); gap-filling
across genuinely uncovered regions (out of fusion's scope — a capture concern).

**Next (STO-SCN-100):** whole-spine verification — render the fused space + seams in the
scout verify surface for the operator to confirm cohesion.

## Out of scope

- Registration (STO-SCN-098); verification (STO-SCN-100).

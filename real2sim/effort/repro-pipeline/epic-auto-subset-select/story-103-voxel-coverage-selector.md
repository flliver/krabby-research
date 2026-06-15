---
xid: STO-SCN-103
parent: ./epic.md
kind: story
effort: scn
size: L
status: in-progress
date: 2026-06-14
depends-on: [STO-SCN-093]
bd-id: krabby-yeu
assignee: krabby
---

# Voxel-coverage best-N view selector (coverage-optimization greedy)

## Summary

A best-N view selector that maximizes **scene-surface coverage from good viewing
angles** — voxelize the scene, reward each camera for the voxel faces it observes
weighted by incidence-angle flux, greedy-add the camera with the largest marginal
coverage gain. Replaces STO-SCN-094's track-covisibility objective, which over-rewarded
clustered same-angle cameras (the operator's "95% same coverage from the same angle").

## Context

STO-SCN-094 selected by **shared SfM track points**, which structurally over-rewards
clustered cameras (they share tracks) and treats "same space, same angle" as good — the
root of the poor-variety result the operator flagged on the 095 verify surface (2026-06-14).
The fix is to measure coverage as **how well the camera network observes the scene's
surface volume**, so a redundant same-angle camera adds ~0 and a complementary angle on
the same surface is rewarded.

This is a **published, validated formulation**, not a local invention:
- *Coverage Optimization for Camera View Selection* (arXiv 2604.05259, 2026) — sample K of N
  posed cameras, greedy-add the most-diverse-baseline camera subject to whole-scene
  visibility. Almost exactly this story's objective.
- *Efficient View Clustering and Selection for City-Scale 3D Reconstruction* (arXiv 2207.08434)
  — the set-cover sibling (each point seen by ≥N_vis cameras).
- Classic MVSNet/COLMAP view-selection score (Gaussian-weighted triangulation angle).

Paired reserve story: **STO-SCN-104** (FisherRF, GPU information-gain) — deferred; this
geometric selector ships first to lock the select→verify loop and de-risk the shape.

## Problem

Choose N posed views that maximize reconstruction quality by **spatial + angular surface
coverage**, not track-sharing. Output: ranked proposed-N + a coverage report the human
verifies in the scout splat (STO-SCN-095). Must be gauge-free (the FastMap/DA3 solve gauge
is non-metric until scale calibration, STO-SCN-016), deterministic, CPU, and verifiable.

## Design

### Approach

Greedy submodular maximization over **voxel-face coverage** (the operator's spec, 2026-06-14):

1. **Voxelize** the scene geometry (`sparse/0` points; optionally scout-gaussian centers).
   Occupied voxels = voxels containing ≥1 point. Voxel size = a fraction of the scene-bbox
   diagonal (gauge-independent; e.g. diag/64), exposed as a knob so a metric size can be
   passed once scale is calibrated.
2. **Exposed faces** = faces of occupied voxels bordering empty space — the surfaces a
   camera could see. These are the coverage targets.
3. **Frustum → face visibility**: a face is observed by camera c if its center lies within
   c's FOV + depth range. (Occlusion via voxel-grid ray-march is a follow-up increment —
   see § Out of scope; first light is frustum + incidence only.)
4. **Flux weight** per (camera, face) = `max(0, cos θ)` between the face normal and the
   face→camera ray — 1.0 at a 90° hit, →0 at grazing. (Operator: "90° best, 0/180 worst.")
5. **Greedy**: each face's coverage = best flux any selected camera achieves on it; total
   objective = Σ face coverage. Add the camera with the largest **marginal** gain, stop at
   N or saturation. The multi-pass behavior the operator described ("down-weight covered,
   up-weight low-coverage boundary") falls out of submodular marginal gain for free.

Multi-scene/spine: a **shared voxel grid across segments is the coverage ledger** — the
same greedy, continued, naturally steers later cameras to uncovered regions (the IN side of
the STO-SCN-096 boundary contract still applies as pinned anchors).

### Changes

| File | Change |
|------|--------|
| `real2sim/voxel_coverage.py` | new — voxelize, exposed-faces, frustum-face flux, greedy select |
| `real2sim/select_views.py` | route to the voxel objective (or keep as the legacy track selector behind a flag) |
| `real2sim/verify_viewer/build_verify.py` | color voxel faces by coverage (red→green) in the surface |
| `real2sim/tests/test_voxel_coverage.py` | add — voxelization, face exposure, flux weight, greedy determinism |

## Definition of Done

- [ ] Posed pool + point cloud → ranked proposed-N maximizing voxel-face coverage flux.
- [ ] Gauge-free voxel sizing (bbox-relative), deterministic, pure-CPU (numpy ok).
- [ ] Coverage report: covered-face %, flux histogram, per-view marginal contribution,
      and a **median view-spread** that beats STO-SCN-094 on the real 001 pool.
- [ ] Verify surface colors voxel faces by coverage so the human SEES coverage (T-012),
      and the proposed-N show real angular variety (operator re-verifies, T-020).
- [ ] Tests written and passing.

## Testing

### Unit / fixture tests

- [ ] Voxelization: a known point set → expected occupied voxels at a given size.
- [ ] Exposed-face detection: interior faces excluded, boundary faces included.
- [ ] Flux weight: 90° hit = 1.0, grazing → 0, behind-face = 0.
- [ ] Greedy: deterministic; a redundant same-angle camera is not chosen second.

### Integration

- [ ] Real 001 pool (`6EHLYO3MF3QU`): proposed-24 has higher view-spread + face coverage
      than the 094 track-greedy on the same pool.

## Out of scope

- **Occlusion** (voxel-grid ray-march line-of-sight) — first light is frustum + incidence
  only; occlusion is the next increment on this story.
- FisherRF / information-gain selection — STO-SCN-104 (reserve).
- The pose solve (STO-SCN-093); human override/edit controls (STO-SCN-095 v2).

## Implementation Notes

_(Fill in during / after implementation.)_

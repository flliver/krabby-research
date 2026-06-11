---
xid: STO-SCN-059
parent: ./epic.md
kind: story
effort: scn
size: S
status: shipped
date: 2026-06-10
depends-on: []
bd-id: krabby-nsi
shipped: 2026-06-10
tasks: 1
complete: 1
---

# Research: VGGT-Omega 1B + Depth Anything 3 (discovery record)

> Discovery record, 2026-06-10. Sourced from web research; numbers are
> the papers'/authors' claims, NOT our measurements (T-010 — our
> measurements start at STO-SCN-060).

## What Fletcher pointed at

1. **VGGT-Omega (VGGT-Ω)** — Oxford VGG + Meta. CVPR 2026 oral;
   successor to VGGT (CVPR 2025 Best Paper). arXiv 2605.15195.
2. **Depth Anything 3 (DA3)** — ByteDance Seed, Nov 2025.
   arXiv 2511.10647.

Both are FEED-FORWARD: multi-view RGB → poses + intrinsics + dense
geometry in one transformer pass. No SfM prepass.

## VGGT-Omega 1B

- Outputs: camera poses, dense depth, point cloud, register tokens.
- Authors' claims: 20× more supervised / 100× more unsupervised
  training data than VGGT; 30% of VGGT's memory; 1.6× faster;
  +77% camera accuracy (Sintel); dynamic-scene support.
- Release: 1B checkpoint + text-aligned variant + HF demo
  (`facebook/vggt-omega`).
- Our hook: `real2sim/run_vggt.sh` + 001-patio `pipeline-vggt`
  already exist for VGGT v1 — Omega is a drop-in upgrade slot.
- Links: project page robots.ox.ac.uk/~vedaldi/research/2026/vggt-omega,
  github.com/facebookresearch/vggt, HF space facebook/vggt-omega.

## Depth Anything 3

- Single transformer, "depth-ray" representation; mono or multi-view,
  poses optional. Outputs: depth, poses, confidence, and DIRECT 3D
  GAUSSIANS (Fletcher: "produced splats instead of point cloud").
- Authors' claims: beats VGGT by 35.7% (pose) / 23.6% (geometry).
- Checkpoints: DA3NESTED-GIANT-LARGE-1.1 (1.40B),
  DA3-LARGE-1.1 (0.35B), DA3-BASE (0.12B), DA3METRIC-LARGE,
  DA3MONO-LARGE.
- **License split: Giant/Large = CC BY-NC 4.0 (non-commercial);
  Base/Small/Metric-Large = Apache 2.0.** Evaluation internally is
  fine; contract deliverables need the Apache tiers or a license
  resolution.
- DA3-Streaming: long video under 12 GB VRAM (fits the 16 GB fleet).
- Install: `pip install xformers torch>=2 torchvision; pip install -e .`
  + pinned gsplat commit for the gaussian head. CLI:
  `da3 auto <image-dir> --export-format glb|ply|npz --export-dir <out>`.
- Caveat: community repro of showcased splat quality is open
  (ByteDance-Seed/Depth-Anything-3 issue #44).

## Why this targets our holes

Holes = no surface where curated views don't overlap (sparse
correspondence + TSDF truncation). Feed-forward dense depth predicts
geometry for EVERY pixel of EVERY view, including low-texture regions
where MASt3R correspondence starves. The bet: fewer holes at equal
frame budget.

## Definition of Done

- [x] Both models identified, capabilities/licenses/claims recorded
      with sources; integration hooks named; risks captured in the
      epic.

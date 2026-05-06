---
kind: note
captured: 2026-05-01T17:46:50-07:00
consolidated: true
tags: []
---
# Submap-based mesh fusion — scaling MAtCha beyond a single run

Architectural pattern surfaced during the SfM-scaling experiment. Jeremy asked: "if we get good at scaling up the camera path, and we get good at LOCAL area mapping, then couldn't we bridge two meshes together by iterating within localized areas and using the unified camera paths as a connection-point?"

Yes — and this is a well-established pattern in large-scale 3D reconstruction (Polycam, Matterport, KinectFusion, the Hierarchical 3D Gaussian Splatting paper Kerbl et al. ACM TOG 2024).

## The decomposition

```
GLOBAL: ONE sparse SfM on keyframes spanning the whole property → unified camera frame
LOCAL:  MANY MAtCha runs on dense subsets within that unified frame
FUSE:   merge local meshes using the unified camera positions as rigid anchors
```

The global SfM is light (sparse keyframes, fast, fits in VRAM regardless of total scene size). The local MAtCha runs are bounded so each fits in 16 GB. The fusion is mostly a coordinate transform + volumetric merge.

The camera path is the spine; local meshes are the flesh; the path tells the meshes how to relate to each other.

## Why this works (and why it doesn't accumulate drift)

A single global SfM gives you a consistent coordinate frame for *every* keyframe in the property. Each subsequent local-region MAtCha run uses `--image_idx` to select frames from that pool. All resulting meshes are already in the same frame — no inter-submap registration step is needed for the *coarse* alignment. ICP / Procrustes is only needed for *fine* refinement if the per-region MAtCha deformation has caused subtle scale drift.

This is the elegance: **drift only enters if you chain SfM windows.** If you do ONE global SfM and then use those poses for many MAtCha runs, you get global consistency for free. Submaps inherit the spine.

## What we already have

- MASt3R-SfM produces unified poses across hundreds of frames (this is what the in-progress scaling experiment is testing).
- MAtCha exposes `--image_idx 5 12 23 ...` so each local run targets a specific frame subset within the unified pool.
- MAtCha's `extract_tsdf_mesh.py` already implements multi-resolution TSDF fusion (designed for unbounded scenes within a single MAtCha run, but the depth-map → TSDF → marching cubes machinery is exactly what we'd use for inter-submap fusion).
- `--alignment_only`, `--refinement_only`, `--mesh_only` flags let us run any stage independently.

## What we'd need to build

- **Submap clusterer** — group cameras by co-visibility (from the SfM scene graph) or by spatial position (k-means on camera centers). Co-visibility is more principled — handles loops, backtracking, multi-room scenes naturally.
- **Per-submap orchestrator** — for each cluster, feed those frame indices to MAtCha, capture the mesh + per-camera depth maps in the unified frame.
- **Fusion driver** — combine per-submap depth maps into a single global TSDF volume, run marching cubes. Open3D has `o3d.pipelines.integration.ScalableTSDFVolume` off the shelf.

Total scope: ~1–2 weeks of careful work. More if we want explicit drift correction (we probably won't need it).

## Closest published precedent

**Hierarchical 3D Gaussian Splatting** — Kerbl, Meuleman, Kopanas, Wimmer, Lanvin, Drettakis. *ACM TOG 2024*. Cited in the MAtCha paper as ref [27]. Same authors as the original 3DGS paper. Specifically titled "*A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets.*" Their result: scenes with 100,000+ images, kilometer-scale areas, real-time rendering. Drone footage of entire neighborhoods.

The MAtCha paper itself uses Kerbl's affine-rescaling idea (paper §7.1). There's already cross-pollination between the two works. **A "Hierarchical MAtCha" extension is a research-shaped problem, not a fundamental novelty.**

## Gotchas

1. **Scale consistency between submaps.** MAtCha's per-chart deformation MLP can re-scale geometry differently in different runs (depth-prediction ambiguity per region). Even with agreed camera positions, mesh scales may drift. Mitigations:
   - Use the unified SfM's sparse 3D points as a constant scale anchor across all submap runs.
   - Add a final scale-alignment step between submaps using shared frames.
   - Skip MAtCha's per-chart deformation entirely for the scale-up case — accept slightly worse local detail in exchange for global consistency.

2. **Free-Gaussians refinement is global.** MAtCha's optional Free-Gaussians stage operates on all cameras simultaneously. If we split into submaps, we lose this global refinement. The mesh-from-TSDF path doesn't strictly need it; you'd accept the loss.

3. **Boundary artifacts.** Per-submap meshes are noisy at edges where cameras saw the surface from poor angles. TSDF fusion handles this via confidence-weighted depth integration — overlap regions where multiple submaps agree get reinforced; edge regions where only one submap saw them get noisy. Not an averaging problem; a confidence problem.

## Connection to the near-term B5 work

The same "global SfM → per-cluster operation" decomposition shows up in the camera-selection UI:

- Run **one** big SfM at N=300+ to get the unified camera frame.
- Use the resulting pose graph to **cluster cameras by co-visibility** (the third filter axis in the proposed Route B viser viewer's grouping table).
- Hand-pick 12 frames informed by the cluster structure.
- Run **one** MAtCha on those 12.

So the scaling architecture and the curation UI converge on the same primitive: **co-visibility clustering of unified-frame cameras**. Building it once supports both directions.

## When this matters

Not for M11. Our scenes are individual rooms (firepit, sky-house dining, patio); each fits in a single MAtCha run today.

For M12+ when we want **whole-property** captures (a complete walk-through of a multi-room space, an entire job site, a backyard with multiple structures), the submap-fusion architecture is the right answer and Hierarchical 3DGS is the most directly relevant prior art to study.

Worth knowing about now so the M11 pieces (unified SfM, candidate clustering, per-cluster MAtCha) are built in a way that this future stitching doesn't require a redesign.

## Status

Future-direction note. Not actionable for M11. May be promoted to a `scale-up` thread when the cross-milestone scope justifies it.

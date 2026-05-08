# 011 — Scene Reconstruction

## Overview

Pipeline: phone video → frame extraction → COLMAP SfM → dense MVS → mesh conditioning → USD → IsaacSim.

Grant spec: `grants/grants/Krabby-Uno/Milestone11-Scene-Reconstruction/OVERVIEW.md`

## Captures

| ID | Video | Lens Mode | Source | Frames |
|----|-------|-----------|--------|--------|
| 001 | `data/videos/001-patio-fisheye/capture.mp4` | Native (155° fisheye) | DJI Osmo Action 3 | 942 (4K HEVC, 30fps hyperlapse) |
| 002 | `data/videos/002-patio-dewarped/capture.mp4` | Dewarp ON | DJI Osmo Action 3 | 944 (4K HEVC, 30fps hyperlapse) |

Scene: Outdoor patio area — stamped concrete, wood deck, A-frame house, covered pavilion, trees.

## Lessons Learned (2026-04-11)

### Camera Model Selection
- DJI Action 3 native FOV is ~155° — true fisheye territory
- `PINHOLE` → total failure (0 images registered)
- `OPENCV` → 2/63 images (can't model enough distortion)
- `OPENCV_FISHEYE` → 2/63 images (too many params, mapper can't grow)
- **`SIMPLE_RADIAL_FISHEYE` → 69/94 images** (sweet spot — few enough params for stable init)
- Dewarped video should allow `SIMPLE_RADIAL` (standard lens model)

### Hyperlapse Handling
- These videos are hyperlapses, NOT standard video — each frame already has significant viewpoint change
- Extracting at 2-3 fps drops ~97% of frames → too sparse for COLMAP matching
- **Must extract at full framerate (30fps)** to preserve inter-frame overlap

### Matching Strategy
- `sequential_matcher` with overlap 10 → only 2 images registered
- `exhaustive_matcher` → much better coverage but O(n²) for large frame counts
- For 942-frame hyperlapse: sequential with overlap 15 is the right balance

### Mapper Settings
Relaxed mapper thresholds needed for action camera footage:
- `init_max_error 8` (default 4)
- `abs_pose_min_num_inliers 15` (default 30)
- `init_min_num_inliers 50` (default 100)
- `multiple_models 0` (force single model)

## Layers

- **Docker** (`docker/`) — Container with COLMAP, Open3D, PyMeshLab, trimesh
- **Workspace** (`workspace/`) — Pipeline scripts synced to outposts
- **Data** (`data/`) — Local data directory (not synced via workspace layer)

## Pipeline Scripts

| Script | Task | Input | Output |
|--------|------|-------|--------|
| `extract_frames.sh` | Video → JPEG frames | `data/videos/*/capture.mp4` | `data/scenes/*/images/` |
| `run_colmap_sparse.sh` | T0: SfM sparse reconstruction | `data/scenes/*/images/` | `data/scenes/*/sparse/` |
| `run_colmap_dense.sh` | T1: MVS dense reconstruction | `data/scenes/*/sparse/` | `data/scenes/*/dense/fused.ply` |
| `run_mesh_conditioning.sh` | T2: Mesh cleanup + collision proxy | `data/scenes/*/dense/fused.ply` | `data/scenes/*/mesh/` |

## Outpost Usage

```bash
# Provision an outpost for this milestone:
AI/outposts/provision.sh <outpost> --milestone 011-scene-reconstruction

# Sync workspace only:
AI/outposts/provision.sh <outpost> --milestone 011-scene-reconstruction --layer 3

# Transfer scene data (tar pipe — faster than rsync for many files):
tar cf - -C data/scenes/<scene>/images . | ssh <outpost> "tar xf - -C ~/outposts/krabby/data/011-scene-reconstruction/scenes/<scene>/images/"

# Run T0 inside container on outpost:
ssh <outpost>
docker run --gpus all \
  -v ~/outposts/krabby/data/011-scene-reconstruction:/data \
  -v ~/outposts/krabby/workspace/milestones/011-scene-reconstruction/workspace:/workspace \
  krabby-011-scene-reconstruction \
  bash /workspace/run_colmap_sparse.sh <scene> [camera_model]

# Camera models:
#   SIMPLE_RADIAL_FISHEYE  — for fisheye/ultra-wide (DJI Action 3 native)
#   SIMPLE_RADIAL          — for dewarped video or normal lenses
```

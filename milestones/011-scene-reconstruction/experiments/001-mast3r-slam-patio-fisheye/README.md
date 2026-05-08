# Experiment 001 — MASt3R-SLAM on patio-fisheye

**Status:** ✅ produced point cloud + Poisson-conditioned mesh
**Date:** ~2026-04-12
**Pipeline:** MASt3R-SLAM (krabby-mast3r:earlier — sbeeprz-built, Ada-only)
**Hardware:** sbeeprz (RTX 4080)
**Reference:** `docker/Dockerfile.mast3r`, `docker/MAST3R-NOTES.md`, OLAI corpus `3d-reconstruction/mast3r-slam`

## Input

- Same patio scene as experiment 001-COLMAP
- Same DJI Action 3 capture, native 155° fisheye, 4K hyperlapse, 942 frames
- Resolution downscaled to 720p for MASt3R-SLAM input

## Process

MASt3R-SLAM's online sliding-window SLAM directly on the video (no need
for the explicit "extract → match → map" stages COLMAP requires).

## Output

- `data/scenes/001-patio-fisheye/mast3r_output/patio/patio_720p.ply` — **424 MB** dense point cloud
- `data/scenes/001-patio-fisheye/mesh/patio_mast3r.obj` — **14 MB** mesh (post Open3D conditioning)

## Quality verdict

Dense, captures the patio scene comprehensibly. The 424 MB cloud is
larger than scene 003 / 004 outputs because this run was at higher
resolution (720p) and likely had more keyframes than later 10fps runs.

The conditioned 14 MB OBJ is roughly comparable in scale to scene 004's
19 MB ball-pivoting mesh. Visual fidelity is "scene shape is correct,
mesh has holes wherever the camera path didn't densely cover."

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | 🟡 — Open3D conditioned, but ball-pivoting / Poisson on SLAM clouds is not reliably watertight | Holes wherever the camera didn't densely sample |
| R3 Camera poses | ✅ — MASt3R-SLAM exports per-frame poses | |
| R5 Wall-clock | 🟡 — multiple hours for 942 frames at 720p (older sbeeprz Ada-only image, not optimized) | |
| R7 M11-validated | ✅ — first MASt3R-SLAM scene | This experiment proved the pipeline could produce a usable cloud from a real DJI fisheye capture |

## Key findings

1. **MASt3R-SLAM ran fine on Ada (sm_89)** with a single-arch image. The
   later multi-arch image work (April 29) was driven by needing to support
   sm_120 / RTX 5080.
2. **Scale of output is sensitive to keyframe count** — this run produced
   a 424 MB cloud; a later run on scene 003 at 720p / 10fps produced only
   72 MB. Sampling rate matters.
3. **The Open3D ball-pivoting mesh is not watertight** — visible holes
   wherever the camera didn't densely sample. For collision-quality use
   we'd need either Poisson with density-cropping or a method that
   produces watertight output natively.

## What this experiment told us

MASt3R-SLAM is a **viable SLAM frontend**. The main M11 question it left
open: how to get a watertight mesh from the cloud. That question was
later answered (in part) by trying MAtCha on scene 004, which produces
a watertight mesh natively via TSDF + tetrahedralization.

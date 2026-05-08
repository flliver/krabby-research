# Experiment 003 — SLAM3R on firepit-fisheye

**Status:** ✅ produced point cloud
**Date:** ~2026-04-12
**Pipeline:** SLAM3R (PKU-VCL-3DV, CVPR 2025 Highlight)
**Hardware:** Probably sbeeprz or dbeeprz (RTX 4080)
**Reference:** `docker/Dockerfile.slam3r`, OLAI corpus `3d-reconstruction/slam3r`

## Input

- Same firepit scene as `003-mast3r-slam-firepit`
- DJI Action 3 native fisheye, 720p downscale, 10fps subsample

## Process

SLAM3R via `Dockerfile.slam3r` (CUDA 12.8 base, Python 3.11, PyTorch
2.5.0+cu124). Single-process feed-forward — no PyTorch multiprocessing
deadlock to manage (no `--shm-size=8g` requirement).

## Output

- `data/scenes/003-firepit-fisheye/slam3r_output/003-firepit-fisheye_slam3r_images_recon.ply` — **15 MB** dense point cloud
- No mesh produced by SLAM3R (point cloud only)
- No camera poses exported

## Quality verdict

15 MB cloud — significantly smaller than MASt3R-SLAM's 72 MB output on
the same scene. Likely fewer points / lower density. Visual quality
remains to be compared head-to-head.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ❌ — no mesh produced | Would need a separate Poisson / ball-pivoting pass |
| R3 Camera poses | ❌ — SLAM3R does not export per-frame intrinsics/extrinsics | Real limitation if downstream needs camera-known data |
| R5 Wall-clock | ✅ — ~15 min on RTX 4080 | Comparable to MASt3R-SLAM |
| R6 Build complexity | ✅ — significantly easier than MASt3R-SLAM (no source patches needed for current PyTorch 2.5) | |
| R7 M11-validated | ✅ — produced output on real M11 footage | |

## Why SLAM3R is interesting

- **Build is dramatically simpler** than MASt3R-SLAM (CUDA 12.8 + cu124 + PyTorch 2.5; minimal patches; single-process feed-forward).
- **No `--shm-size=8g` deadlock** — runs in default Docker without shared-memory tuning.
- **Faster startup time** for iteration if the M11 question is "did the capture work?"

## Why MASt3R-SLAM still wins for M11

- Camera-pose recovery is meaningful for downstream texturing and IsaacSim placement.
- Point cloud is denser at the same input.
- Once the build complexity is paid (and it has been — `krabby-mast3r:latest` is on bbeeprz), MASt3R-SLAM's incremental cost per scene is the same as SLAM3R's.

## Open question

We have outputs from both pipelines on the same scene (`003-firepit-fisheye`).
A side-by-side mesh comparison in Blender would be the cheapest way to
get a quality verdict. We have not done this yet.

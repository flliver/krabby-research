# Experiment 003 — MASt3R-SLAM on firepit-fisheye

**Status:** ✅ produced point cloud + Poisson-conditioned mesh
**Date:** ~2026-04-12
**Pipeline:** MASt3R-SLAM (krabby-mast3r:earlier — sbeeprz-built, Ada-only)
**Hardware:** sbeeprz (RTX 4080)
**Reference:** `docker/Dockerfile.mast3r`, `docker/MAST3R-NOTES.md`

## Input

- Scene: outdoor firepit area
- Capture: DJI Action 3 native fisheye (155° FOV)
- Resolution: downscaled to 720p
- Frame rate: subsampled to 10fps (from likely 30fps source)

## Process

`workspace/run_mast3r.sh` on the firepit video.

## Output

- `data/scenes/003-firepit-fisheye/mast3r_output/firepit/firepit_720p_10fps.ply` — **72 MB** dense point cloud
- `data/scenes/003-firepit-fisheye/mesh/mesh_conditioned_mast3r.obj` — **16 MB** Poisson-conditioned mesh
- `data/scenes/003-firepit-fisheye/mesh/mesh_conditioned.ply` — 7.4 MB (PLY version)

## Quality verdict

Smaller cloud than scene 001 (72 MB vs 424 MB) due to:
- Lower frame rate (10 fps subsample vs full 30 fps)
- Same 720p resolution
- Likely shorter clip

The Poisson-conditioned 16 MB OBJ shows the firepit area as a coherent
surface mesh. With density cropping it's reasonably clean.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | 🟡 — Poisson with density crop, partially watertight | Better than ball-pivoting; still has holes near scene edges |
| R3 Camera poses | ✅ — MASt3R-SLAM exports them | |
| R5 Wall-clock | ✅ — manageable thanks to 10fps subsampling | |
| R7 M11-validated | ✅ — second MASt3R-SLAM scene | Confirmed pipeline reproducibility across scenes |

## Comparison to SLAM3R-003

The same scene was also run through SLAM3R (see `003-slam3r-firepit/`).
Output PLY sizes: MASt3R-SLAM 72 MB, SLAM3R 15 MB. The size delta is real
(MASt3R's keyframe density vs SLAM3R's denser per-frame output, and how
each represents the sparse cloud) but isn't a quality verdict — visual
inspection of both meshes is needed for an apples-to-apples judgement.

## Key finding

This experiment validated the **10fps subsampling capture-side
optimization**. We later codified 2.7K @ 30fps as the recommended profile
for scene 004, but the 720p / 10fps profile here proved that aggressive
temporal subsampling doesn't break MASt3R-SLAM. Useful when bandwidth
or storage is constrained.

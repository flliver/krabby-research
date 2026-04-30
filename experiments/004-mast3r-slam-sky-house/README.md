# Experiment 004 — MASt3R-SLAM on sky-house-dining

**Status:** ✅ produced point cloud + ball-pivoting mesh
**Date:** 2026-04-29
**Pipeline:** MASt3R-SLAM (krabby-mast3r:latest — multi-arch sm_89+sm_120, NGC PyTorch 25.10 base)
**Hardware:** bbeeprz (RTX 5080, Ryzen 9800X3D)
**Reference:** `docker/Dockerfile.mast3r`, `docker/MAST3R-NOTES.md`, `AI/agents/krabby/inbox/handoff-2026-04-29-1347.md`, OLAI corpus `3d-reconstruction/mast3r-slam`

## Input

- Scene: outdoor sky-house dining area
- Capture: DJI Action 3, **2.7K @ 30fps, locked exposure/WB, stable motion** (the validated profile)
- Duration: 3:47 (concatenated from `capture-01.mp4` + `capture-02.mp4`)
- Frame count: 6,804
- Native fisheye (no dewarp)

## Process

`milestones/011-scene-reconstruction/workspace/run_mast3r.sh` on bbeeprz,
container started with `--gpus all --shm-size=8g`.

## Runtime

**~40 minutes** for the full 3:47 video on RTX 5080. Roughly 15–20 min
of processing per minute of 2.7K input — the published estimate that
became the planning rule of thumb.

## Output

- `data/scenes/004-sky-house-dining/mast3r_output/sky_house/004-sky-house-dining.ply` — **153 MB** dense point cloud (~10.7 million points)
- `data/scenes/004-sky-house-dining/mast3r_output/sky_house/004-sky-house-dining.txt` — per-frame camera poses (13 KB)
- `data/scenes/004-sky-house-dining/mesh/sky_house_mast3r.obj` — **19 MB** ball-pivoting mesh, 200K triangles (post Open3D conditioning)

## Quality verdict

This is the production-quality MASt3R-SLAM output that motivated the
broader "is MAtCha better?" question. Per Jeremy's read after running
the pipeline:

> "Honest take: result quality is below what I expected given the input
> matched the paper's spec. Worth a review together — either the capture
> needs more than 'follow the paper' or the Open3D conditioning pass is
> leaving quality on the table."

The mesh is **not watertight** — Open3D ball pivoting follows the cloud
faithfully but leaves holes wherever the SLAM didn't densely sample. For
the M11 collision-quality goal this is a real shortcoming.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ❌ — ball-pivoting is faithful but not watertight | Could try Poisson + density crop, untested for this scene |
| R3 Camera poses | ✅ — `004-sky-house-dining.txt` (13 KB) | |
| R4 Multi-arch | ✅ — image runs on Ada (sm_89) + Blackwell (sm_120) | This was the milestone for the multi-arch image work |
| R5 Wall-clock | 🟡 — 40 min for 4-min video; ~15-20 min/min input | Acceptable, not great |
| R7 M11-validated | ✅ — third MASt3R-SLAM scene | |

## Capture profile that worked

- 2.7K @ 30fps (downsamples to MASt3R's native 512px without major loss)
- Locked exposure / white balance (avoids learned point-map drift)
- Stable handheld motion
- Native fisheye, no dewarp
- DJI Action 3, 155° FOV

This profile is now codified as the recommended M11 capture standard
in OLAI corpus `3d-reconstruction/capture-profiles`.

## Why we tried MAtCha next

The dissatisfaction with this output's mesh quality (relative to the
input's apparent suitability) is what motivated experiment
`004-matcha-sky-house`. MAtCha's TSDF + tetrahedralization promised a
**watertight** mesh natively, addressing R1.

That experiment succeeded. See `004-matcha-sky-house/README.md`.

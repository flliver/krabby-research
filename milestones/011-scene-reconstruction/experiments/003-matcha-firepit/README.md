# Experiment 003 — MAtCha on firepit-fisheye

**Status:** ✅ end-to-end, watertight mesh produced (Phase A2)
**Date:** 2026-04-30
**Pipeline:** MAtCha (`krabby-matcha:latest` on bbeeprz)
**Hardware:** bbeeprz (RTX 5080, 16 GB)
**Reference:** `docker/Dockerfile.matcha`, `docker/MATCHA-NOTES.md`

## Input

- Scene: outdoor firepit area
- Capture: DJI Action 3 **4K @ 60 fps regular video** (not hyperlapse), 5:31 duration, **19,842 frames**, native 155° fisheye
- Frames sampled for MAtCha: **12**, evenly spaced across the 5:31 (stride ≈ 1654 frames), downscaled to 1024×576

## Process

```bash
# Frame extraction — 12 evenly-spaced samples from 19,842 source frames
ffmpeg -i videos/003-firepit-fisheye.mp4 \
  -vf "fps=12/331,scale=1024:-2" -q:v 2 \
  frames/003-matcha-12/frame_%04d.jpg

# MAtCha inference (same as scene 001 / 004)
python train.py -s /data/frames/003-matcha-12 -o /data/matcha_output/003-firepit-fisheye \
  --sfm_config unposed --n_images 12 \
  --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
  --depthanything_encoder vitl
```

## Runtime

**~8 minutes** end-to-end on RTX 5080. Same wall-clock as scene 001 despite 21× more source frames — runtime is gated by MAtCha's per-keyframe processing, not by the source video length.

## Output

- `data/scenes/003-firepit-fisheye/matcha_output/tetra_mesh_binary_search_7.ply` — **238 MB**, **11.8M triangles**, watertight
- `data/scenes/003-firepit-fisheye/matcha_output/mesh/firepit_matcha_200k.obj` — **21 MB**, 200K-tri decimation
- `data/scenes/003-firepit-fisheye/matcha_output/mesh/firepit_matcha_500k.obj` — **37 MB**, 500K-tri decimation
- `data/scenes/003-firepit-fisheye/matcha_output/cameras.json` — 12 camera poses
- `data/scenes/003-firepit-fisheye/matcha_output/points.ply` — 32 MB MASt3R-SfM intermediate cloud

## Quality verdict

> **"Chaotic, but obviously the filmed scene. Also includes too much background noise."** — Jeremy, 2026-04-30 inspection

Same character as scene 001: the firepit area is recognizable, but the mesh extends well beyond what's relevant for collision geometry. Distant tree-line and sky-region artifacts pollute the output.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ✅ | |
| R3 Camera poses | ✅ — 12 poses | Not embedded in mesh visualization |
| R5 Wall-clock | ✅ — 8 min | |
| R7 M11-validated | ✅ | |
| **Quality (subjective)** | 🟡 — chaotic, needs post-processing | Foreground OK, background polluting |

## Lessons specific to this scene

1. **4K @ 60fps "regular" video at 5:31 is overkill for MAtCha** — we used only 12 frames out of 19,842. The bandwidth and storage cost (4.3 GB upload to bbeeprz) was disproportionate to the input MAtCha actually consumed.
2. **Frame stride of ~1654 frames produced workable viewpoint diversity** for this scene because the camera path was a continuous walk-around. For scenes with backtracking or stationary segments, even sampling could miss diverse viewpoints (a future experiment).
3. **Firepit-as-foreground was reasonably preserved** — the central object is recognizable. The "chaotic" character is in the surrounding scene, not the object of interest.

See `CAPTURE-LESSONS.md` for capture-side findings.

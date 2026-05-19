# Experiment 001 — MAtCha on patio-fisheye

**Status:** ✅ end-to-end, watertight mesh produced (Phase A1)
**Date:** 2026-04-30
**Pipeline:** MAtCha (`krabby-matcha:latest` on bbeeprz)
**Hardware:** bbeeprz (RTX 5080, 16 GB)
**Reference:** `docker/Dockerfile.matcha`, `docker/MATCHA-NOTES.md`

## Input

- Scene: outdoor patio at a cabin
- Capture: DJI Action 3 4K hyperlapse, ~30 fps effective, **31 sec, 942 frames**, native 155° fisheye
- Frames sampled for MAtCha: **12**, evenly spaced across the 31 sec, downscaled to 1024×576 (matches scene 004 convention)

## Process

```bash
# Frame extraction (inside krabby-mast3r container, ffmpeg)
ffmpeg -i videos/001-patio-fisheye.mp4 \
  -vf "fps=12/31,scale=1024:-2" -q:v 2 \
  frames/001-matcha-12/frame_%04d.jpg

# MAtCha inference
docker exec matcha-build bash -c '
  source /opt/matcha/bin/activate &&
  cd /opt/MAtCha &&
  export PYTHONPATH="..." &&
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True &&
  python train.py -s /data/frames/001-matcha-12 -o /data/matcha_output/001-patio-fisheye \
    --sfm_config unposed --n_images 12 \
    --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
    --depthanything_encoder vitl
'
```

## Runtime

**~8 minutes end-to-end** on RTX 5080. Faster than scene 004 (~11 min) because the source is smaller — fewer triangles to extract at the tetra-mesh stage.

## Output

- `data/scenes/001-patio-fisheye/matcha_output/tetra_mesh_binary_search_7.ply` — **185 MB**, **9.2M triangles**, watertight
- `data/scenes/001-patio-fisheye/matcha_output/mesh/patio_matcha_200k.obj` — **16 MB**, 200K-tri decimation
- `data/scenes/001-patio-fisheye/matcha_output/mesh/patio_matcha_500k.obj` — **38 MB**, 500K-tri decimation
- `data/scenes/001-patio-fisheye/matcha_output/cameras.json` — 12 camera poses
- `data/scenes/001-patio-fisheye/matcha_output/points.ply` — 30 MB MASt3R-SfM intermediate cloud

## Quality verdict

> **"Chaotic, but obviously the filmed scene. Includes too much 'background noise' (far away things) that would ideally be culled."** — Jeremy, 2026-04-30 inspection

The reconstruction recovers the patio scene recognizably. Foreground geometry (deck, posts, A-frame structure) is present. But there's substantial geometric noise from far-distant objects (trees, sky-region artifacts) that don't belong in a useful collision mesh. See `CAPTURE-LESSONS.md`.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ✅ — TSDF + tetra | Same construction as scene 004 |
| R3 Camera poses | ✅ — 12 poses recovered | But not embedded in mesh — see cross-cutting issues |
| R5 Wall-clock | ✅ — 8 min | |
| R7 M11-validated | ✅ — produced output | |
| **Quality (subjective)** | 🟡 — chaotic, needs post-processing | Foreground OK, background polluting |

## Lessons that informed Phase A retrospective

This was the second MAtCha run (after scene 004). Confirmed:

1. The **12-frame, 1024px-wide, vitl-encoder, unposed-SfM** recipe is reproducible across scenes.
2. **Hyperlapse input works fine for MAtCha** — temporally-uniform frame sampling on a hyperlapse is also viewpoint-uniform because the camera moves smoothly. No special handling needed at this scale.
3. **MAtCha doesn't auto-cull background** — the mesh extends past where any usable collision geometry should be. This is a generic post-processing gap, not specific to this scene.

See `CAPTURE-LESSONS.md` for capture-side findings and `../../PLAN.md` (post-processing section) for the cross-cutting issues.

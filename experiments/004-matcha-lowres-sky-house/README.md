# Experiment 004 — MAtCha lowres on sky-house-dining (PHASE B6a)

**Status:** ❌ **negative result — visibly worse than the 12-frame baseline; do not pursue**
**Date:** 2026-05-01
**Pipeline:** MAtCha (`krabby-matcha:latest`) at 768×432 input vs the 12-frame baseline at 1024×576
**Hardware:** bbeeprz (RTX 5080, 16 GB)
**Reference:** `experiments/004-matcha-sky-house/README.md` (the baseline being compared against)

## Visual verdict (2026-05-01, post-inspection)

> "Complete garbage. More lower-quality photos is certainly worse — at
> least the way we did it." — Jeremy

The quantitative metrics in the comparison table below (better cull
retention, more views per vertex, higher color coverage) **did not
translate to visually better mesh.** The per-pixel detail loss from
768×432 input dominated the apparent gains from more frames. The
tradeoff was strongly net-negative on this scene.

**Conclusion**: dropping input resolution to fit more frames is a bad
lever for MAtCha quality on this scene type. The cure is not "more
lower-detail views" — the resolution loss matters more than view-count
gain at this scale.

What this rules out:
- Lower-res-with-more-frames as a path to better quality.
- Probably also rules out lower-res-at-same-frames (no reason to
  expect quality improvement from less detail at same view count).

What this does NOT rule out:
- Better frame *selection* at the same 12-frame budget (B5 — manual
  curation of the 12 best viewpoints from a wider candidate pool).
- Higher-resolution input at the same 12-frame budget (1280×720), if
  it fits in 16 GB VRAM.
- Other MAtCha quality knobs (gaussian splat iters, TSDF settings) we
  haven't touched.

## Hypothesis

Lower-resolution keyframes should reduce VRAM pressure in MAtCha's
chart-alignment stage (where the 12-frame ceiling on 16 GB lives at
1024×576), letting us fit more keyframes. More keyframes → better
viewpoint coverage → potentially better mesh quality.

## Method

Same scene 004 source video. Same MAtCha pipeline. **Only one variable
changed**: input frame resolution dropped from 1024×576 to **768×432**
(56% the pixel count per frame).

Sweep tested at 768×432:

| Frame count | Result | Memory at chart-align |
|---|---|---|
| 24 | ❌ OOM | 9.84 GiB used + 648 MiB tried (free 680 MiB) |
| 18 | ❌ OOM | 10.23 GiB used + 366 MiB tried (free 272 MiB) |
| **15** | ✅ ran end-to-end | (succeeded) |

So **the new ceiling at 768×432 is somewhere between 15 and 18 frames**,
likely 16 or 17. We stopped at 15 (a clear pass) rather than binary-search
the exact boundary; the comparison vs the 12-frame baseline is the
useful question.

## Outputs

```
data/scenes/004-sky-house-dining/matcha_output/oriented-lr15/
├── cameras.json                          — 15 camera intrinsics + cams2world
├── oriented_cameras.json                 — orient_mesh.py: R + z_shift
├── oriented_500k_colored_culled.ply     ★ — final mesh, 290K verts / 500K tris / 20 MB
├── mast3r_frames/                        — 15 source images at 768×432
└── scene_culled_lr15.blend              ★ — Blender scene with mesh + 15 cameras + image planes
```

## Quantitative comparison vs 12-frame baseline

| Metric | 12-frame baseline (1024×576) | **15-frame lowres (768×432)** |
|---|---|---|
| Frames used | 12 | 15 |
| Per-frame resolution | 1024×576 | 768×432 |
| Pipeline runtime | ~11 min | ~11 min (end-to-end) |
| Tetra mesh | 422 MB / 21M tris | **239 MB / 11.9M tris** |
| Cull retention | 78% v / 65% t | **90.5% v / 87.3% t** ★ |
| Vertex color coverage | 89.4% | **97.7%** ★ |
| Mean views/vertex | 3.39 | **5.86** ★ |
| Median views/vertex | 4 | **6** ★ |
| Final 500K-tri mesh | 14.8 MB OBJ | **20.3 MB PLY** (similar magnitude) |
| Camera mean height | +1.05 m | +0.94 m |

★ = improvement at the lower-res-with-more-frames setting.

The view-count distribution shift is the most telling number: median
4 → 6 means each vertex is now triangulated against ~50% more cameras,
which directly improves both MAtCha's geometric inference and our color
projection's averaging.

The lower-res input does mean **less per-pixel detail in the projected
colors**. That's a real tradeoff — eyeball verdict pending.

## Floor deduction shifted

Worth flagging: B1 picked a different RANSAC candidate as floor:

| | candidate | score | mean cam height | mesh z range |
|---|---|---|---|---|
| 12-frame baseline | cand 1 | 18,865 | +1.05 m | [-0.34, +2.80] |
| **15-frame lowres** | cand 2 | **12,075** | +0.94 m | [-1.26, +2.91] |

The 15-frame mesh has 1.26m of "below-floor" geometry vs the baseline's
0.34m. Either:
- More sub-floor tetra-mesh noise from MAtCha at 15 frames
- Or B1 picked a slightly-different candidate plane (close to the
  second-place option in score)

Cull dropped the obvious below-floor outliers (32K verts with z < -0.5).
Should be harmless but worth confirming visually.

## Open follow-ups (revised after the negative result)

Given the visual verdict, items 2–4 of the original list are dropped
(no point exploring the lowres regime further). What remains:

1. ~~**Visual quality comparison**~~ — done; lowres lost.
2. ~~**Find the exact 768×432 ceiling**~~ — moot.
3. ~~**Try 12 frames at 768×432**~~ — moot (resolution direction was wrong).
4. ~~**Lowres on other scenes**~~ — moot.
5. **Try 12 frames at 1280×720** (the OPPOSITE-direction test): if
   higher resolution improves quality at the known-fitting frame count,
   the bottleneck was per-pixel detail. Risk: 1280×720 is 56% more
   pixels than 1024×576 — chart-alignment may OOM at 12 frames at this
   resolution. If it does, drop to 11 or 10 and continue.
6. **B5 — manual frame curation** at the same 12-frame budget. Picking
   12 viewpoint-diverse frames manually from a denser candidate pool
   (e.g., 60 candidate frames spaced every 4 sec across the 3:47 video)
   should give MAtCha better triangulation geometry than 12 evenly-time-
   spaced frames.

## How to invoke

The runner script in `experiments/004-matcha-sky-house/runner.sh` can be
adapted; this experiment used the inline approach since we were
exploring frame counts. Once we settle on the right (frame_count,
resolution) pair, codify into a runner.

Frame extraction parameters used here:

```bash
ffmpeg -i videos/004-sky-house-dining.mp4 \
  -vf "fps=24/227,scale=768:-2" -q:v 2 \
  frames/004-matcha-24-lr/frame_%04d.jpg
```

MAtCha invocation:

```bash
python train.py \
  -s /data/frames/004-matcha-24-lr \
  -o /data/matcha_output/004-sky-house-lr-15 \
  --sfm_config unposed \
  --n_images 15 \
  --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
  --depthanything_encoder vitl
```

Note the source frames dir contains 24 files; `--n_images 15` makes
MAtCha sample 15 of them with constant spacing.

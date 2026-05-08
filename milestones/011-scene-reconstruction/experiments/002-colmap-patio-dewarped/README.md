# Experiment 002 — COLMAP on patio-DEWARPED

**Status:** ❌ failed — dewarp dead-end
**Date:** 2026-04-11
**Pipeline:** COLMAP 3.11.1 (CUDA, Docker)
**Hardware:** Same fleet split as 001
**Reference:** `REPORT-2026-04-11.md`, OLAI corpus `3d-reconstruction/capture-profiles`

## Input

- Same patio scene as experiment 001
- Same DJI Osmo Action 3 (4K HEVC, 30fps hyperlapse, ~31 sec)
- **Difference**: DJI's in-camera dewarp mode enabled (camera produces a
  rectified-perspective output instead of raw fisheye)
- Frames extracted: 944

## Process

Same pipeline as 001 (`run_colmap_sparse.sh`). Two camera models tested
sequentially:

```bash
# Attempt 1
camera_model = SIMPLE_RADIAL
# Attempt 2
camera_model = SIMPLE_RADIAL_FISHEYE
```

## Result

| Camera model | Images registered |
|--------------|-------------------|
| `SIMPLE_RADIAL` | **2 / 944** |
| `SIMPLE_RADIAL_FISHEYE` | **5 / 944** |

Effectively zero registration in both. The mapper cannot find a stable
initial pair from the dewarped frames.

## Output

- `data/scenes/002-patio-dewarped/sparse/` — empty (mapper produced nothing usable)
- `data/scenes/002-patio-dewarped/dense/` — empty
- No mesh

## Quality verdict

**Pipeline failure.** No reconstruction produced. This is not a parameter
tuning issue — the fundamental problem is the input.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| All | ❌ | No output produced |

## Why dewarp fails for COLMAP

Hypothesised in `REPORT-2026-04-11.md` and confirmed by this experiment:

1. **The dewarp crops FOV** — there's substantially less overlap between
   adjacent frames after dewarp, so feature matching has fewer correspondences.
2. **Non-uniform resampling** — the dewarp is computed by warping the
   fisheye source pixels into a perspective grid. The interpolation
   introduces sub-pixel artifacts that destroy SIFT-feature consistency
   across frames.
3. **Lossy compression on top** — DJI's dewarp output is re-encoded with
   HEVC, compounding the resampling artifacts.

## Key finding (drove all subsequent capture protocols)

> **Always capture in the camera's native fisheye mode. Never rely on
> in-camera dewarp.** Let the reconstruction tool handle the distortion
> mathematically.

This is now codified in:
- The OLAI corpus entry `3d-reconstruction/capture-profiles`
- The capture-side guidance in M11's `REPORT-2026-04-11.md`
- Subsequent experiment captures (003, 004) — both shot in native fisheye

## Value of running a "failed" experiment

Worth keeping this folder in the experiments tree because the **negative
result is the milestone-relevant finding**: it told us how to capture for
all subsequent scenes. Without this experiment we'd have likely lost time
debugging "why does my reconstruction look wrong" on later, more
expensive captures.

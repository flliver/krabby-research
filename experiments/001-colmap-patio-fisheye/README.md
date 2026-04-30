# Experiment 001 — COLMAP on patio-fisheye

**Status:** ✅ sparse only, dense MVS deferred
**Date:** 2026-04-11
**Pipeline:** COLMAP 3.11.1 (CUDA, Docker)
**Hardware:** sbeeprz (RTX 4080) for GPU SIFT + match; jbeeprz (Ryzen 7 5800X, 128 GB RAM) for mapper
**Reference:** `REPORT-2026-04-11.md`, OLAI corpus `3d-reconstruction/colmap`, `3d-reconstruction/capture-profiles`

## Input

- Scene: outdoor patio (stamped concrete, wood deck, A-frame house, covered pavilion, trees)
- Capture: DJI Osmo Action 3, 4K HEVC, 30fps **hyperlapse**, ~31 sec
- FOV: native 155° fisheye
- Frames extracted: 942 (full framerate)

## Process

```bash
workspace/run_colmap_sparse.sh 001-patio-fisheye
```

The full T0 pipeline: extract frames → GPU SIFT extraction → exhaustive
matcher → mapper. Camera model: `SIMPLE_RADIAL_FISHEYE` (chosen after
empirical comparison — see `REPORT-2026-04-11.md`).

## Runtime

| Stage | Host | Time |
|-------|------|------|
| GPU SIFT extraction | sbeeprz (RTX 4080) | ~5 min |
| GPU exhaustive match (944² pairs) | sbeeprz (RTX 4080) | ~15 min |
| Sparse mapper | jbeeprz (5800X) | ~72 min |
| **Total** | | **~92 min** |

## Output

- `data/scenes/001-patio-fisheye/sparse/0/` (21 MB) — 942/942 images registered, 381,673 sparse 3D points
- `data/scenes/001-patio-fisheye/dense/` — empty (T1 dense MVS never run)
- No mesh

## Quality verdict

**Sparse reconstruction is excellent.** 100% image registration is the
ideal outcome. Sparse cloud is consistent with the scene structure on
inspection.

**No dense MVS / mesh produced**, so this experiment is not directly
comparable to the MASt3R-SLAM / SLAM3R / MAtCha experiments which
produced surface meshes (or at least dense clouds).

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ❌ — never reached | Would need T1 dense MVS + Poisson conditioning |
| R3 Camera poses | ✅ — COLMAP-format, 100% registration | The strongest pose recovery of any pipeline tested |
| R5 Wall-clock | 🟡 — 92 min just for sparse | Adding T1 dense MVS would push to 2+ hr per scene |
| R7 M11-validated | 🟡 — sparse only | Path to a watertight mesh demonstrated in principle but not driven through |

## Key findings (informed downstream work)

1. **Camera model selection is the #1 variable for COLMAP** — `PINHOLE` / `OPENCV` / `OPENCV_FISHEYE` all fail on 155° FOV (2/63). `SIMPLE_RADIAL_FISHEYE` succeeds (927+/942).
2. **Hyperlapse breaks the sequential matcher** — must use exhaustive matching with GPU acceleration.
3. **GPU SIFT non-determinism**: tbeeprz's RTX 5080 extraction yielded only 6/942 registration; sbeeprz's RTX 4080 extraction yielded 942/942 on the same video. The mapper is deterministic given a fixed database; the variability is in feature extraction.
4. **3D V-Cache CPUs help the mapper** — Zen 5 V-Cache V2 was 21% faster than Zen 4 V-Cache V1, 34% faster than Zen 3 without V-Cache. Mapper is CHOLMOD/SuiteSparse-bound.

## Why we moved on

COLMAP's *sparse* pipeline is fine; the *dense → mesh* path is too slow
and too involved compared to learned methods that produce comparable or
better meshes in a fraction of the time. We kept the COLMAP infrastructure
for ground-truth comparisons but pivoted reconstruction work to MASt3R-SLAM
and MAtCha.

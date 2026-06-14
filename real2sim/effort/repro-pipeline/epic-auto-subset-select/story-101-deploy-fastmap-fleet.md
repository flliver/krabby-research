---
xid: STO-SCN-101
parent: ./epic.md
kind: story
effort: scn
status: in-progress
date: 2026-06-13
depends-on: []
bd-id: krabby-5v9
assignee: krabby
---

# Deploy FastMap GPU SfM to the fleet

## Summary

A `krabby-fastmap` GPU container is built and pushed to the registry so the
pose stage (STO-SCN-093) can solve cameras + emit a co-visibility graph on the
GPU — no CPU-bound COLMAP mapper anywhere.

## Context

STO-SCN-093 needs a solver that produces a **co-visibility / track graph**, and
the operator's standing directive is **GPU-accelerated SfM only** (classic
COLMAP's incremental mapper is CPU-bound and ruled out). FastMap
([pals-ttic](https://github.com/pals-ttic/fastmap), PyTorch, TTIC) is the GPU
SfM that fits — but it is **not deployed on the fleet** (confirmed 2026-06-13:
no registry image, no host install, no binary). This story deploys it; it
**blocks STO-SCN-093**.

## Problem

Provide a reproducible, GPU-end-to-end SfM on the fleet that takes images →
poses + sparse point cloud (COLMAP format), using only GPU compute.

## Design

### Approach

Build `krabby-fastmap` **extending `krabby-da3`** (operator pick: reuse its
proven CUDA 12.8 / torch 2.7.0+cu128 / `sm_89;sm_120` Blackwell base). Add:
- **COLMAP 4.0.4 built with CUDA** — used ONLY for GPU feature extraction +
  matching (`SiftExtraction/SiftMatching.use_gpu 1`). `GUI_ENABLED=OFF`,
  `CGAL_ENABLED=OFF` (headless; no mesher needed).
- **FastMap** (pinned `dafd165`) + the `jiahaoli95/pyrender` fork + custom CUDA
  kernels (`setup.py build_ext`) — GPU pose estimation + sparse triangulation,
  **replacing COLMAP's CPU mapper**.

Pipeline: `colmap feature_extractor → colmap exhaustive_matcher → fastmap
run.py --headless → sparse/0`. Build on a GPU x86 host (tbeeprz / 5080); push to
`j.pski.org:5000/krabby-fastmap`.

### Changes

| File | Change |
|------|--------|
| `images/fastmap/Dockerfile` | new — extends krabby-da3 + CUDA-COLMAP + FastMap |
| `images/fastmap/README.md` | new — build/push/run + 093 open items |
| `images/fastmap/run_fastmap.sh` | new — GPU solve orchestration w/ phase+percent progress (→ MQTT) |
| `real2sim/lib_progress.sh` | timeout-harden the nanny→MQTT backend (best-effort) |

## Definition of Done

- [x] `krabby-fastmap:0.1` builds (COLMAP 4.0.4 CUDA `sm_120` + FastMap kernels
      compiled; `torch 2.7.0+cu128`, `fastmap import OK`) on tbeeprz.
- [x] Image pushed to `j.pski.org:5000/krabby-fastmap:0.1`.
- [x] Smoke test (real precull-300 of 001-patio): GPU extract + GPU exhaustive
      match + FastMap → valid `sparse/0` (300 image poses, ~10.7 MB `points3D`
      with tracks). `RUN_EXIT_0`.
- [x] GPU-only confirmed: COLMAP used only for `--FeatureExtraction/FeatureMatching.use_gpu`;
      FastMap GPU pose+triangulation; no CPU incremental mapper invoked.

## Out of scope

- Wiring FastMap into the v4 graph + the covis-graph extractor + validity gate
  (those are STO-SCN-093).
- Baking the krabby covis-extractor tool into the image (done in the 093 build,
  so `results.json` provenance covers it).
- Fisheye camera-model handling in FastMap config (a 093 concern; noted).

## Findings for STO-SCN-093 (earned during the smoke)

1. **FastMap supports only `SIMPLE_PINHOLE` / `SIMPLE_RADIAL`** — it rejects
   `SIMPLE_RADIAL_FISHEYE` (camera model 8) at database read. Our 155° DJI
   scenes are fisheye → **093 must undistort fisheye → pinhole before FastMap**
   (or route fisheye to a feed-forward solver). The 091 profile already flags
   fisheye vs dewarped, so the dispatch has the signal.
2. **FastMap needs scale.** 12 images → rotation averaging never terminated
   (degenerate graph). 300 images (its design point) → rotation converged and
   the full solve completed. The pre-cull target (≤300) lands in the right band.
3. **GPU-only confirmed end-to-end** — COLMAP GPU SIFT extract + GPU match,
   FastMap GPU pose + triangulation; the CPU incremental mapper is never used.

(The smoke used `SIMPLE_RADIAL` on fisheye frames — it produced a substantial
model, but undistortion is still the correct path for 093 quality.)

## Implementation Notes

- **COLMAP 4.x dep fix:** 4.0.4 replaced FreeImage with **OpenImageIO** — the
  first build failed at cmake until `libopenimageio-dev openimageio-tools`
  (+ `libgmock-dev`, `libsuitesparse-dev`, `libssl-dev`, and the
  `mkdir /usr/include/opencv4` OIIO-cmake workaround) were added. Used OpenBLAS
  instead of the ~2 GB MKL the docs default to. Built clean on sm_120.
- **CLI:** COLMAP 4.x renamed `SiftExtraction/SiftMatching.use_gpu` →
  `FeatureExtraction/FeatureMatching.use_gpu`; matcher progress is
  `pairing.cc: Processing block [i/N, j/M]` (parsed for percent).
- **`run_fastmap.sh` bug found + fixed in the smoke:** FastMap's `run.py`
  refuses a pre-existing `--output_dir` (`FileExistsError`) — the script now
  `rm -rf`s it and lets FastMap create it. Also added **database reuse**
  (skip extract+match if `database.db` exists; `REUSE_DB=0` to force fresh).
- **Progress → MQTT:** `lib_progress.sh` nanny backend hardened with
  `timeout 2 … || log` (best-effort; broker-down/TLS-hang no longer fails the
  job). `run_fastmap.sh` emits 4 phases with live percent (per-image extract,
  per-block match, per-stage FastMap), throttled.
- **Smoke numbers:** precull-300 (001-patio) → `sparse/0` with 300 poses +
  ~10.7 MB `points3D`; full pipeline GPU-only; `RUN_EXIT_0`.

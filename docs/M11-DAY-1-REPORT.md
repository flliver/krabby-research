# M11 Scene Reconstruction — Day 1 Report (2026-04-11)

## Executive Summary

First working day on Milestone 11. Established the full T0 pipeline (video capture → frame extraction → COLMAP sparse reconstruction), built CUDA-enabled infrastructure across 4 outposts, and produced a sparse reconstruction with **942/942 images registered (100%)** — 381,673 3D points from a 31-second DJI Action 3 hyperlapse of an outdoor patio scene.

---

# Part 1: M11 Milestone Progress

## Scope Recap

**Grant:** Patina Research Foundation — Krabby-Uno Milestone 11
**Pipeline:** Phone/action camera video → COLMAP SfM → Dense MVS → Mesh conditioning → USD → IsaacSim
**Goal:** Reconstructed collision-quality environments for hexapod locomotion evaluation

## What Was Accomplished

### Research & Planning
- Identified the M11 project from workspace structure and grant spec
- Recovered Fletcher meeting context from Firefox history (2026-03-23 browsing session)
- Downloaded reference papers: SLAM3R (arXiv 2412.09401) and Holosoma (arXiv 2512.01996)
- Documented key URLs and their relationships to milestone tasks in `REFERENCES.md`
- Read SLAM3R paper architecture — understood as viable fallback pipeline

### T0: Scene Capture & Sparse Reconstruction
- **Captured 2 videos** with DJI Osmo Action 3 (4K HEVC, 30fps hyperlapse, ~31 sec each)
  - `001-patio-fisheye` — native 155° FOV (fisheye)
  - `002-patio-dewarped` — in-camera dewarp enabled
- Scene: outdoor patio — stamped concrete, wood deck, A-frame house, covered pavilion, trees
- Extracted frames at full framerate (942-944 frames per video)
- **Final result: 942/942 images registered (100%)** — 381,673 3D points, mean track length 4.55, reprojection error 1.38px

### Pipeline Scripts Built
| Script | Purpose |
|--------|---------|
| `extract_frames.sh` | Video → JPEG frames (configurable fps) |
| `run_colmap_sparse.sh` | Full T0 pipeline (extract + match + map) |
| `run_colmap_match_only.sh` | GPU-only: extract + match (for split compute) |
| `run_colmap_map_only.sh` | CPU-only: mapper (runs on high-RAM host) |
| `run_colmap_dense.sh` | T1: MVS dense reconstruction (drafted, not yet run) |
| `run_mesh_conditioning.sh` | T2: Mesh cleanup + collision proxy (drafted, not yet run) |

### Not Yet Started
- T1: Dense MVS reconstruction (script ready, awaiting sparse model finalization)
- T2: Mesh conditioning & USD export (script ready)
- T3: Locomotion model integration (Docker + Extreme Parkour + Holosoma)
- T4: Hexapod adaptation (reward shaping, 18 DOF)

## Key Findings for M11

### Camera Model is Critical
The grant spec recommended `PINHOLE` camera model. This is wrong for the DJI Action 3.

| Camera Model | Registration Rate | Notes |
|-------------|-------------------|-------|
| `PINHOLE` | 0/63 | Total failure |
| `OPENCV` | 2/63 | Insufficient distortion modeling |
| `OPENCV_FISHEYE` | 2/63 | Too many params, mapper can't grow |
| **`SIMPLE_RADIAL_FISHEYE`** | **927+/942** | Correct choice — minimal params, stable init |
| `SIMPLE_RADIAL` (dewarped) | 2/944 | Dewarped video doesn't work with COLMAP |

**Recommendation for spec:** Update T0 to specify `SIMPLE_RADIAL_FISHEYE` for action cameras. The DJI Action 3's 155° FOV is true fisheye territory.

### Dewarped Video is a Dead End
DJI's in-camera dewarp mode produces video that COLMAP cannot reconstruct, regardless of camera model tried (`SIMPLE_RADIAL`: 2/944, `SIMPLE_RADIAL_FISHEYE`: 5/944). The dewarp likely crops FOV and introduces artifacts that destroy feature matching quality.

**Recommendation:** Always capture in native/fisheye mode. Let COLMAP handle the distortion mathematically.

### Hyperlapse is Viable but Requires Exhaustive Matching
The videos are hyperlapses (~10x smaller than normal video). This is a significant advantage for transfer efficiency but breaks COLMAP's sequential matcher, which assumes temporal adjacency = visual adjacency.

| Matching Strategy | Result | Why |
|-------------------|--------|-----|
| Sequential (overlap 10) | 2/94 | Adjacent frames too similar within window |
| Sequential (overlap 15) | 0/944 | Same problem at full framerate |
| **Exhaustive** | **927+/942** | Finds all cross-temporal matches |

**Recommendation:** Use exhaustive matching for hyperlapse video. This requires GPU acceleration (CPU exhaustive on 944 frames would take hours). The tradeoff is correct: burn GPU cycles on matching rather than bandwidth on 10x larger normal video files.

### Grant Spec Adjustments Needed
1. **Camera model:** `PINHOLE` → `SIMPLE_RADIAL_FISHEYE` for action cameras
2. **Input format:** Spec says photos with 60% overlap. Video (especially hyperlapse) works well with exhaustive matching. Consider updating capture guidelines.
3. **COLMAP version:** 3.11.1 produces dramatically better results than 3.10 on the same data (mapper algorithm improvements). Pin the version.
4. **Scale calibration:** Not yet addressed. Monocular reconstruction has no absolute scale. Need known-size reference object in next captures.

---

# Part 2: Infrastructure & Processing

## Fleet Overview

| Host | CPU | RAM | GPU | COLMAP | Role |
|------|-----|-----|-----|--------|------|
| **sbeeprz** | Ryzen 7 7800X3D (Zen 4, 96MB 3D V-Cache) | 32 GB | RTX 4080 16 GB | 3.11.1 CUDA (Docker) | GPU extract + match |
| **dbeeprz** | Ryzen 7 7800X3D (Zen 4, 96MB 3D V-Cache) | 32 GB | RTX 4080 16 GB | 3.11.1 CUDA (Docker) | GPU extract + match |
| **tbeeprz** | Ryzen 7 9800X3D (Zen 5, 96MB 3D V-Cache) | 32 GB | RTX 5080 16 GB | 3.11.1 CUDA (Docker) | GPU extract + match (fastest) |
| **jbeeprz** | Ryzen 7 5800X (Zen 3, 32MB L3) | 128 GB | RX 6900 XT 16 GB (no ROCm) | 3.11.1 CPU (native) | CPU mapper (most RAM) |
| **jdp-mac** | Apple M1 Ultra (16P+4E) | 128 GB | Apple 64-core (Metal) | 4.0.3 CPU (Homebrew) | Backup mapper, local dev |

All Linux hosts: Debian 13 (Trixie), on same 192.168.0.x LAN (gigabit, ~100 MB/s inter-host).

## Processing Benchmarks (944 frames, 4K)

### Feature Extraction
| Machine | GPU | Time |
|---------|-----|------|
| sbeeprz/dbeeprz | RTX 4080 (CPU fallback) | ~15 min |
| sbeeprz/dbeeprz | RTX 4080 (CUDA) | **35 sec** |
| tbeeprz | RTX 5080 (CUDA) | **25 sec** |

GPU SIFT is **25-40x faster** than CPU. RTX 5080 is ~30% faster than RTX 4080.

### Exhaustive Matching (944 frames = ~445K pairs)
| Machine | GPU | Time |
|---------|-----|------|
| dbeeprz | RTX 4080 (CUDA) | 22 min |
| tbeeprz | RTX 5080 (CUDA) | ~15 min (estimated from block rate) |
| CPU | — | Impractical (hours) |

### Mapper (CPU-bound, 942 images → 381K points, same database)
| Machine | CPU | L3 Cache | Time | vs fastest |
|---------|-----|----------|------|------------|
| **tbeeprz** | **9800X3D (Zen 5)** | **96 MB V-Cache V2** | **48 min** | baseline |
| sbeeprz | 7800X3D (Zen 4) | 96 MB V-Cache V1 | 61 min | +27% |
| jbeeprz | 5800X (Zen 3) | 32 MB | 73 min | +52% |

All three produced identical results: 942/942 registered, ~381K points, 1.38px reprojection error. Pure CPU speed difference.

The mapper uses ~500-600% CPU (4-6 cores). It's parallelized within each Ceres BA iteration but the iterations are sequential. 3D V-Cache provides a clear advantage — Zen 5 V-Cache V2 is 21% faster than Zen 4 V-Cache V1, and 34% faster than Zen 3 without V-Cache. The sparse matrix factorization in CHOLMOD/SuiteSparse is cache-bound, which is exactly what V-Cache is designed for.

**Note on reproducibility:** tbeeprz's own GPU extraction (RTX 5080) produced a database that only registered 6/942 images. When given sbeeprz's database (RTX 4080 extraction), it registered 942/942. The mapper result is deterministic given the same database, but different GPU SIFT implementations produce different feature sets that can lead to different (better or worse) initial pair selection. This is a known COLMAP behavior — the initial pair is critical and small feature differences can cascade.

## Data Transfer Lessons

| Method | Speed | Use Case |
|--------|-------|----------|
| rsync (individual files, WAN) | ~1.2 MB/s | Painfully slow for 944 files |
| tar pipe (WAN) | ~3-4 MB/s total | Better, but still WAN-limited |
| **SCP video file (WAN)** | ~3-4 MB/s | **Best WAN approach** — 349 MB vs 1.7 GB frames |
| **LAN inter-outpost** | ~100 MB/s | **16 sec for 1.7 GB** — always use this |

**Key insight:** Transfer the compressed video (HEVC, ~349 MB), extract frames on the remote GPU host. This is 5x less data than transferring extracted JPEG frames (1.7 GB). Combined with LAN transfers between outposts, data movement becomes negligible.

## Optimal Pipeline Architecture

```
jdp-mac (capture)
    │
    │  SCP video (349 MB, ~6 min WAN)
    ▼
GPU outpost (sbeeprz/dbeeprz/tbeeprz)
    │  ffmpeg extract → GPU SIFT → GPU exhaustive match
    │  (~20 min total on RTX 4080, ~15 min on RTX 5080)
    │
    │  SCP database.db (1.7 GB, ~16 sec LAN)
    ▼
CPU outpost (jbeeprz, 128 GB RAM)
    │  COLMAP mapper (~30-60 min depending on scene)
    │
    │  SCP sparse model (small, instant LAN)
    ▼
GPU outpost
    │  Dense MVS (GPU, patch_match_stereo)
    │  → fused.ply
    ▼
Any host
    │  Mesh conditioning (Open3D/trimesh, CPU)
    │  → conditioned OBJ + collision proxy
    ▼
IsaacSim host
    │  USD conversion + import
    ▼
    Done
```

## What Worked
1. **Hyperlapse capture** — 10x smaller files, adequate frame density with exhaustive matching
2. **`SIMPLE_RADIAL_FISHEYE`** — correct camera model for DJI Action 3
3. **GPU COLMAP from source** — 25-40x speedup on extract, enables exhaustive matching
4. **Split compute** — GPU extract+match on one host, CPU mapper on another
5. **LAN inter-outpost transfers** — near-instant data movement
6. **Video-first transfer** — send compressed video, extract remotely
7. **Outpost provisioning system** — consistent deployment across fleet

## What Didn't Work
1. **Dewarped video** — dead end for COLMAP, any camera model
2. **`PINHOLE` / `OPENCV` camera models** — wrong for 155° FOV
3. **Sequential matcher on hyperlapse** — zero matches, total failure
4. **Debian packaged COLMAP** — 3.10 vs 3.11.1 mapper regression, no CUDA
5. **Transferring extracted frames** — 5x more data than compressed video
6. **WAN transfers of large datasets** — ~3 MB/s bottleneck
7. **COLMAP version mismatch** — 3.10 mapper fails on 3.11.1 database (only 3/942 registered)

## Lessons Learned
1. **Camera model selection is the #1 variable** — wrong model = total failure, right model = 98%+ registration
2. **Hyperlapse + exhaustive matching is a valid workflow** — trades GPU compute for bandwidth savings
3. **Always build COLMAP from source** — Debian packages lag and lack CUDA
4. **Pin COLMAP version across all hosts** — mapper algorithm changes between minor versions break cross-host workflows
5. **Transfer video, extract remotely** — obvious in retrospect, 5x bandwidth savings
6. **Use LAN between outposts** — 100 MB/s vs 3 MB/s WAN
7. **The mapper is the bottleneck** — GPU handles extract+match in minutes, mapper grinds for an hour. 3D V-Cache CPUs help.
8. **DJI dewarp destroys reconstruction** — always capture native/raw from action cameras

## Docker Shared Memory Fix for MASt3R-SLAM

MASt3R-SLAM uses Python multiprocessing with shared CUDA tensors. Docker's default shared memory (`/dev/shm`) is **64 MB**, which is insufficient. The process silently deadlocks at startup — 0% GPU utilization, no output, no error message. This wasted multiple debugging hours across several attempts.

**Fix:** Add `--shm-size=8g` to `docker run`:
```bash
docker run --rm --gpus all --shm-size=8g ...
```

**Symptoms:** Container starts, prints config, then hangs forever. GPU at 0%, CPU near 0%. No error message. No crash. Just silence.

**Confirmed working:** With `--shm-size=8g`, GPU immediately jumps to 74% utilization, 8.3 GB VRAM allocated.

**Applies to:** Any PyTorch multiprocessing workload in Docker (not just MASt3R-SLAM). SLAM3R did not hit this because it uses a single-process feed-forward architecture.

**References:**
- [MASt3R-SLAM Issue #94](https://github.com/rmurai0610/MASt3R-SLAM/issues/94)
- [MASt3R-SLAM Issue #73](https://github.com/rmurai0610/MASt3R-SLAM/issues/73)

## Next Steps
1. **Finalize sparse model** — sbeeprz run completing (927+/942)
2. **Run T1 dense MVS** — `run_colmap_dense.sh` on GPU outpost (patch_match_stereo)
3. **Run T2 mesh conditioning** — `run_mesh_conditioning.sh` (Poisson reconstruction, cleanup, collision proxy)
4. **Capture 1-2 more scenes** — different environments for M11 acceptance (2-3 scenes required)
5. **Scale calibration** — add known-size reference object to captures
6. **Consider SLAM3R** — if COLMAP proves too slow for rapid iteration, SLAM3R does everything on GPU in real-time
7. **Set up ROCm on jbeeprz** — the RX 6900 XT is idle, could run PyTorch/SLAM3R workloads

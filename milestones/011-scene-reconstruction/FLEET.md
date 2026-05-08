# Fleet Overview — M11 Scene Reconstruction

Last updated: 2026-04-29

## Hardware support matrix

### What the project specs actually require

| Source | Hardware specified |
|--------|-------------------|
| M11 grant `OVERVIEW.md` | None — RTX 4090 mentioned only as a paper benchmark, not a requirement |
| M2 grant `OVERVIEW.md` | "16 GB NVIDIA RTX 5080, 32 GB RAM, Core i7 (or equivalent)" |
| `research/DEVELOPER.md` | "Recommended configuration: RTX 5080 on Ubuntu 24.04, CUDA 13.0, PyTorch 2.7.0+cu130" |

The committed reference platform is **RTX 5080**. Note: `DEVELOPER.md`'s
PyTorch 2.7.0+cu130 combination is unverified for sm_120 — stable cu130 doesn't
ship Blackwell kernels (cu128 does). May need to flag this back to the doc author.

### What we actually run on

Jeremy's dev fleet has both Ada (4080) and Blackwell (5080) cards. We target both
so any host can run any reconstruction job:

| GPU | Architecture | sm code | Hosts | Project status |
|-----|-------------|---------|-------|----------------|
| RTX 4080 | Ada Lovelace | sm_89 | sbeeprz, dbeeprz | Dev hardware (not in spec) |
| RTX 5080 | Blackwell | sm_120 | tbeeprz, bbeeprz | **Reference platform per M2 + DEVELOPER.md** |

### What the multi-arch container needs

| Component | Constraint | Source |
|-----------|-----------|--------|
| Base image | `nvidia/cuda:12.8-devel` | sm_120 requires CUDA 12.8+ runtime |
| PyTorch | `2.7.0+cu128` (stable) | First stable release with prebuilt sm_120 kernels |
| `TORCH_CUDA_ARCH_LIST` | `"8.9;12.0"` | Source-compiled extensions need both archs |
| curope source patch | `.type()` → `.scalar_type()` | Required for PyTorch ≥ 2.6 API |

See `docker/MAST3R-NOTES.md` for the full lesson catalog (11 distinct constraints discovered).

## Network Topology

```
                    WAN (~6 Mbps upload)
    jdp-mac ─────────────────────────────── LAN (Gigabit, ~100 MB/s)
    (local)                                  │
                                    ┌────────┼────────┬────────┐
                                    │        │        │        │
                                 sbeeprz  dbeeprz  jbeeprz  tbeeprz
                                 .178      .037     .002     .160
                                 (♆)       (☉)      (♃)      (♂)
```

All outposts: 192.168.0.x, Debian 13 (Trixie), same physical network.

## Hardware

| | **sbeeprz (♆)** | **dbeeprz (☉)** | **tbeeprz (♂)** | **jbeeprz (♃)** | **jdp-mac** |
|---|---|---|---|---|---|
| **CPU** | Ryzen 7 7800X3D | Ryzen 7 7800X3D | Ryzen 7 9800X3D | Ryzen 7 5800X | Apple M1 Ultra |
| **Architecture** | Zen 4 | Zen 4 | Zen 5 | Zen 3 | Apple Silicon |
| **Cores/Threads** | 8C/16T | 8C/16T | 8C/16T | 8C/16T | 16P+4E/20T |
| **Max Clock** | 5.05 GHz | 5.05 GHz | 5.27 GHz | 4.85 GHz | ~3.2 GHz |
| **L3 Cache** | 96 MB (3D V-Cache) | 96 MB (3D V-Cache) | 96 MB (3D V-Cache V2) | 32 MB | ~48 MB (SLC) |
| **RAM** | 32 GB | 32 GB | 32 GB | 128 GB | 128 GB |
| **GPU** | RTX 4080 16 GB | RTX 4080 16 GB | RTX 5080 16 GB | RX 6900 XT 16 GB | Apple 64-core |
| **GPU Compute** | CUDA (Ada SM89) | CUDA (Ada SM89) | CUDA (Blackwell SM120) | ROCm (not installed) | Metal (no CUDA) |
| **OS** | Debian 13.4 (Trixie) | Debian 13.4 (Trixie) | Debian 13.3 (Trixie) | Debian 13.4 (Trixie) | macOS 15.6.1 |
| **NVIDIA Driver** | 595.58.03 | 595.58.03 | 590.48.01 | N/A (AMD) | N/A (Apple) |
| **CUDA** | 12.8.0 (Docker) | 12.8.0 (Docker) | 12.8.0 (Docker) | N/A | N/A |
| **Disk** | 458 GB (384 free) | 458 GB (385 free) | 458 GB (420 free) | 458 GB (420 free) | 1.8 TB (26 free) |
| **Docker** | 29.3.1 | 29.4.0 | docker.io | 29.3.1 | N/A |
| **COLMAP** | 3.11.1 CUDA (Docker) | 3.11.1 CUDA (Docker) | 3.11.1 CUDA (Docker) | 3.11.1 CPU (native) | 4.0.3 CPU (Homebrew) |
| **SLAM3R** | — | Building | — | — | — |
| **LAN IP** | 192.168.0.178 | 192.168.0.37 | 192.168.0.160 | 192.168.0.2 | WAN only |

**bbeeprz** is offline. tbeeprz serves as its stand-in during testing.

## Measured Benchmarks (M11 workloads)

### GPU: SIFT Feature Extraction (942 frames, 4K)
| Machine | GPU | Time |
|---------|-----|------|
| sbeeprz/dbeeprz | RTX 4080 (CPU fallback) | ~15 min |
| sbeeprz/dbeeprz | RTX 4080 (CUDA) | 36 sec |
| tbeeprz | RTX 5080 (CUDA) | **26 sec** |

### GPU: Exhaustive Matching (942 frames, ~445K pairs)
| Machine | GPU | Time |
|---------|-----|------|
| sbeeprz | RTX 4080 (CUDA) | 19 min |
| tbeeprz | RTX 5080 (CUDA) | **15 min** |

### CPU: COLMAP Mapper (942 images, same database, 381K points)
| Machine | CPU | L3 Cache | Time | vs fastest |
|---------|-----|----------|------|------------|
| tbeeprz | 9800X3D (Zen 5) | 96 MB V-Cache V2 | **48 min** | baseline |
| sbeeprz | 7800X3D (Zen 4) | 96 MB V-Cache V1 | 61 min | +27% |
| jbeeprz | 5800X (Zen 3) | 32 MB | 73 min | +52% |

### Data Transfer
| Path | Method | Rate | Notes |
|------|--------|------|-------|
| jdp-mac → outpost (WAN) | scp/rsync (no -z) | ~700 KB/s | Bottleneck for large files |
| jdp-mac → outpost (WAN) | rsync -z (compressed) | ~240 KB/s | WORSE — compressing HEVC wastes CPU |
| outpost ↔ outpost (LAN) | scp | ~100 MB/s | 4.3 GB in 43 sec |

## Role Assignment

| Role | Best Machine | Why | Backup |
|------|-------------|-----|--------|
| **Video capture** | jdp-mac (local) | Camera connected here | — |
| **GPU extract + match** | tbeeprz | RTX 5080, fastest GPU | sbeeprz/dbeeprz (RTX 4080) |
| **COLMAP mapper** | tbeeprz | 9800X3D, fastest CPU for BA | jbeeprz (128 GB RAM for huge scenes) |
| **Dense MVS (T1)** | tbeeprz | RTX 5080, patch_match_stereo | sbeeprz/dbeeprz |
| **SLAM3R** | dbeeprz | Building container now | sbeeprz |
| **Mesh processing (T2)** | Any | CPU-bound, Open3D/trimesh | — |
| **Large scene mapper** | jbeeprz | 128 GB RAM for 10K+ image scenes | jdp-mac (128 GB) |

## Known Issues

### GPU SIFT Non-Determinism
Different NVIDIA GPUs produce different SIFT features. This causes the COLMAP mapper to select different initial image pairs, which can cascade into reconstruction failure. Observed:
- RTX 4080 database → 942/942 registered (patio), 24/3310 (firepit)
- RTX 5080 database → 6/942 registered (patio), 1407+/3310 (firepit, in progress)

**Same database on different CPUs produces identical results.** The non-determinism is in GPU SIFT, not the mapper.

**Mitigation:** Run extract+match on a single GPU, verify result, then ship the database to any CPU for mapping.

### COLMAP Version Sensitivity
- COLMAP 3.10 (Debian package) fails on databases created by 3.11.1 — only 3/942 registered
- Must pin COLMAP version across all hosts
- Always build from source (tag 3.11.1) rather than using distro packages

### WAN Transfer Bottleneck
- ~6 Mbps upload from jdp-mac
- 4.2 GB video = ~100 min transfer
- **Never use `rsync -z` on compressed video** — it's 3x slower than uncompressed transfer
- **Always transfer video, extract remotely** — 349 MB video vs 1.7 GB extracted frames
- **Use LAN inter-outpost transfers** when data needs to be on multiple hosts

### Dewarped Video
DJI Action 3 dewarp mode produces video that COLMAP cannot reconstruct regardless of camera model. Always capture in native fisheye mode.

## Outpost Provisioning

```bash
# Provision a new outpost
AI/outposts/provision.sh <user@host> --milestone 011-scene-reconstruction

# Transfer video (WAN → first outpost)
scp video.mp4 <outpost>:~/outposts/krabby/data/011-scene-reconstruction/videos/

# Distribute video (LAN between outposts, ~100 MB/s)
ssh <source> "scp ~/.../<video> jeremy@<dest>:~/.../<video>"

# Run COLMAP (CUDA container)
ssh <outpost> "docker run --rm --gpus all \
  -v ~/outposts/krabby/data/011-scene-reconstruction:/data \
  -v ~/outposts/krabby/workspace/milestones/011-scene-reconstruction/workspace:/workspace \
  krabby-011-scene-reconstruction-cuda \
  bash /workspace/run_colmap_sparse.sh <scene> SIMPLE_RADIAL_FISHEYE"
```

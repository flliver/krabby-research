# scene-reconstruction-base

**Migration target:** `research/images/scene-reconstruction-base/`

## Purpose

Base container for the M11 scene-reconstruction pipeline. Provides
**COLMAP** (source-built with CUDA support for sm_89 + sm_120) plus
**Open3D / pymeshlab / trimesh** for downstream mesh processing.

This is the canonical T0/T1 base per the M11 grant (`OVERVIEW.md`
specifies COLMAP MVS + Poisson). Used directly for the COLMAP path; the
substituted pipelines (matcha, mast3r, slam3r, vggt) build on different
bases per their own requirements.

## Base image

`nvidia/cuda:12.8.0-devel-ubuntu24.04`

## Build

```bash
docker build -f images/scene-reconstruction-base/Dockerfile \
  -t krabby-scene-reconstruction-base \
  images/scene-reconstruction-base/
```

(Or via Makefile: `make build-scene-reconstruction-base-image` — see M-3.2.)

## Run

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
  --env-file .env \
  krabby-scene-reconstruction-base \
  bash -c '
    colmap feature_extractor \
      --database_path /data/scenes/<scene>/database.db \
      --image_path /data/scenes/<scene>/images \
      --ImageReader.camera_model SIMPLE_RADIAL_FISHEYE
    # ... see workspace/run_colmap_*.sh for the full T0/T1 sequence
  '
```

Mandatory PyTorch flags per
`research/docs/DOCKER_DEPENDENCIES.md` — without `--shm-size=8g` PyTorch
silently deadlocks at 0% GPU.

## Runtime requirements

- NVIDIA GPU with CUDA 12.8+ (Ada sm_89 or Blackwell sm_120 verified)
- NVIDIA Container Toolkit installed (run `scripts/setup-docker-gpu.sh` once per host)
- HF_TOKEN in environment if downloading any HuggingFace assets (not strictly required for COLMAP itself)

## Lessons learned

- **Source-built COLMAP** is required because no NVIDIA-published wheel
  ships with CUDA enabled for both sm_89 and sm_120.
- `CMAKE_CUDA_ARCHITECTURES="89;100"` covers both Ada and Blackwell.
  (sm_100 here is the Blackwell consumer variant; sm_120 in driver naming.)
- Ubuntu 24.04 enforces PEP 668 (externally-managed-environment); pip
  installs use `--break-system-packages` rather than a separate venv.

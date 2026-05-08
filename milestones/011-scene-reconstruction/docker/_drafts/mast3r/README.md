# mast3r

**Migration target:** `research/images/mast3r/`

## Purpose

MASt3R-SLAM container — alternative T0 sparse-reconstruction pipeline
listed in M11 grant Appendix A. Produces COLMAP-compatible camera poses
and a dense pointmap from monocular RGB video at 20+ FPS via a
two-hierarchy neural network (I2P local + L2W global).

Used during M11 development as a faster alternative to the COLMAP SfM
path; outputs land in COLMAP-format directories and feed into the same
downstream T1 mesh-extraction pipeline.

Reference paper: arXiv 2412.09401 (MASt3R-SLAM, CVPR 2025).

## Base image

`nvcr.io/nvidia/pytorch:25.10-py3` — same x86 base as the existing
`research/images/testing/x86/` container, providing PyTorch 2.9 + CUDA
13.0 with prebuilt kernels for both Ada (sm_89) and Blackwell (sm_120).

This **shares the project base-image strategy** documented in
`research/docs/DOCKER_DEPENDENCIES.md` rather than rolling a new base.

## Build

```bash
docker build -f images/mast3r/Dockerfile \
  -t krabby-mast3r \
  images/mast3r/
```

(Or `make build-mast3r-image` — added in M-3.2.)

The image is large (~36 GB). For multi-host distribution, use
`docker save | docker load` over the LAN — see
`docker/MAST3R-NOTES.md`.

## Run

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
  --env-file .env \
  krabby-mast3r \
  bash -c '
    cd /opt/MASt3R-SLAM
    python main.py \
      --dataset /data/videos/<scene>.mp4 \
      --config config/base.yaml \
      --no-viz \
      --save-as /data/scenes/<scene>/mast3r_output/<name>
  '
```

There's also a workspace-level wrapper that handles logging and
backgrounding: `workspace/run_mast3r.sh <scene> [video]`.

## Runtime requirements

- NVIDIA GPU with CUDA 12.8+ (sm_89 or sm_120 verified)
- HF_TOKEN in `.env` (MASt3R checkpoints downloaded at build time from
  Naver Labs Europe — see Dockerfile lines 73–76)
- ≥ 16 GB VRAM
- For the pose-extension pipeline (`localize_reference_image.py`),
  ~ 13-frame SfM runs in ~104s on RTX 5080.

## Lessons learned (highlights — see `docker/MAST3R-NOTES.md` for full 11)

- `--shm-size=8g` is **required at run time** (not only at build time).
  Without it, container starts, prints config, and silently deadlocks
  at 0% GPU. See [MASt3R-SLAM Issue #94](https://github.com/rmurai0610/MASt3R-SLAM/issues/94).
- CUDA 13 dropped sm_60/61/70; the upstream MASt3R-SLAM hardcoded
  arch list (`sm_60–sm_86`) won't compile under our NGC PyTorch 25.10
  base. New list: `sm_75;sm_80;sm_86;sm_89;sm_120`. Patched via
  `patches/patch_mast3r_setup.py`.
- `lietorch` builds from source; CMakeLists looks for headers at
  `eigen/Eigen` so the Dockerfile symlinks `/usr/include/eigen3/Eigen`.
- 4 patches applied at build time:
  - `patch_curope.py` — PyTorch 2.6+ `.type → .scalar_type`
  - `patch_mast3r_setup.py` — modern arch list + build-time CUDA check
  - `patch_dataloader.py` — accept `.jpg` + soft pyrealsense2 import
  - `patch_torch_load.py` — PyTorch 2.6+ `weights_only=True` rejects
    MASt3R checkpoints (they contain `argparse.Namespace`)

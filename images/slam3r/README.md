# slam3r

**Migration target:** `research/images/slam3r/`

## Purpose

SLAM3R container — alternative T0/T1 pipeline per M11 grant Appendix A.
Real-time dense scene reconstruction from monocular RGB video via a
two-hierarchy neural network (I2P local + L2W global) producing a
dense pointmap at 20+ FPS. Tagged CVPR 2025 Highlight.

This container was built as a **fallback** during M11 evaluation. The
project ultimately selected MAtCha (T1 watertight meshes) and
MASt3R-SLAM (T0 sparse poses) as the primary pipelines; SLAM3R remains
available for re-evaluation if the primary path runs into issues.

Reference paper: arXiv 2412.09401.

## Base image

`nvidia/cuda:12.8.0-devel-ubuntu24.04`

Python 3.11 venv at `/opt/slam3r` (added via deadsnakes PPA on top of
the base image's default Python 3.12).

## Build

```bash
docker build -f images/slam3r/Dockerfile \
  -t krabby-slam3r \
  images/slam3r/
```

(Or `make build-slam3r-image` — added in M-3.2.)

Build pulls from upstream `https://github.com/PKU-VCL-3DV/SLAM3R`. As
of migration, no upstream SHA is pinned; this is a **reproducibility
gap** (see TODO in `requirements.txt` notes). Consider pinning before
landing in research/main.

## Run

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
  --env-file .env \
  krabby-slam3r \
  bash -c '
    source /opt/slam3r/bin/activate
    cd /opt/SLAM3R
    # ... see SLAM3R upstream README for invocation
  '
```

## Runtime requirements

- NVIDIA GPU with CUDA 12.4+ (PyTorch wheel pinned to cu124)
- HF_TOKEN in `.env` (model weights are HuggingFace-hosted)
- ≥ 16 GB VRAM

## Lessons learned

- PyTorch pin: `torch==2.5.0 + torchvision==0.20.0 + torchaudio==2.5.0`
  on cu124 (not cu128/cu130) is what upstream SLAM3R was tested
  against. Newer combinations may break.
- CuRoPE extension built in-place from `slam3r/pos_embed/curope` —
  fails-soft (`|| true`) since some platforms hit build issues.

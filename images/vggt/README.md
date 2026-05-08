# vggt

**Migration target:** `research/images/vggt/`

## Purpose

VGGT (Visual Geometry Grounded Transformer) container — alternative T0
sparse-reconstruction pipeline per M11 grant Appendix A. CVPR 2025 Best
Paper, feed-forward 3D reconstruction (no iterative SfM). Outputs land
in COLMAP-format `sparse/` directories.

Like SLAM3R, this container was built during M11 pipeline evaluation
and remains available for re-evaluation. The primary M11 path uses
MASt3R-SfM (T0) + MAtCha TSDF (T1).

## Base image

`nvidia/cuda:12.4.0-devel-ubuntu22.04`

Python 3.11 venv at `/opt/vggt`.

## Build

```bash
docker build -f images/vggt/Dockerfile \
  -t krabby-vggt \
  images/vggt/
```

(Or `make build-vggt-image` — added in M-3.2.)

The build pre-downloads the `facebook/VGGT-1B` HuggingFace model (~2 GB)
inside the image. HF_TOKEN may be required if the model is gated; pass
via `--env-file` at build time. Build pulls from
`https://github.com/facebookresearch/vggt`; no SHA pinned — same
reproducibility caveat as slam3r.

## Run

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
  --env-file .env \
  krabby-vggt \
  bash -c '
    source /opt/vggt/bin/activate
    cd /opt/VGGT
    python demo_colmap.py \
      --scene_dir /data/scenes/<scene> \
      --use_ba \
      --max_query_pts 2048 \
      --query_frame_num 5
  '
```

There's a workspace-level wrapper at `workspace/run_vggt.sh`. The
wrapper assumes the container is already running (relies on
`/opt/VGGT` paths) and is intended for `docker exec`.

## Runtime requirements

- NVIDIA GPU with CUDA 12.4+
- ≥ 16 GB VRAM (VGGT uses reduced query params for 16 GB cards:
  `--max_query_pts 2048 --query_frame_num 5`)
- HF_TOKEN in `.env` if the gated model requires authentication

## Lessons learned

- VGGT's `requirements.txt` and `requirements_demo.txt` come from the
  cloned repo; not pinned in this image's `requirements.txt`.
- The model download at build time tightly couples the image to
  HuggingFace availability. If that becomes a problem, move the
  download to runtime via `--env-file` HF_TOKEN.

# matcha

**Migration target:** `research/images/matcha/`

## Purpose

MAtCha (Atlas of Charts) container — the **primary** reconstruction
pipeline selected for M11 over the grant-canonical COLMAP MVS + Poisson.
MAtCha was the only candidate during the M11 pipeline evaluation that
reliably produced **watertight meshes end-to-end**, satisfying the T1
acceptance criterion natively (TSDF + adaptive-tetrahedralization paths).

Tool substitution from grant canon is defensible per grant Appendix A
(which explicitly lists alternatives). Disclosure tracked in bead
**R2** ("T0/T1 Tool-Substitution Disclosure").

## Base image

`nvidia/cuda:12.8.0-devel-ubuntu24.04`

## Build

```bash
docker build -f images/matcha/Dockerfile \
  -t krabby-matcha \
  images/matcha/
```

(Or `make build-matcha-image` — added in M-3.2.)

The build is heavy (~30+ minutes on RTX 5080). It applies **8 patches**
discovered during the port from MAtCha's official PyTorch 2.0.1+cu118
stack to PyTorch 2.7.0+cu128 (required for sm_120 Blackwell support).
See `docker/MATCHA-NOTES.md` for the full backstory.

## Run

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v <data-dir>:/data \
  --env-file .env \
  krabby-matcha \
  bash -c '
    source /opt/matcha/bin/activate
    cd /opt/MAtCha
    python train.py \
      -s /data/frames/<scene>-matcha-24 \
      -o /data/matcha_output/<scene> \
      --sfm_config unposed \
      --n_images 24 \
      --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
      --depthanything_encoder large
  '
```

Without `--shm-size=8g`, the container silently deadlocks at 0% GPU
during PyTorch multiprocessing setup. The other three flags are
NVIDIA's official recommendations.

## Runtime requirements

- NVIDIA GPU with CUDA 12.8+
- ≥ 16 GB VRAM (validated on RTX 5080)
- ≥ 32 GB RAM recommended
- HF_TOKEN in `.env` (Depth-Anything-V2 + MAtCha checkpoints are
  HuggingFace-hosted)
- The MAtCha image bakes ~4.3 GB of model checkpoints in at build time
  (depth + reconstruction); no checkpoint download at run time.

## Configuration knobs

Most knobs live in MAtCha's `train.py --help`. The most-tweaked options
during M11 work:

| Knob | Default we use | Purpose |
|---|---|---|
| `--sfm_config` | `unposed` | Strong-config alignment (vs `default`) — required to produce clean geometry |
| `--n_images` | varies (12, 24) | Number of source frames to use; trades VRAM for quality |
| `--depthanything_encoder` | `large` | Use the large Depth-Anything-V2 encoder |

See `journal/journals/m11-scene-reconstruction/threads/matcha-quality/notes/`
for the running history of parameter discoveries.

## Lessons learned (highlights — see `docker/MATCHA-NOTES.md` for full)

- `xformers` is **off-limits** — it pulls torch 2.11 nightly which
  breaks pytorch3d's compiled `_C.so` (ABI mismatch). MAtCha runs fine
  without it.
- `faiss-gpu-cu12 1.14.1` lacks sm_120 kernels — use `faiss-cpu`.
- `pytorch3d 0.7.8` must be source-built for torch 2.7 (no wheels).
- 8 patches required to make the port to torch 2.7+cu128 actually
  compile and run. They're applied at build time via the
  `patches/patch_matcha_*.py` scripts (migrated from `docker/`).

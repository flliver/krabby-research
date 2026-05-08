# MASt3R-SLAM Container Notes

## Hardware support

### Per project specs

| Doc | Hardware specified |
|-----|-------------------|
| M11 grant `OVERVIEW.md` | None (RTX 4090 mentioned only as paper benchmark) |
| M2 grant `OVERVIEW.md` | "16 GB NVIDIA RTX 5080, 32 GB RAM, Core i7 (or equivalent)" |
| `research/DEVELOPER.md` | "Recommended: RTX 5080 + CUDA 13.0 + PyTorch 2.7.0+cu130" |

Reference platform: **RTX 5080**. (DEVELOPER.md's cu130 combination is unverified
for sm_120 — see § Why PyTorch 2.7 cu128.)

### What we actually need to support

| GPU | Architecture | Compute capability | sm code | Status |
|-----|-------------|--------------------|---------|--------|
| RTX 4080 | Ada Lovelace | 8.9 | sm_89 | Jeremy's dev fleet — not in spec but in active use |
| RTX 5080 | Blackwell | 12.0 | sm_120 | **Project reference platform** |

A single container must run on both. Wheels with only one architecture are a dead end.

## Why `nvcr.io/nvidia/pytorch:25.10-py3` as the base

The Dockerfile was rewritten 2026-04-29 to align with the project's documented
container strategy in `research/docs/DOCKER_DEPENDENCIES.md`. The base image is the
same one used by the project's **Testing Container - x86**.

| Property | Value |
|----------|-------|
| Image | `nvcr.io/nvidia/pytorch:25.10-py3` |
| OS | Ubuntu 24.04.3 |
| PyTorch | 2.9.0a0+145a3a7bda.nv25.10 |
| CUDA | 13.0 |
| cuDNN | 9.14 |
| TensorRT | 10.13 |
| `torch.cuda.get_arch_list()` (verified on RTX 5080) | `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120` |

This base ships pre-built kernels for both Ada (sm_89 implicit via sm_86 + JIT) and
Blackwell (sm_120 native) — single image runs on RTX 4080 and RTX 5080.

### Earlier failed approaches (kept for reference)

Before discovering the NGC base image, we tried several pip wheel approaches:

| Approach | RTX 4080 (sm_89) | RTX 5080 (sm_120) |
|----------|------------------|-------------------|
| `torch==2.5.1+cu124` | Works | Fails: `no kernel image is available for execution on the device` |
| `torch==2.8.0+cu128` | curope fails to build with arch list including 12.0 | Same — never reaches runtime |
| `torch==2.7.0+cu128` (stable) | Works | Works |
| `torch+cu130` (per DEVELOPER.md) | Untested | Probably broken — pip cu130 channel doesn't ship sm_120 |

The DEVELOPER.md cu130 instructions are correct **if** they assume the NGC
container's CUDA 13.0 environment. They're misleading **if** read as advice for
host pip installs from `download.pytorch.org/whl/cu130`. Worth a doc clarification.

## Build constraints discovered

These are pain points encoded in the Dockerfile. Each was a separate debugging cycle:

1. **`TORCH_CUDA_ARCH_LIST` controls source extensions** — set to `"8.9;12.0"` so curope,
   lietorch, and mast3r_slam_backends compile kernels for both Ada and Blackwell. With
   PyTorch 2.5.1 this was rejected (`Unknown CUDA arch (12.0)`); with PyTorch 2.6+ it works.

2. **curope source patch required for PyTorch ≥ 2.6** — `kernels.cu` uses the deprecated
   `.type()` API which was removed in PyTorch 2.6. Replace with `.scalar_type()`.
   See `patch_curope.py`. Reference: [HF transformers issue #35976](https://github.com/huggingface/transformers/issues/35976).

3. **lietorch needs Eigen headers symlinked** — its build looks for headers at
   `eigen/Eigen/Dense` (its own submodule path) but our system has them at
   `/usr/include/eigen3/Eigen/`. Symlink: `ln -sf /usr/include/eigen3/Eigen /opt/lietorch/eigen/Eigen`.

4. **lietorch must be built `--no-build-isolation`** — pip's isolated build env pulls
   in newer PyTorch with a CUDA mismatch. Use the existing torch in the venv.

5. **mast3r_slam_backends needs `wheel` package** — without `pip install wheel`,
   `bdist_wheel` is missing and the build fails with `error: invalid command 'bdist_wheel'`.

6. **mast3r_slam_backends needs `CUDA_HOME` set** — without `CUDA_HOME=/usr/local/cuda`,
   setup.py reports `CUDA not found, cannot compile backend!` at metadata generation.

7. **`LD_LIBRARY_PATH` must include torch/lib** — at runtime, lietorch's `.so` extension
   needs `libc10.so` from PyTorch. Set `LD_LIBRARY_PATH=/opt/mast3r/lib/python3.11/site-packages/torch/lib`.

8. **Hard `import pyrealsense2`** — `mast3r_slam/dataloader.py` has a non-conditional
   import that fails in containers without the RealSense SDK. Patched to soft import.
   See `patch_dataloader.py`.

9. **Default glob is `*.png` only** — `RGBFiles` dataset class only globs `*.png`,
   ignoring `*.jpg`. Patched to glob both. See `patch_dataloader.py`.

10. **Required runtime packages not in `requirements.txt`** — must `pip install natsort plyfile`.

11. **mast3r_slam setup.py is incompatible with Docker build + CUDA 13** —
    multiple problems patched in `patch_mast3r_setup.py`:
    - CUDA detection uses `torch.cuda.is_available()` which is False during
      `docker build` (no GPU exposed). Replaced with `CUDA_HOME` env check.
    - Hardcoded gencode list covers sm_60 through sm_86. Two issues:
      (a) missing sm_89 (Ada) and sm_120 (Blackwell), (b) **CUDA 13 dropped
      sm_60, sm_61, and sm_70 support** so the old list won't compile under
      our NGC PyTorch 25.10 base. New list: sm_75, sm_80, sm_86, sm_89, sm_120.
    - Original sbeeprz build worked around the build-time check by using
      `docker run --gpus all + docker commit` instead of `docker build`.

12. **`--shm-size=8g` REQUIRED at run time** — Docker's default 64 MB shared memory
    is insufficient for MASt3R-SLAM's PyTorch multiprocessing. Without this, the
    container starts, prints the config, and silently deadlocks at 0% GPU. Took
    multiple debugging attempts to discover.
    See [MASt3R-SLAM Issue #94](https://github.com/rmurai0610/MASt3R-SLAM/issues/94).

## Run command

Use the full PyTorch-container flag set per
`research/docs/DOCKER_DEPENDENCIES.md`. Without `--shm-size=8g` the
container silently deadlocks at 0% GPU. The other three (`--ipc=host`,
`--ulimit memlock=-1`, `--ulimit stack=67108864`) are NVIDIA's official
recommendations.

```bash
docker run --rm --gpus all \
  --shm-size=8g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /path/to/data:/data \
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

## Distributing the image between hosts

The image is large (~36 GB) and complex to rebuild. To share between hosts:

```bash
# Source host: pipe save → load to destination
ssh <src> "sg docker -c 'docker save krabby-mast3r'" \
  | ssh <dst> 'sg docker -c "docker load"'
```

LAN transfer at ~250 MB/s observed (1 Gbps link, both ends to local SSD).
Full 36 GB image: ~2.5 minutes.

**Critical caveat:** an image saved from a host with PyTorch built for one architecture
will NOT run on a different architecture. Always build with the multi-arch wheel
(nightly cu128) so the same image works on every GPU in the fleet.

## Files in this directory

- `Dockerfile.mast3r` — the full recipe (start here)
- `patch_curope.py` — applied during build for PyTorch 2.6+ API
- `patch_dataloader.py` — applied during build for jpg + pyrealsense2 fixes
- `verify_imports.py` — run at end of build to catch issues early
- `MAST3R-NOTES.md` — this file
- `working-image-snapshot.txt` — pip freeze + filesystem layout from sbeeprz Apr 12 image (Ada-only, kept for reference)
- `working-image-history.txt` — `docker history` of the same

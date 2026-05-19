# Experiment 004 — MAtCha on sky-house-dining ★ leading candidate

**Status:** ✅ end-to-end, watertight mesh produced
**Date:** 2026-04-30
**Pipeline:** MAtCha (Atlas of Charts, CVPR 2025 Highlight) ported to PyTorch 2.7.0+cu128
**Hardware:** bbeeprz (RTX 5080, 16 GB VRAM, Ryzen 9800X3D, 1.3 TB free on `/home`)
**Container:** `krabby-matcha:latest` (33.9 GB, snapshotted from `matcha-build` after the working interactive session)
**Reference:** `docker/Dockerfile.matcha`, `docker/MATCHA-NOTES.md`, `AI/agents/krabby/active/matcha-pipeline-integration.md`, OLAI corpus `3d-reconstruction/matcha`

## Input

- Same `004-sky-house-dining` capture as `004-mast3r-slam-sky-house` (2.7K @ 30fps, locked exposure/WB, 3:47, native fisheye)
- **Sparse-view subset**: 12 evenly-spaced keyframes sampled from the 6,804-frame video
- Frames at 1024px wide (downscaled from 2.7K)
- Source: `data/videos/004-sky-house-dining/capture.mp4` → sparse extracts via ffmpeg `fps=24/227,scale=1024:-2` filter (originally 24 frames; 24 OOMs at 16 GB, so we cut to 12)

## Why 12 frames, not 24

**Critical empirical finding for M11 hardware**: 24 frames OOMs MAtCha's
chart-alignment optimization on 16 GB VRAM. Specific failure:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 648.00 MiB.
GPU 0 has a total capacity of 15.45 GiB of which 152.50 MiB is free.
```

Failure is in `matcha/dm_modules/matcher_3d.py::get_points_depth_in_depthmap_parallel`,
not the depth encoder. Reducing depth-encoder size from `vitl` → `vitb`
did not help (saved memory in the encoder, but the OOM is in the
chart-alignment math which scales with frame count, not encoder size).

**12 frames fits comfortably with vitl encoder.**

## Process — what actually worked (consolidated runner)

The path is captured in `runner.sh` (this directory). Summary:

1. **Sample frames** (locally on bbeeprz, not in MAtCha image, since `ffmpeg` lives in the `krabby-mast3r` image):
   ```
   docker run --rm -v <data>:/data krabby-mast3r:latest \
     ffmpeg -i /data/videos/004-sky-house-dining.mp4 \
            -vf "fps=24/227,scale=1024:-2" -q:v 2 \
            /data/frames/004-matcha-24/frame_%04d.jpg
   ```

2. **Run MAtCha**:
   ```
   docker exec matcha-build bash -c '
     source /opt/matcha/bin/activate
     export PYTHONPATH="/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn:$PYTHONPATH"
     export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     cd /opt/MAtCha
     python train.py \
       -s /data/frames/004-matcha-24 \
       -o /data/matcha_output/004-sky-house \
       --sfm_config unposed \
       --n_images 12 \
       --depthanythingv2_checkpoint_dir /opt/MAtCha/Depth-Anything-V2/checkpoints \
       --depthanything_encoder vitl
   '
   ```

3. **Decimate the 422 MB tetra mesh** to 200K triangles for Blender import:
   ```
   docker exec matcha-build bash -c '
     source /opt/matcha/bin/activate && python /tmp/decimate.py
   '
   ```
   (`decimate.py` saved as `decimate.py` in this dir — uses Open3D's
   quadric edge collapse decimation.)

## Runtime

| Stage | Time | Notes |
|-------|------|-------|
| Frame extraction (ffmpeg → 24 jpegs at 1024px) | 30 s | one-time |
| MASt3R-SfM (12 frames → poses + sparse) | ~1 min | produces `mast3r_sfm/cameras.json` + COLMAP-format sparse |
| align_charts (DepthAnythingV2 + chart optim) | ~3 min | the OOM-prone stage; fits at 12 frames |
| train_with_charts (2D Gaussian Splatting, 7000 iters @ ~22-25 it/s) | ~5 min | loss converges 0.06 → 0.003 |
| extract_tetra_mesh (TSDF + binary search level 7) | ~2 min | produces 422 MB watertight tetra mesh |
| **Total end-to-end** | **~11 min** | on RTX 5080 |
| Open3D decimation (21M → 200K tris) | ~3.5 min | per target; we ran 200K and 500K |

## Output

- `data/scenes/004-sky-house-dining/matcha_output/tetra_mesh_binary_search_7.ply` — **422 MB**, **21M triangles**, **watertight** (TSDF + tetrahedralization)
- `data/scenes/004-sky-house-dining/matcha_output/points.ply` — 32 MB MASt3R-SfM intermediate sparse cloud
- `data/scenes/004-sky-house-dining/matcha_output/cameras.json` — 4.4 KB camera poses (12 cameras)
- `data/scenes/004-sky-house-dining/matcha_output/mesh/sky_house_matcha_200k.obj` — **15 MB**, 200K-triangle decimation (matches MASt3R-SLAM mesh size for direct comparison)
- `data/scenes/004-sky-house-dining/matcha_output/mesh/sky_house_matcha_200k.ply` — 6.7 MB
- `data/scenes/004-sky-house-dining/matcha_output/mesh/sky_house_matcha_500k.obj` — 41 MB, 500K-triangle (higher fidelity, still snappy in Blender)

## Quality verdict

Per Jeremy: **"MAtCha by far was the best so far."**

The watertight property is a structural advantage over MASt3R-SLAM's
ball-pivoting output for the M11 collision-quality goal. Geometric
density (21M tris from 12 input frames) is dramatically higher than the
MASt3R-SLAM cloud's effective resolution.

## Milestone fit

| Req | Score | Notes |
|-----|-------|-------|
| R1 Watertight mesh | ✅ — TSDF + tetra produces watertight by construction | The first pipeline in this milestone to satisfy R1 natively |
| R2 Metric scale | 🟡 — same as everyone, needs a reference object in capture | Capture protocol unchanged |
| R3 Camera poses | ✅ — via MASt3R-SfM stage, COLMAP-format | |
| R4 Multi-arch | ✅ — `krabby-matcha:latest` runs on RTX 5080 (sm_120) | First time MAtCha has been run on Blackwell, our own port |
| R5 Wall-clock | ✅ — **11 min** end-to-end on 12 frames | Faster than MASt3R-SLAM (40 min on the full video) |
| R6 Build complexity | 🟡 — high (8 patches, 6 native CUDA extensions, careful PyTorch version pinning) | All captured in `Dockerfile.matcha` + `MATCHA-NOTES.md`; reproducible |
| R7 M11-validated | ✅ — produced output on real M11 video | Today's session |

## The 8 patches that made the build work

(Full backstory in `docker/MATCHA-NOTES.md`. Listed here for cross-reference.)

1. `pytorch3d 0.7.8` from source with `--no-build-isolation` (no wheel for torch 2.7)
2. `curope/kernels.cu`: `.type()` → `.scalar_type()` (PyTorch 2.6+ removed the deprecated form)
3. `<cstdint>` includes added to 4 GS extension headers (gcc 13 stopped transitively including this)
4. `<cfloat>` prepended to `simple_knn.cu` (must be at line 1, not after `#pragma once` — the file starts with a comment block)
5. `_GLIBCXX_USE_CXX11_ABI=1` in `tetra-triangulation/cmake/FindTorch.cmake` (PyTorch 2.7 wheels are CXX11_ABI=1)
6. Explicit `/usr/local/cuda/include` in `tetra-triangulation/CMakeLists.txt` (non-conda build needed it)
7. `faiss-cpu` instead of `faiss-gpu-cu12==1.14.1` (the latter lacks sm_120 kernels — same class as VGGT's wheel-availability problem)
8. `weights_only=False` added to 41 `torch.load` sites across the MAtCha tree (PyTorch 2.6+ default flip)

## MAtCha-specific runtime gotchas

- **Don't install xformers** — it pulls torch 2.11.0 nightly without a version constraint, breaking pytorch3d's compiled `_C.so` (ABI mismatch). MAtCha's perf is fine without it.
- **`train.py` shells out to `python` (no version)** — the venv must be activated; bare-environment `python` resolution will fail with "python: not found."
- **`PYTHONPATH` must explicitly include the simple-knn submodule** — its editable install uses a flat layout (no `simple_knn/` directory), so `import simple_knn` from outside the source tree fails without explicit path.
- **`--depthanything_encoder` accepts `large/base/small/giant` from `train.py`'s CLI but the depth-loader expects `vitl/vitb/vits/vitg`** — `train.py` does the translation when it shells out to `train_with_charts.py`, but **not** when it shells out to `align_charts.py`. Workaround: pass `vitl` directly.
- **`patch_matcha_torch_load.py` has a regex corner case**: `torch.load(os.path.join(...), map_location=...)` calls with multi-line first arguments get the `weights_only=False` inserted *inside* the `os.path.join`. We hit this on `matcha/pointmap/depthanythingv2.py`, fixed manually. Note for future cleanup: improve the regex to handle the `os.path.join` case explicitly.

## Lessons that will inform every subsequent scene

1. **The 12-frame ceiling on 16 GB VRAM is the planning constraint** — until we resolve it (smaller resolution? batch the chart alignment?) every M11 MAtCha capture should target ≤ 12 keyframes per scene.
2. **The watertight mesh is dense by construction** (21M tris from 12 frames). Decimating to 200K via Open3D's quadric collapse takes ~3.5 min per target and is the right finishing step for the M11 deliverable.
3. **Build the matcha image once, distribute via `docker save | docker load`** — the image is 33.9 GB; rebuilding from `Dockerfile.matcha` takes ~30+ min while the LAN transfer takes ~2.5 min.

## Open follow-ups (next session candidates)

- Convert one MAtCha mesh to USD and walk a hexapod through it in IsaacSim (the actual M11 deliverable validation).
- Try MAtCha on a captured scene with a known-size reference object to validate metric scale.
- Investigate the 24-frame VRAM ceiling — is it per-frame-area? Lowering to 768px input might let us double frame count.
- Compare MAtCha-004 vs MASt3R-SLAM-004 in Blender side-by-side (Jeremy is doing this now).

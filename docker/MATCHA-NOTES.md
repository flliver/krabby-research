# MAtCha Container Notes

Companion to `Dockerfile.matcha`. Captures the porting battle notes from
the April 2026 build session.

## Why we built this on a non-official stack

MAtCha's `environment.yml` pins **PyTorch 2.0.1 + CUDA 11.8 + Python 3.9**
via conda. CUDA 11.8 cannot run on RTX 5080 (Blackwell, sm_120) — the
project's reference platform. So we ported MAtCha to a stack that
**does** support sm_120:

| Component | MAtCha official | Our build (Path B) |
|-----------|-----------------|--------------------|
| Python | 3.9 | 3.11 |
| CUDA toolkit | 11.8 | 12.8 |
| PyTorch | 2.0.1+cu118 | 2.7.0+cu128 |
| pytorch3d | 0.7.4 (conda channel) | 0.7.8 (built from source) |
| GPU support | sm_75–sm_89 | sm_75–sm_120 (multi-arch) |
| Manager | conda | venv + pip |

This was Path B from the MASt3R-SLAM build sweep that established
PyTorch 2.7.0+cu128 as the sweet spot for sm_89 + sm_120 multi-arch.
Port effort was ~2 hours of interactive debugging.

## The 8 patches (in build order)

### 1. pytorch3d `--no-build-isolation`

pytorch3d's `setup.py` imports torch at build time. pip's isolated build
env doesn't have torch installed, so the build fails with
`ModuleNotFoundError: No module named 'torch'`. Same lesson as `lietorch`
in MASt3R-SLAM. Fix: `pip install --no-build-isolation`.

### 2. curope kernels.cu: `.type()` → `.scalar_type()`

PyTorch 2.6 removed `tensor.type()`. MAtCha's curope C++ code uses it.
Same code as MASt3R-SLAM's curope (literally — they share the submodule).
Patch script: `patch_matcha_curope.py`.

### 3. Missing `<cstdint>` includes (4 headers)

Ubuntu 24.04 ships gcc 13, which stopped transitively including `<cstdint>`
through other headers. The Inria 2D Gaussian Splatting code (~2023) relied
on the transitive include. Affected:
- `diff-surfel-rasterization/cuda_rasterizer/rasterizer_impl.h`
- `diff-surfel-rasterization/cuda_rasterizer/auxiliary.h`
- `simple-knn/simple_knn.h`
- `simple-knn/spatial.h`

Symptom: `error: namespace "std" has no member "uintptr_t"` and similar
for `uint32_t` / `uint64_t`. Patch script: `patch_matcha_includes.py`.

### 4. Missing `<cfloat>` include (simple-knn)

`simple-knn/simple_knn.cu` uses `FLT_MAX` without including `<cfloat>`.
Same gcc-13 transitive-include issue. **Caveat**: cannot use `sed -i "1a..."`
because `simple_knn.cu` starts with a multi-line `/* ... */` comment block,
and inserting at line 2 puts the include INSIDE the comment. Must
**prepend** at line 1. Patch script: `patch_matcha_includes.py`.

### 5. ASMK requires faiss before install

`asmk` setup.py raises a hard error if faiss isn't already installed
(`ERROR: faiss package not installed`). Pre-install faiss before pip
install asmk.

### 6. faiss-gpu-cu12 lacks sm_120 kernels

`faiss-gpu-cu12==1.14.1` (the latest cu12 wheel as of April 2026) was
compiled without sm_120 architecture. `import faiss` works; `get_num_gpus()`
returns 1; **but actual GPU computation fails with**:

```
Faiss assertion 'err__ == cudaSuccess' failed in runL2Norm at L2Norm.cu:257;
details: CUDA error 209 no kernel image is available for execution on the device
```

Same symptom as our MASt3R-SLAM RTX 5080 wheel-availability finding,
just for faiss instead of torch.

**Fix**: use `faiss-cpu` instead. ASMK works with either backend; for the
sparse-view sizes MAtCha targets (24-30 frames), the performance hit is
negligible. If you need faiss-gpu on Blackwell, build faiss from source
with `-DCMAKE_CUDA_ARCHITECTURES="89;120"`.

### 7. tetra-triangulation: ABI flag + CUDA include

Two CMakeLists problems:
- `cmake/FindTorch.cmake` forces `-D_GLIBCXX_USE_CXX11_ABI=0`. PyTorch
  2.7+cu128 wheels are built with `=1`. Linking against them with `=0`
  produces undefined-symbol errors at import time. Flip to `=1`.
- `CMakeLists.txt` assumes `${CONDA_PREFIX}/include` provides CUDA
  headers. In a non-conda Docker build, .cpp files don't get
  `/usr/local/cuda/include` on their search path (only .cu via nvcc
  auto-include), so `cuda_runtime.h: No such file or directory`. Add
  `include_directories(/usr/local/cuda/include)` before
  `find_package(Torch)`.

Patch script: `patch_matcha_tetra_cmake.py`.

### 8. torch.load weights_only=True (PyTorch 2.6+)

Same flip as MASt3R-SLAM. PyTorch 2.6 changed `torch.load` default to
`weights_only=True`, which rejects checkpoints containing
`argparse.Namespace` (which all of MASt3R, DUST3R, DepthAnythingV2 do).
Walk the tree and add `weights_only=False` to every call.

MAtCha has **41 sites** to patch (vs ~12 in MASt3R-SLAM) because it
pulls in DUST3R + MASt3R-SfM + DepthAnythingV2 + 2D-Gaussian-Splatting
all in one project. Patch script: `patch_matcha_torch_load.py`.

## Two MAtCha-specific runtime gotchas

### Don't install xformers

xformers (0.0.35 from PyPI as of April 2026) declares torch as an
install dep without a version constraint. `pip install xformers` pulls
**torch 2.11.0 nightly**, which uninstalls torch 2.7.0 and breaks
everything compiled against it (pytorch3d, all 6 native extensions).

Symptom after xformers install:
```
RuntimeError: operator torchvision::nms does not exist
ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

xformers is optional in MAtCha — it's only used as a perf optim if
present. Don't install it.

### MAtCha shells out to `python` (no version)

`train.py` invokes `python mast3r/run_mast3r.py ...` via `os.system()`.
In a venv, the `python` binary lives at `/opt/matcha/bin/python` and
isn't on the system PATH unless the venv is activated. Either source
the activate script before running, or symlink `/opt/matcha/bin/python`
to `/usr/local/bin/python`.

### PYTHONPATH must include MAtCha submodule paths

`train.py` doesn't add its own submodule directories to `sys.path`
before subprocess'ing into them. Set `PYTHONPATH` explicitly:

```
export PYTHONPATH="/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn:$PYTHONPATH"
```

The `simple-knn` entry is needed because `pip install -e .` on a
flat-layout package (no inner `simple_knn/` directory) creates an
egg-link pointing at the source dir; without that on PYTHONPATH the
package isn't found from outside the source tree.

## Validated state (April 2026)

| Stage | Status |
|-------|--------|
| Build (all 6 native extensions) | ✅ |
| Image runs on RTX 5080 | ✅ |
| MASt3R-SfM stage (24 frames → 24 pointmap JSONs) | ✅ |
| align_charts stage | ❌ — cameras.json not produced by MASt3R-SfM |
| train_with_charts (2D Gaussian Splatting) | ❌ — "Could not recognize scene type!" |
| extract_tetra_mesh | ❌ — no input from previous stage |

The build works end-to-end on Blackwell. The pipeline integration
between MAtCha's stages has issues that need separate debugging — likely
a config / arg-passing mismatch between MASt3R-SfM's output format and
what `align_charts.py` expects to read.

## Image distribution

The image is ~34 GB (smaller than `krabby-mast3r:latest` because it
doesn't include the NGC PyTorch base layers — we built on a thinner
CUDA-only base).

Distribute via `docker save | docker load` between fleet hosts (~2.5
min on gigabit LAN). See `docker/pytorch-containers` corpus topic.

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
    -v <data-dir>:/data \
    --env-file ../.env \
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

## Files in this directory

- `Dockerfile.matcha` — the recipe
- `MATCHA-NOTES.md` — this file
- `patch_matcha_curope.py` — applied during build (.type → .scalar_type)
- `patch_matcha_includes.py` — applied during build (cstdint, cfloat)
- `patch_matcha_tetra_cmake.py` — applied during build (ABI, CUDA include)
- `patch_matcha_torch_load.py` — applied during build (weights_only=False)

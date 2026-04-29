# PyTorch GPU Support Matrix

This document captures the relationship between NVIDIA GPU architectures, PyTorch
wheel index URLs, NGC container choices, and the source-extension build patches
needed for modern PyTorch (2.6+) and CUDA 13.

This was assembled from real failures encountered while building MASt3R-SLAM
across the project's RTX 4080 and RTX 5080 hosts in April 2026. The shortcuts
documented here will save the next person ~2 days of debugging.

## GPU architectures in active use

| GPU | Architecture | Compute capability | sm code |
|-----|-------------|--------------------|---------|
| Jetson Orin (Ampere) | Ampere | 8.7 | sm_87 |
| RTX 4080 | Ada Lovelace | 8.9 | sm_89 |
| RTX 5080 | Blackwell | 12.0 | sm_120 |

The project's reference platform per `DEVELOPER.md` and the M2 grant is RTX 5080.
RTX 4080s are also in active use as dev hardware.

## PyTorch wheel selection

Pick the wheel that has prebuilt kernels for the GPUs you need to run on. The
PyTorch pip wheels are **self-contained** — they bundle their own CUDA runtime
and don't depend on the host CUDA Toolkit version, so the cu1xx index version
is independent of what's installed on the OS.

| Index URL | First version with sm_120 | Notes |
|-----------|---------------------------|-------|
| `whl/cu118` | Never — Pascal/Volta era | EOL for Blackwell |
| `whl/cu121` | Never | EOL |
| `whl/cu124` | Never | Last cu version before Blackwell launch — runs on Ada/Hopper but **fails on RTX 5080** |
| `whl/cu126` | Some nightly | Skip — fragmented support |
| **`whl/cu128`** | **`torch==2.7.0` (stable)** | **Recommended.** Wheel includes sm_75/80/86/90/100/120. Single wheel runs on RTX 4080 + 5080. |
| `whl/cu130` | Not yet (as of April 2026) | Stable wheels do not include sm_120. Available only in nightly 2.11+. |
| `whl/nightly/cu128` | Yes | Use only if cu128 stable is missing a feature you need. |

### Verifying what's in a wheel

```python
import torch
print(torch.cuda.get_arch_list())
# Should include 'sm_120' for RTX 5080 support
```

If `get_arch_list()` doesn't include your GPU's sm code, you'll get this
runtime error when trying to use the device:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Recommended host install (Ubuntu 24.04)

```bash
python3.11 -m venv testenv
source testenv/bin/activate
pip3 install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

This works on both RTX 4080 and RTX 5080 hosts regardless of installed CUDA Toolkit.

## NGC PyTorch container alternative

For ML-heavy containers (testing-x86, IsaacSim, future locomotion variants on x86),
the NGC PyTorch container is the preferred path because it ships pre-built kernels
for everything and includes optimized cuDNN/TensorRT/etc.

| Image | OS | PyTorch | CUDA | Arch list (verified on RTX 5080) |
|-------|----|---------|------|----------------------------------|
| `nvcr.io/nvidia/pytorch:25.10-py3` | Ubuntu 24.04 | 2.9.0a0 | 13.0 | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120 |

Picking this base in a Dockerfile means you don't have to manage `pip install
torch ...` yourself — and the version PyTorch comes pre-tuned for NVIDIA's stack.

## Source-extension compatibility (PyTorch 2.6+ and CUDA 13)

If your code builds CUDA extensions from source (custom kernels, torch C++
extensions, projects like lietorch, mast3r_slam_backends, curope, etc.),
several breaking changes between PyTorch ≤ 2.5 and PyTorch 2.6+ may bite you.

### Required source patches for PyTorch 2.6+

| Old API | New API | Where it appears |
|---------|---------|------------------|
| `tensor.type()` | `tensor.scalar_type()` | `AT_DISPATCH_*` macros, anywhere you query a tensor's dtype |
| `torch::linalg::norm()` | `at::linalg_norm()` | C++ linalg API (the entire `torch::linalg::` namespace was removed in PyTorch 2.9) |
| `torch::linalg::det()` | `at::linalg_det()` | Same |
| `torch.load(path)` | `torch.load(path, weights_only=False)` | Default flipped in 2.6 — checkpoints with non-tensor objects (e.g. `argparse.Namespace`) fail to load |

### Build-environment quirks

1. **`torch.cuda.is_available()` returns False during `docker build`** — no GPU is
   exposed at build time. If your `setup.py` uses this to decide whether to compile
   the CUDA extension, it will silently skip the build. Workaround:
   ```python
   has_cuda = bool(os.environ.get('CUDA_HOME'))
   ```

2. **`pip install --no-build-isolation`** — if your extension links against the
   already-installed PyTorch, use this flag. Without it, pip creates an isolated
   build env that pulls a different (mismatched) PyTorch and produces a binary
   that fails at runtime with `libcudart.so.X: cannot open shared object file`.

3. **`LD_LIBRARY_PATH` must include torch/lib** — at runtime, custom `.so`
   extensions need libtorch's shared libraries:
   ```dockerfile
   ENV LD_LIBRARY_PATH=/path/to/python/site-packages/torch/lib:$LD_LIBRARY_PATH
   ```

### CUDA 13 dropped old architectures

CUDA 13.0 (used by NGC PyTorch 25.10 and recent host installs) **removed support
for sm_60, sm_61, and sm_70** (Pascal and Volta). If you have a project that
hardcodes a gencode list like:

```python
# OLD — fails on CUDA 13 with "Unsupported gpu architecture 'compute_60'"
"-gencode=arch=compute_60,code=sm_60",
"-gencode=arch=compute_61,code=sm_61",
"-gencode=arch=compute_70,code=sm_70",
"-gencode=arch=compute_75,code=sm_75",
"-gencode=arch=compute_80,code=sm_80",
"-gencode=arch=compute_86,code=sm_86",
```

Replace with:

```python
# NEW — covers the project's actual hardware
"-gencode=arch=compute_75,code=sm_75",
"-gencode=arch=compute_80,code=sm_80",
"-gencode=arch=compute_86,code=sm_86",
"-gencode=arch=compute_87,code=sm_87",   # Jetson Orin
"-gencode=arch=compute_89,code=sm_89",   # RTX 4080
"-gencode=arch=compute_120,code=sm_120", # RTX 5080
```

### Setting `TORCH_CUDA_ARCH_LIST`

For Dockerfiles that build CUDA extensions, set this before the build steps:

```dockerfile
ENV TORCH_CUDA_ARCH_LIST="8.7;8.9;12.0"
```

Listing only the architectures we need keeps the build fast. Adding `;12.0`
requires PyTorch 2.7+ — earlier versions reject it as `Unknown CUDA arch`.

## References

- [PyTorch Forum: sm_120 support thread](https://discuss.pytorch.org/t/pytorch-support-for-sm120/216099)
- [GitHub: pytorch/pytorch issue #164342 — official sm_120 support](https://github.com/pytorch/pytorch/issues/164342)
- [HuggingFace transformers issue #35976 — `.type()` deprecation](https://github.com/huggingface/transformers/issues/35976)
- [PyTorch 1.9 release notes](https://github.com/pytorch/pytorch/releases/tag/v1.9.0) — original `linalg_norm` → `norm` rename
- [NVIDIA NGC PyTorch container catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

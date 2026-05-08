# MAtCha (Atlas of Charts) container for sparse-view 3D reconstruction
#   Build: docker build -t krabby-matcha -f Dockerfile.matcha .
#   Run:   docker run --rm --gpus all --shm-size=8g \
#               -v <data-dir>:/data \
#               --env-file ../.env \
#               krabby-matcha
#
# Hardware: builds multi-arch (sm_89 Ada + sm_120 Blackwell). Validated on
#   RTX 5080 (Blackwell, 16 GB).
#
# Stack (Path B from MASt3R-SLAM build sweep):
#   - CUDA 12.8 base
#   - Python 3.11 venv
#   - PyTorch 2.7.0 + torchvision 0.22.0 + cu128 (wheel)
#   - pytorch3d 0.7.8 (built from source — no wheels for torch 2.7)
#   - 6 native CUDA extensions: curope, ASMK, diff-surfel-rasterization,
#     simple-knn, tetra-triangulation (all source-compiled)
#   - faiss-cpu (faiss-gpu-cu12 1.14.1 lacks sm_120 kernels — see
#     docker/MATCHA-NOTES.md)
#
# This Dockerfile encodes 8 patches discovered while porting MAtCha from
# its officially-supported PyTorch 2.0.1 + cu118 stack. See MATCHA-NOTES.md
# for the full backstory.

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"
ENV FORCE_CUDA=1

# --- System deps ---
RUN apt-get update && apt-get install -y \
        software-properties-common \
        git wget curl ffmpeg cmake ninja-build \
        libgl1 libglib2.0-0 libeigen3-dev \
        libcgal-dev libboost-all-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/matcha
ENV PATH="/opt/matcha/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel setuptools ninja cython

# --- PyTorch 2.7.0 + cu128 (validated for sm_120) ---
RUN pip install --no-cache-dir \
        torch==2.7.0 torchvision==0.22.0 \
        --index-url https://download.pytorch.org/whl/cu128

# --- pytorch3d 0.7.8 from source ---
# No-build-isolation lets setup.py see the torch we just installed.
RUN pip install --no-cache-dir fvcore iopath
RUN pip install --no-cache-dir --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# --- Clone MAtCha ---
RUN git clone --recursive https://github.com/Anttwo/MAtCha.git /opt/MAtCha

# --- Patch & build curope (PyTorch 2.6+ .type → .scalar_type) ---
COPY patch_matcha_curope.py /tmp/patch_matcha_curope.py
RUN python /tmp/patch_matcha_curope.py
RUN cd /opt/MAtCha/mast3r/dust3r/croco/models/curope && \
    python setup.py build_ext --inplace

# --- ASMK (Cython + faiss-cpu; faiss-gpu-cu12 1.14.1 lacks sm_120) ---
RUN pip install --no-cache-dir faiss-cpu
RUN cd /opt/MAtCha/mast3r/asmk/cython && cythonize *.pyx
RUN cd /opt/MAtCha/mast3r/asmk && \
    pip install --no-cache-dir --no-build-isolation .

# --- 2D Gaussian Splatting submodules ---
# Patch missing <cstdint> / <cfloat> includes before building.
COPY patch_matcha_includes.py /tmp/patch_matcha_includes.py
RUN python /tmp/patch_matcha_includes.py

RUN cd /opt/MAtCha/2d-gaussian-splatting/submodules/diff-surfel-rasterization && \
    pip install --no-cache-dir --no-build-isolation -e .
RUN cd /opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn && \
    pip install --no-cache-dir --no-build-isolation -e .

# --- Tetra-triangulation: cmake + make (separate from pip) ---
# patch_matcha_tetra_cmake.py adds explicit CUDA include and flips CXX11_ABI to 1.
COPY patch_matcha_tetra_cmake.py /tmp/patch_matcha_tetra_cmake.py
RUN python /tmp/patch_matcha_tetra_cmake.py
RUN cd /opt/MAtCha/2d-gaussian-splatting/submodules/tetra-triangulation && \
    cmake -DPYTHON_EXECUTABLE=/opt/matcha/bin/python . && \
    make -j4 && \
    pip install --no-cache-dir --no-build-isolation -e .

# --- MAtCha runtime Python deps (originally conda packages in environment.yml) ---
# Note: do NOT install xformers — it pulls torch 2.11 nightly which breaks
# pytorch3d's compiled _C.so (ABI mismatch). The benefit is small and
# MAtCha runs fine without it.
RUN pip install --no-cache-dir \
        plyfile open3d trimesh tqdm matplotlib roma einops \
        huggingface-hub safetensors transformers timm \
        opencv-python==4.11.0.86 imageio imageio-ffmpeg natsort \
        scikit-learn scikit-image pyyaml gradio

# --- Patch torch.load calls (PyTorch 2.6+ default weights_only=True) ---
# Run AFTER all source has been cloned/installed, since this rewrites .py
# files in the source tree.
COPY patch_matcha_torch_load.py /tmp/patch_matcha_torch_load.py
RUN python /tmp/patch_matcha_torch_load.py

# --- Download model checkpoints (~4.3 GB total) ---
WORKDIR /opt/MAtCha
RUN python download_checkpoints.py --depth_encoder large

# --- Runtime config ---
ENV PYTHONPATH="/opt/MAtCha:/opt/MAtCha/mast3r:/opt/MAtCha/mast3r/dust3r:/opt/MAtCha/2d-gaussian-splatting:/opt/MAtCha/2d-gaussian-splatting/submodules/simple-knn:${PYTHONPATH}"

WORKDIR /workspace
CMD ["/bin/bash"]

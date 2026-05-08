# MASt3R-SLAM container
# Built on outpost: docker build -t krabby-mast3r -f Dockerfile.mast3r .
# RUN with: docker run --gpus all --shm-size=8g ...   (shm-size REQUIRED — see notes)
#
# Aligned with project container strategy in research/docs/DOCKER_DEPENDENCIES.md:
# uses the same x86 base image as the testing container (NGC PyTorch 25.10).
# This base ships PyTorch 2.9 with CUDA 13.0 and pre-built kernels for both
# Ada (sm_89) and Blackwell (sm_120) — single image runs on RTX 4080 + RTX 5080.
#
# See MAST3R-NOTES.md for the full backstory of every choice in this file.

FROM nvcr.io/nvidia/pytorch:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive

# HF_TOKEN is supplied at run time, not baked into the image.
# Pass it via:  docker run --env-file ../.env --gpus all --shm-size=8g ...
# See ../.env.example for the expected variables.

# Both Ada (RTX 4080 = 8.9) and Blackwell (RTX 5080 = 12.0).
# Required for source-compiled extensions: curope, lietorch, mast3r_slam_backends.
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"

# System deps (Eigen needed for lietorch build)
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg cmake ninja-build \
    libgl1 libglib2.0-0 libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Use the base image's Python (3.12 in 25.10) — no separate venv needed.
# Base image already has torch, torchvision, numpy, pillow, etc.

# Clone repos
RUN git clone https://github.com/rmurai0610/MASt3R-SLAM.git --recursive /opt/MASt3R-SLAM
RUN git clone https://github.com/princeton-vl/lietorch.git /opt/lietorch

WORKDIR /opt/MASt3R-SLAM

RUN pip install --no-cache-dir wheel setuptools

# Patch curope CUDA kernel for PyTorch 2.6+ API (.type() → .scalar_type()).
COPY patches/patch_curope.py /tmp/patch_curope.py
RUN python /tmp/patch_curope.py

# Install MASt3R thirdparty
RUN pip install --no-cache-dir -e thirdparty/mast3r
RUN pip install --no-cache-dir -e thirdparty/in3d || true

# Build lietorch from source. CMakeLists looks for headers at eigen/Eigen.
RUN ln -sf /usr/include/eigen3/Eigen /opt/lietorch/eigen/Eigen && \
    cd /opt/lietorch && python setup.py install

# Patch mast3r_slam setup.py for build-time CUDA check + modern arch list
COPY patches/patch_mast3r_setup.py /tmp/patch_mast3r_setup.py
RUN python /tmp/patch_mast3r_setup.py

# Build mast3r_slam_backends C extension
RUN pip install --no-cache-dir --no-build-isolation -e .

# Runtime deps not in requirements.txt
RUN pip install --no-cache-dir natsort plyfile

# Patch dataloader: accept jpg + soft pyrealsense2 import
COPY patches/patch_dataloader.py /tmp/patch_dataloader.py
RUN python /tmp/patch_dataloader.py

# Patch torch.load calls: PyTorch 2.6+ defaults to weights_only=True, which rejects
# MASt3R checkpoints (they contain argparse.Namespace).
COPY patches/patch_torch_load.py /tmp/patch_torch_load.py
RUN python /tmp/patch_torch_load.py

# Download checkpoints (~1.5 GB)
RUN mkdir -p checkpoints && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/ && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/ && \
    wget -q https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/

# Verify everything imports
COPY verify_imports.py /tmp/verify_imports.py
RUN python /tmp/verify_imports.py

WORKDIR /workspace
CMD ["/bin/bash"]

# VGGT: Visual Geometry Grounded Transformer
# CVPR 2025 Best Paper — feed-forward 3D reconstruction
# Built on outpost: docker build -t krabby-vggt -f Dockerfile.vggt .

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# HF_TOKEN is supplied at run time, not baked into the image.
# Pass it via:  docker run --env-file ../.env --gpus all --shm-size=8g ...
# See ../.env.example for the expected variables.
ENV TORCH_CUDA_ARCH_LIST="8.9;12.0"

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    python3-pip git wget curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/vggt
ENV PATH="/opt/vggt/bin:$PATH"

# PyTorch for CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone VGGT
RUN git clone https://github.com/facebookresearch/vggt.git /opt/VGGT

WORKDIR /opt/VGGT

# Install deps
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_demo.txt || true

# Pre-download model weights from HuggingFace
RUN python -c "from vggt.models.vggt import VGGT; model = VGGT.from_pretrained('facebook/VGGT-1B'); print('Model downloaded')"

# Also install Open3D for point cloud → mesh
RUN pip install --no-cache-dir open3d trimesh

WORKDIR /workspace

CMD ["/bin/bash"]

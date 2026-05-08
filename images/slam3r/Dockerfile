# SLAM3R container for scene reconstruction
# Built on outpost: docker build -t krabby-slam3r -f Dockerfile.slam3r .

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    software-properties-common git wget curl ffmpeg cmake \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Create venv with python3.11
RUN python3.11 -m venv /opt/slam3r
ENV PATH="/opt/slam3r/bin:$PATH"

# Clone SLAM3R
RUN git clone https://github.com/PKU-VCL-3DV/SLAM3R.git /opt/SLAM3R

WORKDIR /opt/SLAM3R

# PyTorch + deps
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_optional.txt || true

# Build CuRoPE extension for performance
RUN cd slam3r/pos_embed/curope && python setup.py build_ext --inplace || true

# HF_TOKEN is supplied at run time, not baked into the image.
# Pass it via:  docker run --env-file ../.env --gpus all --shm-size=8g ...
# See ../.env.example for the expected variables.

WORKDIR /workspace

CMD ["/bin/bash"]

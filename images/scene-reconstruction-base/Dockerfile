# Layer 2: Platform software for scene reconstruction
# Built on outpost: docker build -t krabby-011-scene-reconstruction .

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Build deps for COLMAP (source build for CUDA support)
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    git wget curl ffmpeg cmake ninja-build \
    libboost-all-dev libeigen3-dev libflann-dev libfreeimage-dev \
    libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev \
    libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev \
    libceres-dev \
    && rm -rf /var/lib/apt/lists/*

# Build COLMAP from source WITH CUDA
# Ada Lovelace = 89 (RTX 4080/4090), Blackwell = 100 (RTX 5080/5090)
RUN git clone --depth 1 --branch 3.11.1 https://github.com/colmap/colmap.git /tmp/colmap && \
    cd /tmp/colmap && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="89;100" \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF && \
    ninja -j$(nproc) && \
    ninja install && \
    rm -rf /tmp/colmap

# Python packages for mesh processing (T1-T2)
RUN pip3 install --break-system-packages \
    open3d \
    pymeshlab \
    trimesh \
    numpy \
    pillow

# HF_TOKEN is supplied at run time, not baked into the image.
# Pass it via:  docker run --env-file ../.env --gpus all --shm-size=8g ...
# See ../.env.example for the expected variables.

WORKDIR /workspace

CMD ["/bin/bash"]

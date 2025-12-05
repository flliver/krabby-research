# Production Dockerfile for Jetson Orin robot deployment
# This container combines inference logic and HAL server for production use on the robot
# Uses inproc ZMQ for same-process communication
#
# Building:
#   On Jetson (ARM64): docker build -f images/locomotion/Dockerfile -t krabby-locomotion:latest .
#   On x86_64: docker buildx build --platform linux/arm64 -f images/locomotion/Dockerfile -t krabby-locomotion:latest --load .

FROM nvcr.io/nvidia/pytorch:25.10-py3-igpu

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libzmq3-dev \
    wget \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install ZED SDK
# Note: ZED SDK is required for ZED camera functionality
# See docs/DOCKER_DEPENDENCIES.md for JetPack version compatibility and limitations
# L4T 36.4 corresponds to JetPack 6.1/6.2
RUN echo "Installing ZED SDK 5.1.1 for L4T 36.4 (JetPack 6.1/6.2)..." && \
    wget -O /tmp/zed_sdk.run \
    "https://download.stereolabs.com/zedsdk/5.1.1/l4t36.4/jetsons" && \
    chmod +x /tmp/zed_sdk.run && \
    /tmp/zed_sdk.run --noexec --target /tmp/zed_installer && \
    cd /tmp/zed_installer && \
    ./linux_install_release.sh -- silent && \
    rm -rf /tmp/zed_sdk.run /tmp/zed_installer && \
    echo "ZED SDK installed successfully"

# Copy requirements file first (for better Docker layer caching)
COPY images/locomotion/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: Base image already includes PyTorch, NumPy, CUDA, cuDNN, TensorRT
# Note: pyzed requires ZED SDK to be installed (installed above)
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
# Note: Wheels must be built before building this Docker image
COPY hal/client/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/jetson/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
# Note: hal/tools is excluded by default (only install when needed for debugging)
RUN bash -c "pip3 install --no-cache-dir /tmp/wheels/*.whl"

# Set entrypoint to HAL server with integrated inference
# This runs both Jetson HAL server and policy inference in the same process
ENTRYPOINT ["python3", "-m", "hal.server.jetson.main"]

# Default arguments (can be overridden at runtime)
# Users should provide: --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


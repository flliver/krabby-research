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
#
# GStreamer + GI typelibs so the data collector can H.265-encode via in-process appsrc
# (PyGObject in requirements.txt). JetPack still supplies nvv4l2h265enc when present.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libzmq3-dev \
    wget \
    zstd \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    pkg-config \
    gobject-introspection \
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
# Use Docker build cache mount for pip to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
# Note: Wheels must be built before building this Docker image
COPY hal/client/dist/*.whl /tmp/wheels/
COPY controller/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/jetson/dist/*.whl /tmp/wheels/
COPY hal/tools/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
COPY data_collection/dist/*.whl /tmp/wheels/
COPY teleop/edge/dist/*.whl /tmp/wheels/
RUN bash -c "pip3 install --no-cache-dir /tmp/wheels/*.whl"

COPY controller/ /workspace/controller/
COPY firmware/ /workspace/firmware/
ENV PYTHONPATH=/workspace
# Jetson camera stack requires its libnvjpeg for pyzed/Argus interop.
# If removed, pyzed may resolve CUDA libnvjpeg first and fail to import with:
# "undefined symbol: jpeg_set_hardware_acceleration_parameters_enc".
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/nvidia/libnvjpeg.so

# Set entrypoint to HAL server with integrated inference
# This runs both Jetson HAL server and policy inference in the same process
ENTRYPOINT ["python3", "-m", "hal.server.jetson.main"]

# Default arguments (can be overridden at runtime)
# Users should provide: --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


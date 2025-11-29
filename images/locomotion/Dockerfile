# Production Dockerfile for Jetson Orin robot deployment
# This container combines inference logic and HAL server for production use on the robot
# Uses inproc ZMQ for same-process communication

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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/locomotion/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: Base image already includes PyTorch, NumPy, CUDA, cuDNN, TensorRT
# Note: pyzed requires ZED SDK to be installed on the system (via NVIDIA installer)
# If ZED SDK is not available, the code will gracefully fall back to mock mode
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
COPY hal/krabby-hal-client/dist/*.whl /tmp/wheels/
COPY hal/krabby-hal-server/dist/*.whl /tmp/wheels/
# Add krabby-hal-server-jetson wheel when created
# Note: krabby-hal-tools is excluded by default (only install when needed for debugging)
RUN pip3 install --no-cache-dir /tmp/wheels/krabby-hal-client-*.whl
RUN pip3 install --no-cache-dir /tmp/wheels/krabby-hal-server-*.whl

# Copy remaining code that's not in wheels
# Copy compute/parkour/ (policy wrapper and inference logic)
COPY compute/ /workspace/compute/

# Copy locomotion/jetson/ (HAL server and production runtime)
COPY locomotion/ /workspace/locomotion/

# Copy parkour/ (custom RSL-RL extensions)
# Note: policy_interface.py imports directly from parkour.scripts.rsl_rl, so we need the full path
COPY parkour/ /workspace/parkour/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Set entrypoint to production main.py
# This runs the InferenceRunner which coordinates HAL server and policy inference
ENTRYPOINT ["python3", "locomotion/jetson/main.py"]

# Default arguments (can be overridden at runtime)
# Users should provide: --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


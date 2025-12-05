# Dockerfile for Isaac Sim container
# This container combines inference logic and Isaac Sim HAL server for simulation testing
# Uses inproc ZMQ for same-process communication

FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Set working directory
WORKDIR /workspace

# Switch to root user for package installation
# The base image runs as non-root user (isaac-sim) by default, but apt-get requires root privileges
USER root

# Install system dependencies
# Most dependencies are included in Isaac Sim base image
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/isaacsim/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: Isaac Sim base image already includes Isaac Sim, Omni, PyTorch, NumPy
# Use --break-system-packages to bypass PEP 668 protection (acceptable in Docker containers)
RUN pip3 install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
# Note: Wheels must be built before building this Docker image
COPY hal/client/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/isaac/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
# Note: hal/server/jetson is not needed for Isaac Sim container
# Note: hal/tools is excluded by default (only install when needed for debugging)
RUN bash -c "pip3 install --no-cache-dir --break-system-packages /tmp/wheels/*.whl"

# Set entrypoint to HAL server with integrated inference
# This runs both Isaac Sim HAL server and policy inference in the same process
ENTRYPOINT ["python3", "-m", "hal.server.isaac.main"]

# Default arguments (can be overridden at runtime)
# Users should provide: --task, --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


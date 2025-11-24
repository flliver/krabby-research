# Dockerfile for ARM testing container (Jetson-compatible)
# This container includes inference logic + inference test runner for testing on ARM hardware
# The test runner simulates the game loop (inference logic) for testing purposes

FROM nvcr.io/nvidia/pytorch:25.10-py3-igpu

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/testing/arm/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy project code
COPY compute/ /workspace/compute/
COPY HAL/ /workspace/HAL/
# Copy parkour/ (custom RSL-RL extensions)
# Note: policy_interface.py imports directly from parkour.scripts.rsl_rl, so we need the full path
COPY parkour/ /workspace/parkour/
COPY tests/ /workspace/tests/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden)
CMD ["/bin/bash"]


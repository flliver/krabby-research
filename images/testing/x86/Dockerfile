# Dockerfile for x86 testing container
# This container includes inference logic + inference test runner for testing on x86 workstations
# The test runner simulates the game loop (inference logic) for testing purposes
# Uses inproc ZMQ for same-process communication
# NOTE: This is for testing/development only. NOT used in production.

FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    build-essential \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/testing/x86/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: Base image already includes PyTorch, NumPy, CUDA, cuDNN, TensorRT
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
COPY hal/krabby-hal-client/dist/*.whl /tmp/wheels/
COPY hal/krabby-hal-server/dist/*.whl /tmp/wheels/
COPY hal/krabby-hal-server-isaac/dist/*.whl /tmp/wheels/
# Note: krabby-hal-tools is excluded by default (only install when needed for debugging)
RUN bash -c "pip3 install --no-cache-dir /tmp/wheels/*.whl"

# Copy remaining code that's not in wheels
# Copy compute/ (policy interface and inference logic, including inference test runner)
COPY compute/ /workspace/compute/

# Copy parkour/ (custom RSL-RL extensions)
# Note: policy_interface.py imports directly from parkour.scripts.rsl_rl, so we need the full path
COPY parkour/ /workspace/parkour/

# Copy locomotion/ (for Jetson HAL server tests)
COPY locomotion/ /workspace/locomotion/

# Copy tests/ (test scenarios)
COPY tests/ /workspace/tests/

# Copy coverage configuration
COPY .coveragerc /workspace/.coveragerc

# Set Python path
ENV PYTHONPATH=/workspace

# Set entrypoint to inference test runner script
# This runs the test runner which simulates the game loop (inference logic) for testing
# Can be overridden to run pytest or other commands
ENTRYPOINT []

# Default command (can be overridden at runtime)
# For inference test: python3 compute/testing/main.py --checkpoint ... --action_dim ... --obs_dim ...
# For pytest: pytest tests/...
CMD ["/bin/bash"]


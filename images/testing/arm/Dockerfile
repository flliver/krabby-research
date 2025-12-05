# Dockerfile for ARM testing container
# This container includes inference logic + inference test runner for testing on ARM hardware
# The test runner simulates the game loop (inference logic) for testing purposes
# NOTE: This is for testing/development only. NOT used in production.
#
# Building:
#   On ARM testing environment (ARM64): docker build -f images/testing/arm/Dockerfile -t krabby-testing-arm:latest .
#   On x86_64: docker buildx build --platform linux/arm64 -f images/testing/arm/Dockerfile -t krabby-testing-arm:latest --load .

FROM nvcr.io/nvidia/pytorch:25.10-py3-igpu

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/testing/arm/requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Note: Base image already includes PyTorch, NumPy, CUDA, cuDNN, TensorRT
# Use Docker build cache mount for pip to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r /tmp/requirements.txt

# Copy and install wheels (built via 'make build-wheels')
COPY hal/client/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/isaac/dist/*.whl /tmp/wheels/
COPY hal/server/jetson/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
# Note: krabby-hal-tools is excluded by default (only install when needed for debugging)
RUN bash -c "pip3 install --no-cache-dir /tmp/wheels/*.whl"

# Copy remaining code that's not in wheels (e.g., compute/testing for test runner)
COPY compute/ /workspace/compute/

# Copy tests/ (test scenarios)
COPY tests/ /workspace/tests/

# Copy pytest configuration
COPY pytest.ini /workspace/pytest.ini

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


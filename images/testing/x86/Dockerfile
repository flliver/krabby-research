# Dockerfile for x86 testing container
# This container includes inference logic + inference test runner for testing on x86 workstations
# The test runner simulates the game loop (inference logic) for testing purposes
# Uses inproc ZMQ for same-process communication
# NOTE: This is for testing/development only. NOT used in production.

FROM nvcr.io/nvidia/pytorch:25.10-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
#
# GStreamer + GObject introspection (below): same purpose as images/isaacsim/Dockerfile—pip PyGObject
# in the test venv plus system typelibs and plugins so `make test` can run tests/unit/hal/test_gstreamer_runtime.py
# (software x264 + h264parse path). Not duplicated on images/locomotion/Dockerfile: that Jetson image has
# no PyGObject in requirements and relies on JetPack/L4T + ZED for on-robot pipelines, not this CI stack.
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    build-essential \
    libzmq3-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libcairo2-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    pkg-config \
    gobject-introspection \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/testing/x86/requirements.txt /tmp/requirements.txt

# Install Isaac Lab (all packages - required for parkour package)
# Requires local git cache (cloned via 'make isaaclab-cache')
# This allows seeing clone progress outside Docker and reusing the cache across builds
COPY .build-cache/isaaclab /workspace/isaaclab

# Validate Isaac Lab cache
RUN if [ ! -d /workspace/isaaclab ] || [ ! -d /workspace/isaaclab/.git ]; then \
        echo "ERROR: Isaac Lab cache not found or invalid. Run 'make isaaclab-cache' first."; \
        exit 1; \
    fi

# Make installation script executable
RUN chmod +x /workspace/isaaclab/isaaclab.sh

# Create virtual environment for Isaac Lab installation
# Use Python 3.12 to match system Python version
RUN python3.12 -m venv /workspace/testenv

# Upgrade pip and install requirements into venv
# Note: Base image already includes PyTorch, NumPy, CUDA, cuDNN, TensorRT
# Install typing-extensions and matplotlib before Isaac Lab to avoid compatibility issues
# Use Docker build cache mount for pip to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install Isaac Lab packages using the installation script
# Use Docker build cache mount for pip to speed up subsequent builds
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    cd /workspace/isaaclab && \
    TERM=xterm-256color DEBIAN_FRONTEND=noninteractive ./isaaclab.sh --install || \
    echo "Warning: isaaclab.sh --install exited with error"

# Copy and install wheels (built via 'make build-wheels')
COPY hal/client/dist/*.whl /tmp/wheels/
COPY controller/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/isaac/dist/*.whl /tmp/wheels/
COPY hal/server/jetson/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
COPY data_collection/dist/*.whl /tmp/wheels/
COPY teleop/edge/dist/*.whl /tmp/wheels/
COPY teleop/portal/dist/*.whl /tmp/wheels/
# Note: krabby-hal-tools is excluded by default (only install when needed for debugging)
# Install wheels into venv since that's where production code runs
RUN . /workspace/testenv/bin/activate && \
    bash -c "pip install --no-cache-dir /tmp/wheels/*.whl"

# Copy hal and controller source so coverage can report on our packages (not just site-packages).
# Reinstall hal in editable mode so tests run from /workspace/hal and coverage includes it.
COPY hal/ /workspace/hal/
RUN . /workspace/testenv/bin/activate && \
    pip install --no-cache-dir -e /workspace/hal/client -e /workspace/hal/server \
        -e /workspace/hal/server/isaac -e /workspace/hal/server/jetson

# Copy remaining code that's not in wheels (e.g., compute/testing for test runner)
COPY compute/ /workspace/compute/
COPY controller/ /workspace/controller/
COPY firmware/ /workspace/firmware/

# Copy tests/ (test scenarios)
COPY tests/ /workspace/tests/

# Copy pytest configuration
COPY pytest.ini /workspace/pytest.ini

# Copy coverage configuration
COPY .coveragerc /workspace/.coveragerc

# Copy test assets (checkpoints, test data, etc.) to generic test assets directory
# This directory can be used for any test content files
RUN mkdir -p /workspace/test_assets/checkpoints
COPY parkour/assets/weights/*.pt /workspace/test_assets/checkpoints/
COPY parkour/assets/weights/*.json /workspace/test_assets/checkpoints/

# Set Python path
# Entrypoint uses venv Python directly, so ENV is needed
ENV PYTHONPATH=/workspace
# Add venv to PATH so pytest and other commands use venv Python
ENV PATH=/workspace/testenv/bin:$PATH

# Set checkpoint path for tests (can be overridden at runtime if needed)
ENV PARKOUR_CHECKPOINT_PATH=/workspace/test_assets/checkpoints

# Set entrypoint to inference test runner script
# This runs the test runner which simulates the game loop (inference logic) for testing
# Can be overridden to run pytest or other commands
ENTRYPOINT []

# Default command (can be overridden at runtime)
# For inference test: python3 compute/testing/main.py --checkpoint ... --action_dim ... --obs_dim ...
# For pytest: pytest tests/...
CMD ["/bin/bash"]


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
    python3-venv \
    python3-pip \
    git \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY images/isaacsim/requirements.txt /tmp/requirements.txt

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
# Use system Python (whatever version the base image provides)
# Use --system-site-packages so venv can access Isaac Sim from base image
RUN python3 -m venv --system-site-packages /workspace/testenv

# Upgrade pip and install requirements into venv
# Install typing-extensions and matplotlib before Isaac Lab to avoid compatibility issues
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Install Isaac Lab packages using the installation script
# Use Docker build cache mount for pip to speed up subsequent builds
# With --system-site-packages, the venv can import isaacsim from base image,
# allowing the VSCode setup script to find ISAAC_PATH automatically
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    cd /workspace/isaaclab && \
    TERM=xterm-256color DEBIAN_FRONTEND=noninteractive ./isaaclab.sh --install || \
    echo "Warning: isaaclab.sh --install exited with error"

# Add Isaac Sim Python path to venv activation script
RUN echo "export PYTHONPATH=\"\$PYTHONPATH:/isaac-sim/python_packages\"" >> /workspace/testenv/bin/activate

# Copy and install wheels (built via 'make build-wheels')
# Note: Wheels must be built before building this Docker image
COPY hal/client/dist/*.whl /tmp/wheels/
COPY hal/server/dist/*.whl /tmp/wheels/
COPY hal/server/isaac/dist/*.whl /tmp/wheels/
COPY compute/parkour/dist/*.whl /tmp/wheels/
COPY parkour/dist/*.whl /tmp/wheels/
# Note: hal/server/jetson is not needed for Isaac Sim container
# Note: hal/tools is excluded by default (only install when needed for debugging)
# Install wheels into venv since that's where production code runs
RUN . /workspace/testenv/bin/activate && \
    bash -c "pip install --no-cache-dir /tmp/wheels/*.whl"

# Copy test assets (checkpoints, test data, etc.) to generic test assets directory
# This directory can be used for any test content files
RUN mkdir -p /workspace/test_assets/checkpoints
COPY parkour/assets/weights/*.pt /workspace/test_assets/checkpoints/
COPY parkour/assets/weights/*.json /workspace/test_assets/checkpoints/

# Copy tests and pytest configuration for running real inference tests
COPY tests/ /workspace/tests/
COPY pytest.ini /workspace/pytest.ini

# Set PYTHONPATH environment variable to include /workspace and Isaac Sim Python packages
# Entrypoint uses venv Python directly, so ENV is needed
ENV PYTHONPATH=/workspace:/isaac-sim/python_packages

# Set checkpoint path for tests
ENV PARKOUR_CHECKPOINT_PATH=/workspace/test_assets/checkpoints

# Set entrypoint to HAL server with integrated inference
# This runs both Isaac Sim HAL server and policy inference in the same process
# Use venv Python since that's where Isaac Lab packages are installed
ENTRYPOINT ["/workspace/testenv/bin/python", "-m", "hal.server.isaac.main"]

# Default arguments (can be overridden at runtime)
# Users should provide: --task, --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


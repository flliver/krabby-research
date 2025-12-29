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
# Use Isaac Sim's Python (3.11) to match Isaac Sim's Python version
# This avoids Python version mismatches when mixing venv packages with Isaac Sim
# Use --system-site-packages so venv can access Isaac Sim from base image
RUN /isaac-sim/python.sh -m venv --system-site-packages /workspace/testenv

# Upgrade pip and install requirements into venv
# Install typing-extensions and matplotlib before Isaac Lab to avoid compatibility issues
# Note: Do NOT install scipy in venv - Isaac Sim has its own bundled scipy that must be used
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip uninstall -y scipy || true  # Remove scipy if installed to avoid conflicts with Isaac Sim's bundled scipy

# Install Isaac Lab packages using the installation script
# Use Docker build cache mount for pip to speed up subsequent builds
# With --system-site-packages, the venv can import isaacsim from base image,
# allowing the VSCode setup script to find ISAAC_PATH automatically
RUN --mount=type=cache,target=/root/.cache/pip \
    . /workspace/testenv/bin/activate && \
    cd /workspace/isaaclab && \
    TERM=xterm-256color DEBIAN_FRONTEND=noninteractive ./isaaclab.sh --install || \
    echo "Warning: isaaclab.sh --install exited with error"

# Remove scipy after Isaac Lab installation (Isaac Lab may install it as a dependency)
# Isaac Sim has its own bundled scipy that must be used instead
RUN . /workspace/testenv/bin/activate && \
    pip uninstall -y scipy || true

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

# Copy parkour source so parkour_isaaclab is available
# Note: parkour_isaaclab is not included in the wheel, so we need the source
COPY parkour/ /workspace/parkour/

# Copy compute source so compute/testing is available (for testing tools)
# Note: compute/testing is included in the wheel but we want latest source for development
COPY compute/ /workspace/compute/

# Copy test assets (checkpoints, test data, etc.) to generic test assets directory
# This directory can be used for any test content files
RUN mkdir -p /workspace/test_assets/checkpoints
COPY parkour/assets/weights/*.pt /workspace/test_assets/checkpoints/
COPY parkour/assets/weights/*.json /workspace/test_assets/checkpoints/

# Copy tests and pytest configuration for running real inference tests
COPY tests/ /workspace/tests/
COPY pytest.ini /workspace/pytest.ini

# Copy standalone test runner (no pytest - runs tests directly with AppLauncher)
COPY images/isaacsim/test_runner.py /workspace/test_runner.py
COPY images/isaacsim/run_test_runner.sh /workspace/run_test_runner.sh
RUN chmod +x /workspace/run_test_runner.sh
# Copy HAL server wrapper script
COPY images/isaacsim/run_hal_server.sh /workspace/run_hal_server.sh
RUN chmod +x /workspace/run_hal_server.sh

# Set PYTHONPATH environment variable to include /workspace and Isaac Sim Python packages
# Entrypoint uses venv Python directly, so ENV is needed
ENV PYTHONPATH=/workspace:/isaac-sim/python_packages

# Set EXP_PATH for AppLauncher (required by Isaac Lab)
# This points to the base Python experience file
ENV EXP_PATH=/isaac-sim/apps/isaacsim.exp.base.python.kit

# Set checkpoint path for tests
ENV PARKOUR_CHECKPOINT_PATH=/workspace/test_assets/checkpoints

# Set entrypoint to HAL server with integrated inference
# This runs both Isaac Sim HAL server and policy inference in the same process
# Use wrapper script to properly initialize Isaac Sim
ENTRYPOINT ["/workspace/run_hal_server.sh"]

# Default arguments (can be overridden at runtime)
# Users should provide: --task, --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


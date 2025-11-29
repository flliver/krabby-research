# Dockerfile for Isaac Sim container
# This container combines inference logic and Isaac Sim HAL server for simulation testing
# Uses inproc ZMQ for same-process communication

FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Set working directory
WORKDIR /workspace

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
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Copy project code
# Copy compute/parkour/ (policy wrapper and inference logic)
COPY compute/ /workspace/compute/

# Copy hal/ (HAL types, ZMQ client/server, Isaac Sim HAL server)
COPY hal/ /workspace/hal/

# Copy locomotion/isaacsim/ (Isaac Sim container entry point)
COPY locomotion/ /workspace/locomotion/

# Copy parkour/ (custom RSL-RL extensions)
# Note: policy_interface.py imports directly from parkour.scripts.rsl_rl, so we need the full path
COPY parkour/ /workspace/parkour/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Set entrypoint to combined main.py
# This runs both Isaac Sim HAL server and policy inference in the same process
ENTRYPOINT ["python3", "locomotion/isaacsim/main.py"]

# Default arguments (can be overridden at runtime)
# Users should provide: --task, --checkpoint, --action_dim, --obs_dim
CMD ["--help"]


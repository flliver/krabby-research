# Isaac Sim Container

This directory contains the Dockerfile for the Isaac Sim container that combines inference logic and Isaac Sim HAL server for simulation testing.

## Overview

The Isaac Sim container combines:
- **Policy inference** (`compute/parkour/`) - Policy wrapper and model inference
- **HAL server** (`HAL/Isaac/`) - Isaac Sim HAL server with simulation integration
- **Inference runner** (`images/isaacsim/main.py`) - Combined control loop

All components communicate via **inproc ZMQ** (same process, zero-copy) for optimal performance.

## Building the Container

```bash
cd images/isaacsim
docker build -t krabby-isaacsim:latest .
```

**Note**: Requires NVIDIA NGC account and authentication to pull the Isaac Sim base image.

## Running the Container

```bash
docker run --rm --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-isaacsim:latest \
    --task Isaac-Parkour-Anymal-D-v0 \
    --checkpoint /workspace/checkpoints/checkpoint.pt \
    --action_dim 12 \
    --obs_dim <OBS_DIM> \
    --control_rate 100.0 \
    --device cuda \
    --observation_bind inproc://hal_observation \
    --command_bind inproc://hal_commands
```

## Configuration

- **Base image**: `nvcr.io/nvidia/isaac-sim:5.1.0` (Isaac Sim 5.1.0, Kit 107.3.3)
- **OS**: Ubuntu 22.04 (Isaac Sim requirement)
- **Architecture**: x86_64 (primary), aarch64 (supported but live-streaming not available)
- **Communication**: inproc ZMQ (same process)
- **Control rate**: 100+ Hz

## Dependencies

See `docs/DOCKER_DEPENDENCIES.md` for complete dependency list.

## Notes

- Requires Isaac Sim license/access (NVIDIA NGC account)
- GPU required for Isaac Sim rendering
- Large image size (~20GB+) due to Isaac Sim
- Display access needed for GUI (use `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`)
- Container runs as rootless user by default
- Live-streaming features not yet supported on aarch64 architecture
- All ZMQ communication uses inproc endpoints for same-process communication


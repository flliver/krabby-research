# Production Locomotion Container

This directory contains the production Dockerfile for the Jetson Orin robot deployment.

## Overview

The production container combines:
- **Policy inference** (`compute/parkour/`) - Policy wrapper and model inference
- **HAL server** (`locomotion/jetson/`) - Jetson HAL server with ZED camera integration
- **Inference runner** (`locomotion/jetson/inference_runner.py`) - Production control loop

All components communicate via **inproc ZMQ** (same process, zero-copy) for optimal performance.

## Building the Container

```bash
cd images/locomotion
docker build -t krabby-locomotion:latest .
```

## Running the Container

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-locomotion:latest \
    --checkpoint /workspace/checkpoints/checkpoint.pt \
    --action_dim 12 \
    --obs_dim <OBS_DIM> \
    --control_rate 100.0 \
    --device cuda \
    --observation_bind inproc://hal_observation \
    --command_bind inproc://hal_commands
```

## Configuration

- **Base image**: `nvcr.io/nvidia/pytorch:25.10-py3-igpu` (JetPack 6+)
- **Architecture**: ARM64 (aarch64) for Jetson
- **Communication**: inproc ZMQ (same process)
- **Control rate**: 100+ Hz

## Dependencies

See `DOCKER_DEPENDENCIES.md` for complete dependency list.

## Notes

- ZED SDK Python bindings (`pyzed`) are installed via pip, but require ZED SDK to be installed on the system (via NVIDIA installer)
- If ZED SDK is not available, the code will gracefully fall back to mock camera mode
- Container runs as production entry point (`locomotion/jetson/main.py`)
- All ZMQ communication uses inproc endpoints for same-process communication


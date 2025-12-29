# Isaac Sim Container

This directory contains the Dockerfile for the Isaac Sim container that combines inference logic and Isaac Sim HAL server for simulation testing.

## Overview

The Isaac Sim container combines:
- **Policy inference** (`compute/parkour/`) - Policy wrapper and model inference
- **HAL server** (`hal/Isaac/`) - Isaac Sim HAL server with simulation integration
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

## Running with Visual Display

You can run the HAL server inside Docker with visual display forwarded to your host. This allows you to see the Isaac Sim environment while the HAL server gathers observations, runs inference, and controls the robot.

**Prerequisites:**
1. Allow X11 access on your host (run once per session):
   ```bash
   xhost +local:docker
   # Or more securely (only for root):
   xhost +local:root
   ```

2. Ensure your DISPLAY environment variable is set:
   ```bash
   echo $DISPLAY  # Should show something like :0 or :1
   ```

3. **Test X11 forwarding** (optional but recommended):
   ```bash
   docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --entrypoint /bin/bash krabby-isaacsim:latest -c "apt-get update && apt-get install -y x11-apps && xeyes"
   ```
   This will run a simple X11 application (xeyes) to verify display forwarding is working. If you see the xeyes window appear on your host, X11 forwarding is configured correctly.

**Run HAL server with visual display (16 parallel robots, matching play.py):**

Single command (automatically configures X11 access):
```bash
xhost +local:docker 2>/dev/null; docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix krabby-isaacsim:latest --task Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0 --checkpoint /workspace/test_assets/checkpoints/unitree_go2_parkour_teacher.pt --action_dim 12 --obs_dim 753 --num_envs 16
```

Note: The `xhost +local:docker 2>/dev/null` part suppresses errors if X11 is already configured, and the semicolon ensures it runs before docker.

**Options:**
- `--num_envs 16` - 16 parallel robot simulations (default, matching play.py)
- `--real-time` - Run in real-time mode
- `--video` - Record videos
- `PYTHONUNBUFFERED=1` before `docker run` - Real-time output

**Troubleshooting:**
- If you see "No protocol specified" or "cannot connect to X server", make sure you've run `xhost +local:docker`
- If the display doesn't appear, check that `$DISPLAY` is set correctly on the host
- For remote X11 forwarding (SSH), you may need additional X11 forwarding setup

## Running Tests

Isaac Sim tests use a custom test runner (`test_runner.py`) instead of pytest to properly initialize Isaac Sim via AppLauncher and manage CUDA context.

**Run a specific test:**
```bash
PYTHONUNBUFFERED=1 timeout 300 docker run --rm --gpus all \
    --entrypoint /workspace/run_test_runner.sh \
    krabby-isaacsim:latest \
    test_isaacsim_noop
```

**Run all tests:**
```bash
make test-isaacsim
# Or: PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \
#     --entrypoint /workspace/run_test_runner.sh krabby-isaacsim:latest
```

**Options:**
- `PYTHONUNBUFFERED=1` - Real-time output (recommended)
- `timeout 300` - Kill after 5 minutes (single test) or `timeout 600` (all tests)
- `--gpus all` - Required for GPU access
- Last argument is test name (omit to run all)

**Adding a new test:**
1. Add function `run_test_your_test_name()` to `test_runner.py`
2. Register in `tests` dictionary in `main()`
3. Rebuild image: `make build-isaacsim-image`

**Available tests:**
- `test_isaacsim_noop` - Verify Isaac Sim initialization
- `test_isaacsim_hal_server_with_real_isaaclab` - Full integration with Isaac Lab environment
- `test_inference_latency_requirement` - Performance test (< 15ms latency, requires checkpoint)

## Notes

- Requires Isaac Sim license/access (NVIDIA NGC account)
- GPU required for Isaac Sim rendering
- Large image size (~20GB+) due to Isaac Sim
- Display access needed for GUI (use `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`)
- Container runs as rootless user by default
- Live-streaming features not yet supported on aarch64 architecture
- All ZMQ communication uses inproc endpoints for same-process communication


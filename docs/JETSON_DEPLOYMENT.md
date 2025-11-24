# Jetson Deployment Guide

This guide explains how to build and deploy the parkour policy runtime on Jetson Orin hardware.

## Prerequisites

- Jetson Orin device with JetPack 5.x installed
- ZED Mini camera (optional, for depth sensing)
- Docker with NVIDIA Container Toolkit installed
- Model checkpoint file (`.pt` format)

## Building Docker Images

### Build Locomotion Container

Build the production locomotion container for Jetson:

```bash
cd images/locomotion
docker build -t krabby-locomotion:latest --platform linux/arm64 .
```

### Build Testing Container (Optional)

For development and testing:

```bash
cd images/testing/arm
docker build -t krabby-testing-arm:latest --platform linux/arm64 .
```

## Running on Jetson

### Basic Usage

Run the production inference runner:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    python locomotion/jetson/main.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda \
        --control_rate 100.0
```

### With ZED Camera

If using ZED Mini camera, mount the camera device:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    --device /dev/video0 \
    krabby-locomotion:latest \
    python locomotion/jetson/main.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda \
        --control_rate 100.0
```

### Network Mode (Cross-Container Communication)

For running HAL server and client in separate containers:

**HAL Server Container:**
```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    -p 6000:6000 -p 6001:6001 -p 6002:6002 \
    --name hal-server \
    krabby-locomotion:latest \
    python locomotion/jetson/main.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda \
        --observation_bind tcp://*:6001 \
        --command_bind tcp://*:6002
```

**HAL Client Container (on same or different machine):**
```bash
docker run --rm --gpus all \
    --name hal-client \
    krabby-locomotion:latest \
    python -m compute.testing.inference_test_runner \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --observation_endpoint tcp://hal-server:6001 \
        --command_endpoint tcp://hal-server:6002
```

### In-Process Mode (Default)

By default, the inference runner uses in-process communication (`inproc://`) which is more efficient for single-container deployment:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    -v /dev:/dev \
    krabby-locomotion:latest \
    python locomotion/jetson/main.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda
```

## Testing and Verification

### Test Checkpoint Loading

Verify that checkpoints can be loaded on Jetson:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-testing-arm:latest \
    python locomotion/jetson/test_checkpoint_loading.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda
```

### Benchmark Inference Latency

Measure inference latency to ensure it meets real-time requirements (< 15ms):

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-testing-arm:latest \
    python locomotion/jetson/benchmark_inference.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --action_dim 12 \
        --obs_dim 753 \
        --device cuda \
        --iterations 100 \
        --warmup 10
```

### Run Integration Tests

Run integration tests:

```bash
docker run --rm --gpus all \
    krabby-testing-arm:latest \
    pytest tests/integration/test_jetson_hal.py -v
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Control which GPU to use (default: all)
- `HAL_BASE_PORT`: Base port for HAL endpoints (default: 6000)
- `LOG_LEVEL`: Logging level (default: INFO)

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:
- Reduce batch size if applicable
- Use TensorRT optimization (see `export_to_tensorrt.py`)
- Ensure no other processes are using GPU memory

### ZED Camera Not Detected

- Verify camera is connected: `lsusb | grep ZED`
- Check device permissions: `ls -l /dev/video*`
- Try running with `--device /dev/video0` explicitly

### High Latency

- Verify GPU is being used: Check logs for "CUDA available"
- Run benchmark to measure actual latency
- Consider TensorRT export for optimization
- Check system load and thermal throttling

## Performance Tuning

### TensorRT Optimization

For faster inference, export model to TensorRT:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-testing-arm:latest \
    python locomotion/jetson/export_to_tensorrt.py \
        --checkpoint /workspace/checkpoints/unitree_go2_parkour_teacher.pt \
        --output /workspace/checkpoints/model.trt \
        --obs_dim 753 \
        --precision fp16
```

### Control Rate Adjustment

Adjust control rate based on actual latency:

```bash
# If latency is consistently < 10ms, can run at 100 Hz
--control_rate 100.0

# If latency is 10-15ms, reduce to 80 Hz
--control_rate 80.0
```

## Notes

- Ensure Jetson has sufficient CUDA memory for model loading
- TensorRT export requires additional dependencies
- Benchmark results should show mean latency < 15ms for 100 Hz control
- For production, use in-process mode (`inproc://`) for lowest latency
- Network mode (`tcp://`) is useful for debugging and multi-container setups


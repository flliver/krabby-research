# Jetson HAL Server and Production Runtime

This directory contains the Jetson HAL server implementation and production runtime for robot deployment.

## Step 4.1: Verify Policy Inference on Jetson

### 1. Build Jetson-Compatible Docker Image

Build the ARM testing container:

```bash
cd images/testing/arm
docker build -t krabby-testing-arm:latest .
```

Or for production locomotion container:

```bash
cd images/locomotion
docker build -t krabby-locomotion:latest .
```

### 2. Test Loading Parkour Checkpoint

Test that checkpoints can be loaded on Jetson:

```bash
python locomotion/jetson/test_checkpoint_loading.py \
    --checkpoint /path/to/checkpoint.pt \
    --action_dim 12 \
    --obs_dim <OBS_DIM> \
    --device cuda
```

### 3. Benchmark Inference Latency

Benchmark inference to ensure it meets real-time requirements (< 15ms target):

```bash
python locomotion/jetson/benchmark_inference.py \
    --checkpoint /path/to/checkpoint.pt \
    --action_dim 12 \
    --obs_dim <OBS_DIM> \
    --device cuda \
    --iterations 100 \
    --warmup 10
```

### 4. Optionally Export to TensorRT

If inference latency is too high, export to TensorRT for optimization:

```bash
python locomotion/jetson/export_to_tensorrt.py \
    --checkpoint /path/to/checkpoint.pt \
    --output /path/to/output.trt \
    --obs_dim <OBS_DIM> \
    --precision fp16
```

## Running in Docker

To run tests in the Docker container:

```bash
docker run --rm --gpus all \
    -v /path/to/checkpoints:/workspace/checkpoints \
    krabby-testing-arm:latest \
    python locomotion/jetson/test_checkpoint_loading.py \
        --checkpoint /workspace/checkpoints/checkpoint.pt \
        --action_dim 12 \
        --obs_dim <OBS_DIM>
```

## Notes

- Ensure Jetson has sufficient CUDA memory for model loading
- TensorRT export requires additional dependencies and setup
- Benchmark results should show mean latency < 15ms for 100 Hz control


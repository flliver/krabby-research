# Krabby HAL Server - Jetson

HAL server implementation for Jetson robot deployment with integrated parkour policy inference.

## Overview

This package provides an entry point that runs both the Jetson HAL server and parkour inference client in the same process using inproc ZMQ for zero-copy communication.

### Architecture

```
┌─────────────────────────────────────────┐
│         Jetson Process                  │
│                                         │
│  ┌──────────────┐    inproc (ZMQ)     │
│  │ HAL Server   │◄──────────────────┐  │
│  │ (main thread)│                   │  │
│  │  - ZED camera│                   │  │
│  │  - Sensors   │                   │  │
│  │  - Actuators │                   │  │
│  └──────────────┘                   │  │
│         │                           │  │
│         │ publishes observations    │  │
│         │ receives commands         │  │
│         │                           │  │
│  ┌──────────────────────────────────┴─┐│
│  │ Parkour Inference Client           ││
│  │ (separate thread)                  ││
│  │  - Polls observations              ││
│  │  - Runs policy inference           ││
│  │  - Sends joint commands            ││
│  └────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

## Installation

### From source (development)

```bash
cd hal/server/jetson
pip install -e .
```

### With optional dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Command line

After installation, use the `krabby-hal-server-jetson` command:

```bash
krabby-hal-server-jetson \
  --checkpoint /path/to/model.pt \
  --action_dim 12 \
  --obs_dim 753 \
  --control_rate 100.0 \
  --device cuda
```

### Python module

```bash
python -m hal.server.jetson.main \
  --checkpoint /path/to/model.pt \
  --action_dim 12 \
  --obs_dim 753
```

### Arguments

**Required:**
- `--checkpoint`: Path to model checkpoint file
- `--action_dim`: Action dimension (typically 12)
- `--obs_dim`: Observation dimension (typically 753)

**Optional:**
- `--control_rate`: Control loop rate in Hz (default: 100.0)
- `--device`: Device for inference, `cuda` or `cpu` (default: cuda)
- `--observation_bind`: Observation endpoint (default: `inproc://hal_observation`)
- `--command_bind`: Command endpoint (default: `inproc://hal_commands`)

## Components

### HAL Server (`hal.server.jetson.JetsonHalServer`)
- Integrates with ZED camera for depth perception
- Interfaces with real sensors (IMU, encoders)
- Applies commands to actuators (motors)
- Publishes observations via ZMQ PUB socket
- Receives joint commands via ZMQ PULL socket

### Parkour Inference Client (`compute.parkour.inference_client.ParkourInferenceClient`)
- Runs in separate thread
- Polls observations from HAL server
- Runs parkour policy inference
- Sends joint commands back to HAL server

## Hardware Requirements

- **NVIDIA Jetson** (Orin, AGX Xavier, or compatible)
- **ZED Camera** (requires ZED SDK and pyzed)
- **Robot Hardware** (motors, IMU, encoders)

## Development

### Project Structure

```
hal/server/jetson/
├── pyproject.toml      # Package configuration
├── README.md           # This file
├── __init__.py         # Package init
├── main.py             # Entry point with integrated inference
└── hal_server.py       # JetsonHalServer implementation
```

### Running Tests

```bash
pytest tests/integration/test_jetson_hal.py
```

**Note:** Most tests require Jetson hardware or ZED SDK and are skipped in x86 environments.

## Notes

- This package uses **inproc ZMQ** by default for same-process communication (zero-copy, high performance)
- For distributed deployment, use TCP endpoints instead
- The parkour inference client runs on a separate thread to avoid blocking the sensor loop
- Camera, sensors, and actuators are initialized during startup

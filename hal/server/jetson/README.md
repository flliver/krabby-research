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
  --checkpoint /path/to/model.pt
```

### Python module

```bash
python -m hal.server.jetson.main --checkpoint /path/to/model.pt
```

### Arguments

**Required:**
- `--checkpoint`: Path to model checkpoint file

**Optional:**
- `--log-level`: Python logging level for this process (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; default: `INFO`)

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

## ZED IMU → body-frame state

The ZED 2i's onboard IMU supplies the model's body-frame angular velocity and
orientation:

- After each grab, `ZedCamera` reads the IMU sample aligned to that frame
  (`TIME_REFERENCE.IMAGE`) via `zed_imu.parse_zed_imu_data()` — angular velocity
  converted deg/s → **rad/s**, attitude quaternion **(x, y, z, w)**, sensor frame.
- `JetsonHalServer.set_observation()` rotates it into the robot base frame with
  the primary camera's catalog `SensorPose` (`zed_imu.apply_mount_to_imu_sample`)
  and populates `HardwareObservations.base_ang_vel_b` / `base_quat_w`. The same
  mount quaternion drives ZED tracking linear velocity and the Isaac sim path, so
  update the mount pose in the sensor catalog, not here.
- No IMU sample: observations keep the zero angular velocity / identity quaternion
  defaults.

### Bench verification

With just the ZED on USB (no chassis), run `scripts/zed_imu_probe.py` on the Orin
to verify the installed pyzed's API names and units, then tilt the camera by hand
and confirm `base_ang_vel_b` reacts (`scripts/hal_imu_watch.py` tails it from HAL).

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
├── main.py             # Entry point; routes on --control-source
├── main_gamepad.py     # gamepad: HAL over TCP for krabby-uno (no torch/teleop imports)
├── main_model.py       # inference / portal: policy client and/or WebRTC teleop
├── runtime.py          # HAL setup, data collector, and control loop shared by both
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

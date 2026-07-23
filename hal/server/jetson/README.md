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

The ZED 2i's onboard IMU (Bosch BMI088) supplies the model's body-frame
angular velocity and orientation until the MCU IMU (BMI270, M16) lands:

- `ZedCamera.get_imu()` (`zed_camera.py`) reads one sample per observation tick
  via pyzed `get_sensors_data(..., TIME_REFERENCE.IMAGE)` and returns a
  `ZedImuSample`: angular velocity (converted deg/s → **rad/s**), linear
  acceleration (m/s²), and the IMU-integrated attitude quaternion (**x, y, z, w**),
  all in the **camera** frame. It returns `None` when the fetch fails.
- `JetsonHalServer.set_observation()` rotates the sample into the robot body
  frame and populates `HardwareObservations.base_ang_vel_b` and `base_quat_w`
  every tick. The mapper (`compute/parkour/mappers/hardware_to_model.py`)
  derives roll/pitch from `base_quat_w`.

### Mount pose and the camera→body rotation contract

The IMU reports in the camera's coordinate frame; a fixed rotation matrix
`r_camera_to_body` maps it into the body frame (`x_body = R @ x_camera`).
It is loaded at server construction from `config/zed_mount.yaml` (packaged
with this module) and can be overridden per robot with the
`KRABBY_ZED_MOUNT_YAML` env var pointing at another YAML. A missing config
falls back to identity; an invalid matrix (non-orthonormal, reflection, wrong
shape) raises at startup.

**Assumed mount pose (M17 default):** ZED at the front-center of the krab
body, axes aligned with the body (identity rotation). The ZED is opened with
`COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP`; the body frame is X forward, Y left,
Z up. When the physical mount changes, update the matrix and the pose comment
in `config/zed_mount.yaml` (or the file `KRABBY_ZED_MOUNT_YAML` points to) —
no code change needed. M15 Task 2 refines the pose against the V0.2 chassis.

### Failure behavior

No IMU-capable camera, or a failed sample fetch, never crashes the loop:
observations carry zero angular velocity and the identity quaternion
(`[0, 0, 0, 1]` xyzw — "stationary and level"). Missing samples log a
rate-limited WARNING (first miss, then every 100th) and increment
`_imu_miss_count`; a non-advancing sensor timestamp logs INFO once until it
recovers.

### Bench verification

With just the ZED on USB (no chassis), run `scripts/zed_imu_probe.py`
on the Orin to verify the installed pyzed's API names and units, then tilt the
camera by hand and confirm `base_ang_vel_b` reacts and roll/pitch
(`proprioceptive[3:5]` in the mapper) track the tilt.

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

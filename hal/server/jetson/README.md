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

## ZED onboard IMU → body-frame state

The primary ZED 2i's onboard IMU (Bosch BMI088) supplies the model's body-frame
angular velocity and orientation until M16 adds the canonical BMI270 on the MCU.
After M16 the ZED IMU stays as a sanity check / failover.

### Data path

```
ZedCamera.grab()                                 # hal/server/jetson/zed_camera.py
  └─ _update_imu()
       get_sensors_data(_, TIME_REFERENCE.IMAGE)  # sample aligned to the frame
       parse_zed_imu_data(...)                     # hal/server/jetson/zed_imu.py
         · gyro deg/s → rad/s
         · orientation quaternion (x, y, z, w)
         → ZedImuSample{ base_quat_w, base_ang_vel_b }

JetsonHalServer.set_observation()                 # hal/server/jetson/hal_server.py
  └─ _primary_zed_imu_sample()
       get_imu_sample()                            # last sample from the grab above
       apply_mount_to_imu_sample(sample, mount)    # camera frame → robot body frame
  → HardwareObservations.base_quat_w   (4,) xyzw
    HardwareObservations.base_ang_vel_b (3,) rad/s

compute/parkour/mappers/hardware_to_model.py
  └─ _extract_proprioceptive(): base_quat_w → roll/pitch at proprioceptive[3:5]
```

`get_imu_sample()` returns the sample captured by the most recent `grab()`; the
ZED IMU runs at 400 Hz and `TIME_REFERENCE.IMAGE` returns the sample aligned with
the latest frame, so no separate IMU poll is needed.

### Camera → body rotation (mount pose)

The IMU reports in the **camera frame**, not the robot body frame. The fixed
camera→body transform is carried as the primary catalog row's `SensorPose`
quaternion (`JETSON_SENSOR_CATALOG` `is_primary` row, `pose` = base→sensor), and
`apply_mount_to_imu_sample()` applies it:

- orientation: `q_world_base = q_world_sensor * conj(q_base_sensor)`
- angular velocity: `omega_base = R_base_sensor @ omega_sensor`

A pose of `(qx, qy, qz, qw) = (0, 0, 0, 1)` is identity (camera frame == body
frame) and is short-circuited as a no-op. **Assumed mount pose:** ZED 2i at the
front-center of the krab body, camera forward aligned with robot +X, camera up
with robot +Z. The default is identity; M15 Task 2 refines this against the
actual camera location on the V0.2 chassis (and moves the sim camera to match).

**To update the rotation when the mount changes:** set the `pose` quaternion on
the primary `JETSON_SENSOR_CATALOG` row to the new base→sensor rotation. No code
change is needed — the transform reads from the catalog at every tick.

### Failure handling

When the primary ZED has an IMU but a given tick has no sample
(`get_sensors_data` non-SUCCESS, or the parse rejects it),
`set_observation()` emits the "stationary" fallback — `base_ang_vel_b = [0,0,0]`,
`base_quat_w = [0,0,0,1]` — increments `_imu_miss_count`, and logs a WARNING
throttled to every 100th miss. The control loop never crashes; the model reads
zero angular velocity + identity quaternion as "stationary", the safe default.
When no ZED IMU is present at all (`_zed_imu_active` is False) the fields stay at
their placeholder defaults and no warning fires.

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

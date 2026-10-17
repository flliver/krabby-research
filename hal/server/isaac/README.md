# Krabby HAL Server - IsaacSim

HAL server implementation for IsaacSim with integrated parkour policy inference.

## Overview

This package provides an entry point that runs both the IsaacSim HAL server and parkour inference client in the same process using inproc ZMQ for zero-copy communication.

### Architecture

```
┌─────────────────────────────────────────┐
│         IsaacSim Process                │
│                                         │
│  ┌──────────────┐    inproc (ZMQ)     │
│  │ HAL Server   │◄──────────────────┐  │
│  │ (main thread)│                   │  │
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

### From PyPI

```bash
pip install krabby-hal-server-isaac==0.1.2
```

Pulls **`krabby-data-collection`**, **`krabby-controller`**, **`krabby-teleop-edge`**, and other
runtime deps from PyPI. The **`krabby-isaacsim`** Docker image bakes the same wheels at
**`make build-isaacsim-image`** time (local **`make build-wheels`**, not a live PyPI fetch during
`docker build`).

### From source (development)

```bash
cd hal/server/isaac
pip install -e .
```

### With optional dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Command line

After installation, use the `krabby-hal-server-isaac` command:

```bash
krabby-hal-server-isaac \
  --task Isaac-Anymal-D-v0 \
  --checkpoint /path/to/model.pt \
  --action_dim 12 \
  --obs_dim 753
```

### Python module

```bash
python -m hal.server.isaac.main \
  --task Isaac-Anymal-D-v0 \
  --checkpoint /path/to/model.pt \
  --action_dim 12 \
  --obs_dim 753
```

### Arguments

**Required:**
- `--task`: IsaacSim task name (e.g., `Isaac-Anymal-D-v0`)
- `--checkpoint`: Path to model checkpoint file
- `--action_dim`: Action dimension (typically 12)
- `--obs_dim`: Observation dimension (typically 753)

**Optional:**
- `--inference_device`: Device for inference, `cuda` or `cpu` (default: cuda)
- `--observation_bind`: Observation endpoint (default: `inproc://hal_observation`)
- `--command_bind`: Command endpoint (default: `inproc://hal_commands`)
- `--backend`: `direct` (default) or `fullsim` (three-board host firmware on the hex env)
- `--firmware-transport`: with `fullsim`, `in_process` (default) or `serial_sdk` (KrabbyMCUSDK over PTY)

Plus all AppLauncher arguments for IsaacSim configuration.

### Fullsim firmware transport (crab hex)

**`--backend fullsim`** runs three-board host firmware on the hex scene (18 DOFs). The
**`krabby-isaacsim`** entrypoint (`run_hal_server.sh`) adds **`--backend fullsim`** automatically
for crab hex launches (`--usd`, `--robot hex`, or a `CrabHex` task) when you omit **`--backend`**.
Pass **`--backend direct`** to force the old Isaac-only path on hex. Default
**`--firmware-transport in_process`** is usually enough; use **`serial_sdk`** only when you
need the Jetson/GUI pyserial path in the loop.

| Transport | Behavior | Pick when |
|-----------|----------|-----------|
| **`in_process`** (default) | Host bridge jog + firmware tick; no PTY | Policy/firmware Isaac work, throughput (`test_hal_fullsim_teleop_e2e.py`) |
| **`serial_sdk`** | **`KrabbyMCUSDK`** over PTY (`B`/`J`/`H` wire) | Debug teleop, SDK reader, serial framing before hardware (`test_hal_fullsim_serial_sdk.py`; Unix PTY only) |
| **`direct`** backend (no fullsim) | Isaac actions only, no host firmware | Default HAL; not a `--firmware-transport` target |

```bash
python -m hal.server.isaac.main --task <crab-hex-task> --backend fullsim \
  --firmware-transport in_process …
```

Jetson production opens real USB with **`KrabbyMCUSDK`**; it does not use these flags.
**`serial_sdk`** on Isaac uses sparse `B` batches on the PTY shim; Jetson still sends full
18-joint batches on hardware USB. PTY setup for GUI/scripts:
[`firmware/sim/VIRTUAL_SERIAL.md`](../../../firmware/sim/VIRTUAL_SERIAL.md). Isaac firmware
pytest: `./images/isaacsim/run_firmware_isaac_pytest.sh` (mounts live `hal/` / `compute/`).

Fleet registration and teleop check (simulated krab, firmware in the loop):
[`fleet/FULLSIM-FLEET-TELEOP-SMOKE.md`](../../../fleet/FULLSIM-FLEET-TELEOP-SMOKE.md).

## Components

### HAL Server (`hal.server.isaac.IsaacSimHalServer`)
- Extracts observations from IsaacSim environment
- Publishes observations via ZMQ PUB socket
- Receives joint commands via ZMQ PULL socket
- Applies commands to simulation

### Parkour Inference Client (`compute.parkour.inference_client.ParkourInferenceClient`)
- Runs in separate thread
- Polls observations from HAL server
- Runs parkour policy inference
- Sends joint commands back to HAL server

## Development

### Project Structure

```
hal/server/isaac/
├── pyproject.toml      # Package configuration
├── README.md           # This file
├── __init__.py         # Package init
├── main.py             # Entry point with integrated inference
└── hal_server.py       # IsaacSimHalServer implementation

compute/parkour/
├── inference_client.py # Parkour inference client
├── policy_interface.py # Policy model interface
├── types.py            # Parkour-specific types
└── mappers/            # HW ↔ Model mappers
```

### Running Tests

```bash
pytest tests/
```

## Notes

- This package uses **inproc ZMQ** by default for same-process communication (zero-copy, high performance)
- For distributed deployment (server and client on separate machines), use TCP endpoints instead
- The parkour inference client runs on a separate thread to avoid blocking the simulation loop

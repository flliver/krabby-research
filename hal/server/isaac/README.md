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
  --obs_dim 753 \
  --control_rate 100.0 \
  --device cuda
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
- `--control_rate`: Control loop rate in Hz (default: 100.0)
- `--device`: Device for inference, `cuda` or `cpu` (default: cuda)
- `--observation_bind`: Observation endpoint (default: `inproc://hal_observation`)
- `--command_bind`: Command endpoint (default: `inproc://hal_commands`)

Plus all AppLauncher arguments for IsaacSim configuration.

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

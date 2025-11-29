# Project Folder Layout

This document describes the folder structure under `/home/shriop/projects/krabs/krabby-research/`. The current structure reflects the current implementation, and will be extended with additional components in the future.

## Overview

Currently, the project includes:
- **`hal/`**: Hardware Abstraction Layer components, organized into wheel-based packages:
  - **`krabby-hal-client`**: Client implementation and types (installed via wheel)
  - **`krabby-hal-server`**: Base server class (installed via wheel)
  - **`krabby-hal-server-isaac`**: IsaacSim server implementation (installed via wheel)
  - **`krabby-hal-server-jetson`**: Jetson server implementation (installed via wheel)
  - **`krabby-hal-tools`**: Debugging tools (installed via wheel)
- **`compute/`**: Production inference and computation logic:
  - **`compute/parkour/`**: Production-ready inference implementation (used in production container)
- **`locomotion/`**: Production runtime that combines inference logic and HAL server for robot deployment

**Key distinction**: 
- **Game loop** = The core inference logic (poll HAL → build observation → run inference → send command)
  - Production: `locomotion/jetson/inference_runner.py::InferenceRunner.run()`

All containers use inproc ZMQ for communication within the same process:
- **Production container** (`images/locomotion/`): Combines inference (`compute/parkour/`) and HAL server (`locomotion/jetson/`) to run on the actual robot (Jetson/ARM). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-jetson`
- **IsaacSim container** (`images/isaacsim/`): Combines inference (`compute/parkour/`) and HAL server (`krabby-hal-server-isaac`) for simulation (x86). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`
- **Testing containers** (`images/testing/x86/` and `images/testing/arm/`): Containers for running tests and development. Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`

These are separate from the existing `parkour/` directory which contains model-specific training and evaluation code.

## Directory Structure

```
krabby-research/
├── parkour/                          # Existing parkour model code (unchanged)
│   ├── scripts/rsl_rl/               # Training and evaluation scripts
│   ├── parkour_isaaclab/             # IsaacLab environment code
│   └── parkour_tasks/                # Task configurations
│
├── hal/                              # Hardware Abstraction Layer
│   ├── __init__.py                   # Minimal stub (packages installed via wheels or editable mode)
│   │
│   ├── krabby-hal-client/            # HAL client package (single source of truth)
│   │   ├── pyproject.toml
│   │   └── hal/
│   │       ├── client/
│   │       │   ├── client.py         # HalClient (ZMQ logic black-boxed)
│   │       │   └── config.py          # HalClientConfig
│   │       ├── observation/          # Observation types (NavigationCommand, ParkourObservation, etc.)
│   │       └── commands/             # Command types (JointCommand, InferenceResponse, etc.)
│   │
│   ├── krabby-hal-server/            # HAL server base package (wheel source)
│   │   ├── pyproject.toml
│   │   └── hal/
│   │       └── server/
│   │           ├── server.py         # HalServerBase (ZMQ logic black-boxed)
│   │           └── config.py          # HalServerConfig
│   │
│   ├── krabby-hal-server-isaac/      # IsaacSim HAL server package (single source of truth)
│   │   ├── pyproject.toml
│   │   └── hal/
│   │       └── isaac/
│   │           ├── hal_server.py     # IsaacSimHalServer (extends HalServerBase)
│   │           └── main.py            # Entry point (registered as console script)
│   │
│   ├── krabby-hal-server-jetson/    # Jetson HAL server package (wheel source)
│   │   ├── pyproject.toml
│   │   └── hal/
│   │       └── jetson/
│   │           └── hal_server.py     # JetsonHalServer (extends HalServerBase)
│   │
│   └── krabby-hal-tools/           # HAL debugging tools package (single source of truth)
│       ├── pyproject.toml
│       └── hal/
│           └── tools/
│               └── hal_dump.py       # CLI tool (registered as console script)
│
├── compute/                          # Inference and computation logic (current)
│   └── parkour/                      # Parkour inference implementation (used in production)
│       ├── __init__.py
│       ├── policy_interface.py     # Parkour policy inference interface
│       └── model_loader.py           # Model loading and checkpoint management
│
├── locomotion/                       # Production runtime
│   ├── jetson/                       # Jetson-specific runtime
│   │   ├── hal_server.py             # JetsonHalServer implementation (real sensors, source file packaged in krabby-hal-server-jetson wheel)
│   │   ├── inference_runner.py       # Production inference orchestration
│   │   ├── camera.py                 # ZED camera integration
│   │   └── main.py                   # Production entry point (runs inference + HAL server on Jetson)
│   └── isaacsim/                     # IsaacSim runtime (if exists)
│       └── main.py                   # IsaacSim entry point
│
├── images/                           # OS images, Dockerfiles, and container configs (current)
│   ├── locomotion/                   # Production container (Jetson: inference + HAL server, inproc ZMQ)
│   │   ├── Dockerfile                # Jetson-compatible Dockerfile
│   │   └── requirements.txt
│   ├── isaacsim/                     # IsaacSim container (inference + HAL server, inproc ZMQ)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── testing/                      # Testing containers
│       ├── x86/                      # x86 testing container
│       │   ├── Dockerfile
│       │   └── requirements.txt
│       └── arm/                      # ARM testing container
│           ├── Dockerfile
│           └── requirements.txt
│
└── scripts/                          # Deployment and utility scripts (current)
    └── deploy/                       # Deployment scripts
        ├── run_isaac_simulation.sh  # Launch IsaacSim HAL server
        └── run_locomotion.sh        # Launch locomotion container (Jetson, inproc ZMQ)
```

## Key Points

### HAL Package Structure (Wheel-based)
- **`hal/krabby-hal-client/`**: HAL client package (installed via wheel)
  - `hal/client/client.py`: HalClient implementation (ZMQ black-boxed)
  - `hal/client/config.py`: HalClientConfig
  - `hal/observation/`: Observation types (NavigationCommand, ParkourObservation, ParkourModelIO)
  - `hal/commands/`: Command types (JointCommand, InferenceResponse)
  
- **`hal/krabby-hal-server/`**: HAL server base package (installed via wheel)
  - `hal/server/server.py`: HalServerBase implementation (ZMQ black-boxed)
  - `hal/server/config.py`: HalServerConfig
  
- **`hal/krabby-hal-server-isaac/`**: IsaacSim HAL server package (installed via wheel)
  - `hal/isaac/hal_server.py`: IsaacSimHalServer (extends HalServerBase)
  - `hal/isaac/main.py`: Entry point (console script: `krabby-hal-server-isaac`)
  
- **`hal/krabby-hal-server-jetson/`**: Jetson HAL server package (installed via wheel)
  - `hal/jetson/hal_server.py`: JetsonHalServer (extends HalServerBase)
  
- **`hal/krabby-hal-tools/`**: HAL debugging tools package (installed via wheel)
  - `hal/tools/hal_dump.py`: CLI tool (console script: `hal-dump`)

### Single Source of Truth with Editable Installs

The HAL components use a **single source of truth** approach:

- **Source files** are located in `hal/krabby-hal-*/` directories (e.g., `hal/krabby-hal-client/hal/observation/`, `hal/krabby-hal-server-isaac/hal/isaac/`)
- **No duplicate source directories** - all code lives in the wheel package directories
- **Editable installs for development**: Run `make install-editable` to install packages in editable mode
  - This allows you to edit files in `hal/krabby-hal-*/` directories and see changes immediately
  - No need to rebuild wheels during development
- **Wheel builds for distribution**: Run `make build-wheels` to create distributable wheels
- **Production/Docker**: Install wheels from `hal/krabby-hal-*/dist/*.whl`

**Development workflow:**
```bash
# Install packages in editable mode (one-time setup)
make install-editable

# Now you can edit files in hal/krabby-hal-*/ and changes are immediately available
# No need to rebuild or reinstall

# To build wheels for distribution/Docker
make build-wheels
```

### Other Components
- **`compute/parkour/`**: Production inference logic (used in production container)
- **`locomotion/jetson/`**: Production runtime combining inference and HAL server for robot deployment
- **`images/locomotion/`**: Production container that runs on the robot (uses wheels)
- **`images/isaacsim/`**: IsaacSim container for simulation (uses wheels)
- **`images/testing/`**: Testing containers for x86 and ARM platforms (uses wheels)


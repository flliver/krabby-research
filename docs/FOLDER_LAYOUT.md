# Project Folder Layout

This document describes the folder structure under `/home/shriop/projects/krabs/krabby-research/`. The current structure reflects the current implementation, and will be extended with additional components in the future.

## Overview

Currently, the project includes:
- **`hal/`**: Hardware Abstraction Layer components, organized into wheel-based packages:
  - **`hal/client/`**: Client implementation and types (package: `krabby-hal-client`, installed via wheel)
  - **`hal/server/`**: Base server class (package: `krabby-hal-server`, installed via wheel)
  - **`hal/server/isaac/`**: IsaacSim server implementation (package: `krabby-hal-server-isaac`, installed via wheel)
  - **`hal/server/jetson/`**: Jetson server implementation (package: `krabby-hal-server-jetson`, installed via wheel)
  - **`hal/tools/`**: Debugging tools (package: `krabby-hal-tools`, installed via wheel)
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
│   ├── client/                       # HAL client package (package: krabby-hal-client)
│   │   ├── __init__.py               # Re-exports HalClient, HalClientConfig
│   │   ├── client.py                 # HalClient (ZMQ logic black-boxed)
│   │   ├── config.py                 # HalClientConfig
│   │   ├── observation/              # Observation types (NavigationCommand only)
│   │   │   ├── __init__.py
│   │   │   └── types.py
│   │   ├── commands/                 # Command types (empty, kept for future use)
│   │   │   ├── __init__.py
│   │   │   └── types.py
│   │   ├── data_structures/           # Hardware data structures
│   │   │   ├── __init__.py
│   │   │   └── hardware.py           # KrabbyHardwareObservations, KrabbyDesiredJointPositions
│   │   └── pyproject.toml
│   │
│   ├── server/                       # HAL server base package (package: krabby-hal-server)
│   │   ├── __init__.py               # Re-exports HalServerBase, HalServerConfig
│   │   ├── server.py                 # HalServerBase (ZMQ logic black-boxed)
│   │   ├── config.py                 # HalServerConfig
│   │   ├── pyproject.toml
│   │   │
│   │   ├── isaac/                    # IsaacSim HAL server (package: krabby-hal-server-isaac)
│   │   │   ├── __init__.py           # Re-exports IsaacSimHalServer
│   │   │   ├── hal_server.py         # IsaacSimHalServer (extends HalServerBase)
│   │   │   ├── main.py               # Entry point (console script: krabby-hal-server-isaac)
│   │   │   └── pyproject.toml
│   │   │
│   │   └── jetson/                    # Jetson HAL server (package: krabby-hal-server-jetson)
│   │       ├── __init__.py            # Re-exports JetsonHalServer
│   │       ├── hal_server.py          # JetsonHalServer (extends HalServerBase)
│   │       └── pyproject.toml
│   │
│   └── tools/                        # HAL debugging tools (package: krabby-hal-tools)
│       ├── __init__.py
│       ├── hal_dump.py                # CLI tool (console script: hal-dump)
│       └── pyproject.toml
│
├── compute/                          # Inference and computation logic (current)
│   └── parkour/                      # Parkour inference implementation (used in production)
│       ├── __init__.py
│       ├── policy_interface.py       # Parkour policy inference interface
│       ├── types.py                   # Parkour-specific types (ParkourObservation, ParkourModelIO, InferenceResponse)
│       └── mappers/                   # Data mappers (hardware ↔ model)
│           ├── __init__.py
│           ├── hardware_to_model.py  # KrabbyHardwareObservations → ParkourObservation
│           └── model_to_hardware.py   # InferenceResponse → KrabbyDesiredJointPositions
│
├── locomotion/                       # Production runtime
│   ├── jetson/                       # Jetson-specific runtime
│   │   ├── inference_runner.py       # Production inference orchestration
│   │   ├── camera.py                 # ZED camera integration
│   │   └── main.py                   # Production entry point (runs inference + HAL server on Jetson)
│   │                                 # Note: JetsonHalServer is in hal/server/jetson/hal_server.py
│   └── isaacsim/                     # IsaacSim runtime
│       └── main.py                   # IsaacSim entry point
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── helpers.py                    # Test helpers (create_dummy_hw_obs, etc.)
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
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

The HAL packages are organized with a clean directory structure that matches the import namespace:

- **`hal/client/`**: HAL client package (package name: `krabby-hal-client`, installed via wheel)
  - `hal/client/client.py`: HalClient implementation (ZMQ black-boxed)
  - `hal/client/config.py`: HalClientConfig
  - `hal/client/__init__.py`: Re-exports `HalClient`, `HalClientConfig` for cleaner imports
  - `hal/client/observation/`: Observation types (NavigationCommand only - generic HAL type)
  - `hal/client/commands/`: Command types (empty, kept for future generic types)
  - `hal/client/data_structures/`: Hardware data structures (KrabbyHardwareObservations, KrabbyDesiredJointPositions)
  
**Model-specific types** (ParkourObservation, ParkourModelIO, InferenceResponse, etc.) are in `compute/parkour/types.py`.

**Mappers** for converting between hardware and model formats are in `compute/parkour/mappers/`.
  
- **`hal/server/`**: HAL server base package (package name: `krabby-hal-server`, installed via wheel)
  - `hal/server/server.py`: HalServerBase implementation (ZMQ black-boxed)
  - `hal/server/config.py`: HalServerConfig
  - `hal/server/__init__.py`: Re-exports `HalServerBase`, `HalServerConfig` for cleaner imports
  
- **`hal/server/isaac/`**: IsaacSim HAL server package (package name: `krabby-hal-server-isaac`, installed via wheel)
  - `hal/server/isaac/hal_server.py`: IsaacSimHalServer (extends HalServerBase)
  - `hal/server/isaac/main.py`: Entry point (console script: `krabby-hal-server-isaac`)
  - `hal/server/isaac/__init__.py`: Re-exports `IsaacSimHalServer` for cleaner imports
  
- **`hal/server/jetson/`**: Jetson HAL server package (package name: `krabby-hal-server-jetson`, installed via wheel)
  - `hal/server/jetson/hal_server.py`: JetsonHalServer (extends HalServerBase)
  - `hal/server/jetson/__init__.py`: Re-exports `JetsonHalServer` for cleaner imports
  
- **`hal/tools/`**: HAL debugging tools package (package name: `krabby-hal-tools`, installed via wheel)
  - `hal/tools/hal_dump.py`: CLI tool (console script: `hal-dump`)

**Import Patterns:**
```python
# HAL client/server
from hal.client import HalClient, HalClientConfig
from hal.server import HalServerBase, HalServerConfig
from hal.server.isaac import IsaacSimHalServer
from hal.server.jetson import JetsonHalServer

# Generic HAL types
from hal.client.observation.types import NavigationCommand
from hal.client.data_structures.hardware import (
    KrabbyHardwareObservations,
    KrabbyDesiredJointPositions,
)

# Model-specific types (Parkour)
from compute.parkour.types import (
    ParkourObservation,
    ParkourModelIO,
    InferenceResponse,
)

# Mappers
from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
```

### Single Source of Truth with Editable Installs

The HAL components use a **single source of truth** approach with a clean directory structure:

- **Source files** are located directly in `hal/client/`, `hal/server/`, `hal/server/isaac/`, `hal/server/jetson/`, and `hal/tools/` directories
- **Directory structure matches import namespace**: `hal/client/` → `from hal.client import ...`
- **No redundant nesting**: Clean paths like `hal/client/client.py` instead of `hal/krabby-hal-client/hal/client/client.py`
- **Editable installs for development**: Run `make install-editable` to install packages in editable mode
  - This allows you to edit files in `hal/client/`, `hal/server/`, etc. and see changes immediately
  - No need to rebuild wheels during development
- **Wheel builds for distribution**: Run `make build-wheels` to create distributable wheels
- **Production/Docker**: Install wheels from `hal/*/dist/*.whl` (each package has its own `dist/` directory)

**Development workflow:**
```bash
# Install packages in editable mode (one-time setup)
cd hal/client && pip install -e .
cd ../server && pip install -e .
cd isaac && pip install -e .
cd ../jetson && pip install -e .
cd ../../tools && pip install -e .

# Or use make if available:
make install-editable

# Now you can edit files in hal/client/, hal/server/, etc. and changes are immediately available
# No need to rebuild or reinstall

# To build wheels for distribution/Docker
cd hal/client && python -m build
cd ../server && python -m build
# etc.
```

### Other Components
- **`compute/parkour/`**: Production inference logic (used in production container)
- **`locomotion/jetson/`**: Production runtime combining inference and HAL server for robot deployment
- **`images/locomotion/`**: Production container that runs on the robot (uses wheels)
- **`images/isaacsim/`**: IsaacSim container for simulation (uses wheels)
- **`images/testing/`**: Testing containers for x86 and ARM platforms (uses wheels)


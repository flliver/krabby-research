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
  - **`compute/parkour/`**: Production-ready inference implementation (used in production container), including **`ParkourInferenceClient`** and mappers
- **`controller/`**: On-robot scripts (e.g. gamepad / bring-up) that may use the HAL client; Jetson HAL **server** entrypoint lives under **`hal/server/jetson/`**

**Key distinction**: 
- **Game loop** = The core control logic (poll HAL → build observation → run inference → send `JointCommand`)
  - Typical stack: **`compute.parkour.inference_client.ParkourInferenceClient`** + **`hal.client.HalClient`** against a running Jetson HAL (`python -m hal.server.jetson.main` in the locomotion image)

All containers use inproc ZMQ for communication within the same process:
- **Production container** (`images/locomotion/`): Bundles **`compute/parkour/`** and **`hal/server/jetson/`** (Jetson HAL server) for the robot (Jetson/ARM). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-jetson`
- **IsaacSim container** (`images/isaacsim/`): Combines inference (`compute/parkour/`) and HAL server (`krabby-hal-server-isaac`) for simulation (x86). Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`
- **Testing containers** (`images/testing/x86/` and `images/testing/arm/`): Containers for running tests and development. Uses wheels: `krabby-hal-client`, `krabby-hal-server`, `krabby-hal-server-isaac`

These are separate from the existing `parkour/` directory which contains model-specific training and evaluation code.

## Directory Structure

```
krabby-research/
├── parkour/                          # Existing parkour model code (unchanged)
│   ├── scripts/rsl_rl/               # Training and evaluation scripts
│   ├── parkour_isaaclab/             # IsaacLab environment code
│   └── parkour_tasks/                # Task configurations (crab hexapod policy config: docs/crab-hexapod-policy-config.md)
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
│   │   │   └── hardware.py           # HardwareObservations, JointCommand
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
│           ├── hardware_to_model.py  # HardwareObservations → ParkourObservation
│           └── model_to_hardware.py   # InferenceResponse → JointCommand
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── helpers.py                    # Test helpers (create_dummy_hw_obs, etc.)
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
│
├── images/                           # OS images, Dockerfiles, and container configs
│   ├── locomotion/                   # Locomotion stack: production (Jetson)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── isaacsim/                     # Locomotion stack: IsaacSim (x86)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── testing/                      # Locomotion stack: testing
│   │   ├── x86/
│   │   │   ├── Dockerfile
│   │   │   └── requirements.txt
│   │   └── arm/
│   │       ├── Dockerfile
│   │       └── requirements.txt
│   ├── scene-reconstruction-base/    # M11 base (COLMAP source-built + Open3D)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── matcha/                       # M11 PRIMARY T1 pipeline (watertight TSDF/tetra)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── NOTES.md                  # Detailed port lessons
│   │   └── patches/                  # 4 build-time patches
│   ├── mast3r/                       # M11 T0 alternative (MASt3R-SLAM)
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── NOTES.md
│   │   ├── patches/                  # 4 build-time patches
│   │   └── verify_imports.py
│   ├── slam3r/                       # M11 alternative per grant Appendix A
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── README.md
│   └── vggt/                         # M11 alternative per grant Appendix A
│       ├── Dockerfile
│       ├── requirements.txt
│       └── README.md
│
├── real2sim/                         # M11: real-to-sim pipeline (SfM, mesh, USD prep)
│   ├── README.md                     # Layout + quick-start
│   ├── *.py                           # Mesh-conditioning + scene-build tooling
│   ├── *.sh                           # Pipeline runners (run_colmap_*, run_mast3r, etc.)
│   ├── camera_viewer/                 # T0.B5: interactive 3D camera-selection UI
│   └── rate_renders/                  # Web app for rating rendered comparisons
│
├── environments/                     # Real-to-sim output: per-scene IsaacSim assets
│   └── reconstructed/                # M11 scenes (mesh + USD + manifest)
│       ├── README.md                 # Storage convention: >100 MB → S3
│       └── <scene-name>/             # Per-scene subdir (populated as scenes ship)
│
├── experiments/                      # Reproducibility provenance per milestone
│   └── m11-scene-reconstruction/     # M11 pipeline-evaluation experiments (16 dirs)
│       ├── 001-{colmap,mast3r-slam,matcha,vggt}-patio-fisheye/
│       ├── 003-{matcha,slam3r,mast3r-slam}-firepit/
│       ├── 004-{matcha,sfm-scaling,r-knob-sweep,...}-sky-house/
│       └── DECISION-MATRIX.md
│
├── .beads/                           # Beads issue-tracking DB (M11+ uses prefix m11-*)
│   ├── issues.jsonl                  # Portable canonical export
│   └── (config.yaml, hooks/, README.md)
│
├── .env.example                      # Template for HF_TOKEN etc. (copy to .env at runtime)
│
└── scripts/                          # Deployment and utility scripts
    ├── setup-docker-gpu.sh           # NVIDIA Container Toolkit installer
    └── deploy/                        # Per-container launch scripts
        ├── run_isaac_simulation.sh
        └── run_locomotion.sh
```

> **M11 environments**: `environments/reconstructed/<scene>/` is the
> landing surface for reconstructed scene assets. Per-scene USD + mesh
> outputs commit directly to git up to 100 MB; anything larger (raw
> video, dense point clouds, vertex-colored meshes) goes to S3 (bucket
> TBD). See `environments/reconstructed/README.md` for the full
> storage convention.

## Key Points

### HAL Package Structure (Wheel-based)

The HAL packages are organized with a clean directory structure that matches the import namespace:

- **`hal/client/`**: HAL client package (package name: `krabby-hal-client`, installed via wheel)
  - `hal/client/client.py`: HalClient implementation (ZMQ black-boxed)
  - `hal/client/config.py`: HalClientConfig
  - `hal/client/__init__.py`: Re-exports `HalClient`, `HalClientConfig` for cleaner imports
  - `hal/client/observation/`: Observation types (NavigationCommand only - generic HAL type)
  - `hal/client/commands/`: Command types (empty, kept for future generic types)
  - `hal/client/data_structures/`: Hardware data structures (`HardwareObservations`, `JointCommand`)
  
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
  - `hal/server/jetson/zed_camera.py`: ZED camera integration for depth sensing
  - `hal/server/jetson/sensor_backend_jetson.py`: `JETSON_SENSOR_CATALOG`, `JetsonSensorInterface`
  - `hal/server/jetson/main.py`: Console entry (`python -m hal.server.jetson.main`)
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
    HardwareObservations,
    JointCommand,
)

# Model-specific types (Parkour)
from compute.parkour.parkour_types import (
    ParkourObservation,
    ParkourModelIO,
    InferenceResponse,
)

# Mappers
from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper
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
- **`hal/server/jetson/`**: Jetson HAL server (sensors, observations, `python -m hal.server.jetson.main`)
- **`images/locomotion/`**: Production container that runs on the robot (uses wheels)
- **`images/isaacsim/`**: IsaacSim container for simulation (uses wheels)
- **`images/testing/`**: Testing containers for x86 and ARM platforms (uses wheels)


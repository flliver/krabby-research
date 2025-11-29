# Project Folder Layout

This document describes the folder structure under `/home/shriop/projects/krabs/krabby-research/`. The current structure reflects the current implementation, and will be extended with additional components in the future.

## Overview

Currently, the project includes three new directories:
- **`hal/`**: Hardware Abstraction Layer components, organized by observation (input) and commands (output) types/interfaces, transport implementation (ZMQ), backend implementations (Isaac), and model-specific data structures
- **`compute/`**: Production inference and computation logic:
  - **`compute/parkour/`**: Production-ready inference implementation (used in production container)
- **`locomotion/`**: Production runtime that combines inference logic and HAL server for robot deployment

**Key distinction**: 
- **Game loop** = The core inference logic (poll HAL → build observation → run inference → send command)
  - Production: `locomotion/jetson/inference_runner.py::InferenceRunner.run()`

All containers use inproc ZMQ for communication within the same process:
- **Production container** (`images/locomotion/`): Combines inference (`compute/parkour/`) and HAL server (`locomotion/`) to run on the actual robot (Jetson/ARM)
- **IsaacSim container** (`images/isaacsim/`): Combines inference (`compute/parkour/`) and HAL server (`hal/isaac/`) for simulation (x86)
- **Testing containers** (`images/testing/x86/` and `images/testing/arm/`): Containers for running tests and development

These are separate from the existing `parkour/` directory which contains model-specific training and evaluation code.

## Directory Structure

```
krabby-research/
├── parkour/                          # Existing parkour model code (unchanged)
│   ├── scripts/rsl_rl/               # Training and evaluation scripts
│   ├── parkour_isaaclab/             # IsaacLab environment code
│   └── parkour_tasks/                # Task configurations
│
├── hal/                              # Hardware Abstraction Layer (current)
│   ├── __init__.py
│   ├── config.py                     # HAL configuration classes
│   │
│   ├── observation/                  # Input types and interfaces (sensor data, state)
│   │   ├── __init__.py
│   │   ├── types.py                  # NavigationCommand, ParkourObservation, ParkourModelIO
│   │   └── interfaces.py             # Observation input interfaces
│   │
│   ├── commands/                     # Output types and interfaces (actuator commands)
│   │   ├── __init__.py
│   │   ├── types.py                  # JointCommand, InferenceResponse
│   │   └── interfaces.py             # Command output interfaces
│   │
│   ├── zmq/                          # ZMQ-based HAL implementation
│   │   ├── __init__.py
│   │   ├── client.py                 # HAL client implementation
│   │   ├── server.py                 # HAL server base class
│   │   └── transport.py              # ZMQ transport utilities
│   │
│   ├── isaac/                        # IsaacSim HAL backend
│   │   ├── __init__.py
│   │   ├── hal_server.py             # IsaacSimHalServer implementation
│   │   ├── config.py                 # IsaacSim-specific config
│   │   └── main.py                   # Entry point for IsaacSim HAL server
│   │
│   ├── parkour/                      # Parkour-specific HAL components
│   │   ├── __init__.py
│   │   └── model_io.py               # ParkourModelIO and related types
│   │
│   └── tools/                        # HAL debugging and utility tools
│       ├── __init__.py
│       ├── hal_dump.py               # CLI tool to inspect HAL state
│       └── debug_logger.py           # Runtime debug logging utilities
│
├── compute/                          # Inference and computation logic (current)
│   └── parkour/                      # Parkour inference implementation (used in production)
│       ├── __init__.py
│       ├── policy_interface.py     # Parkour policy inference interface
│       └── model_loader.py           # Model loading and checkpoint management
│
├── locomotion/                       # Production runtime for Jetson (current)
│   ├── __init__.py
│   ├── hal_server.py                 # JetsonHalServer implementation (real sensors)
│   ├── inference_runner.py            # Production inference orchestration
│   ├── config.py                     # Jetson-specific config
│   ├── camera.py                     # ZED camera integration
│   └── main.py                       # Production entry point (runs inference + HAL server on Jetson)
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

- **`hal/observation/`**: Input types and interfaces (NavigationCommand, ParkourObservation, ParkourModelIO)
- **`hal/commands/`**: Output types and interfaces (JointCommand, InferenceResponse)
- **`hal/zmq/`**: ZMQ transport implementation (client, server, transport utilities)
- **`hal/isaac/`**: IsaacSim HAL backend implementation
- **`compute/parkour/`**: Production inference logic (used in production container)
- **`locomotion/`**: Production runtime combining inference and HAL server for robot deployment
- **`images/locomotion/`**: Production container that runs on the robot
- **`images/isaacsim/`**: IsaacSim container for simulation
- **`images/testing/`**: Testing containers for x86 and ARM platforms


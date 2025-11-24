# Project Folder Layout

This document describes the folder structure under `/home/shriop/projects/krabs/krabby-research/`. The current structure reflects the current implementation, and will be extended with additional components in the future.

## Overview

Currently, the project includes three new directories:
- **`HAL/`**: Hardware Abstraction Layer components, organized by telemetry (input) and commands (output) types/interfaces, transport implementation (ZMQ), backend implementations (Isaac), and model-specific data structures
- **`compute/`**: Production inference and computation logic:
  - **`compute/parkour/`**: Production-ready inference implementation (used in production container)
- **`locomotion/`**: Production runtime that combines inference logic and HAL server for robot deployment

**Key distinction**: 
- **Game loop** = The core inference logic (poll HAL → build observation → run inference → send command)
  - Production: `locomotion/jetson/inference_runner.py::InferenceRunner.run()`
  - Test simulation: `compute/testing/inference_test_runner.py::InferenceTestRunner`
- **Test scenarios** (`tests/game_loops/` if exists) = Testing tools that simulate different runtime conditions:
  - **Overloaded**: Control loop runs faster than inference (inference is the bottleneck)
  - **Underloaded**: Control loop runs slower than inference (inference waits/idle)
  - **Brief/Extended**: Different duration tests

All containers use inproc ZMQ for communication within the same process:
- **Production container** (`images/locomotion/`): Combines inference (`compute/parkour/`) and HAL server (`locomotion/`) to run on the actual robot (Jetson/ARM)
- **IsaacSim container** (`images/simulation/isaac/`): Combines inference (`compute/parkour/`) and HAL server (`HAL/Isaac/`) for simulation testing (x86)
- **Testing containers** (`images/testing/x86/` and `images/testing/arm/`): Combine inference (`compute/parkour/`) and inference test runner (`compute/testing/inference_test_runner.py`) for testing

These are separate from the existing `parkour/` directory which contains model-specific training and evaluation code.

## Directory Structure

```
krabby-research/
├── parkour/                          # Existing parkour model code (unchanged)
│   ├── scripts/rsl_rl/               # Training and evaluation scripts
│   ├── parkour_isaaclab/             # IsaacLab environment code
│   └── parkour_tasks/                # Task configurations
│
├── HAL/                              # Hardware Abstraction Layer (current)
│   ├── __init__.py
│   ├── config.py                     # HAL configuration classes
│   │
│   ├── telemetry/                    # Input types and interfaces (sensor data, state)
│   │   ├── __init__.py
│   │   ├── types.py                  # NavigationCommand, RobotState, DepthObservation
│   │   └── interfaces.py             # Telemetry input interfaces
│   │
│   ├── commands/                     # Output types and interfaces (actuator commands)
│   │   ├── __init__.py
│   │   ├── types.py                  # JointCommand, InferenceResponse
│   │   └── interfaces.py             # Command output interfaces
│   │
│   ├── ZMQ/                          # ZMQ-based HAL implementation
│   │   ├── __init__.py
│   │   ├── client.py                 # HAL client implementation
│   │   ├── server.py                 # HAL server base class
│   │   └── transport.py              # ZMQ transport utilities
│   │
│   ├── Isaac/                        # IsaacSim HAL backend
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
│   ├── testing/                      # Testing containers (for testing with game loop)
│   │   ├── x86/                      # x86 testing container (inference + game loop, inproc ZMQ)
│   │   │   ├── Dockerfile            # x86 Dockerfile for development/testing
│   │   │   └── requirements.txt
│   │   └── arm/                      # ARM testing container (inference + game loop, inproc ZMQ)
│   │       ├── Dockerfile            # ARM Dockerfile for ARM-specific testing
│   │       └── requirements.txt
│   ├── locomotion/                   # Production container (Jetson: inference + HAL server, inproc ZMQ)
│   │   ├── Dockerfile                # Jetson-compatible Dockerfile
│   │   └── requirements.txt
│   └── simulation/                   # Simulation container images
│       └── isaac/                    # IsaacSim container (inference + HAL server, inproc ZMQ)
│           ├── Dockerfile
│           └── requirements.txt
│
├── scripts/                          # Deployment and utility scripts (current)
│   ├── deploy/                       # Deployment scripts
│   │   ├── run_isaac_simulation.sh  # Launch IsaacSim HAL server
│   │   └── run_locomotion.sh        # Launch locomotion container (Jetson, inproc ZMQ)
│   └── test/                          # Integration test scripts
│       ├── test_hal_contract.py      # Test HAL ZMQ contract
│       ├── test_compute_integration.py # Test compute/inference integration
│       └── test_end_to_end.py        # End-to-end tests
│
└── tests/                            # Unit and integration tests (current)
    ├── unit/
    │   ├── test_hal_types.py         # Test HAL types and config
    │   ├── test_hal_zmq_client.py    # Test ZMQ HAL client
    │   ├── test_hal_zmq_server.py     # Test ZMQ HAL server
    │   └── test_compute_parkour_policy.py
    │
    ├── integration/
    │   ├── test_isaac_hal.py
    │   ├── test_jetson_hal.py
    │   └── test_full_stack.py
    │
    └── (test scenarios if exist)     # Test scenarios for different runtime conditions
        # Note: Game loop = inference logic (poll → build → infer → send)
        # These test scenarios simulate different control loop rates vs inference speeds
```

## Key Points

- **`HAL/telemetry/`**: Input types and interfaces (NavigationCommand, RobotState, DepthObservation)
- **`HAL/commands/`**: Output types and interfaces (JointCommand, InferenceResponse)
- **`HAL/`**: ZMQ transport and backend implementations (Isaac, parkour-specific types)
- **`compute/parkour/`**: Production inference logic (used in production container)
- **`compute/testing/inference_test_runner.py`**: Test runner that simulates the game loop (inference logic) for testing - NOT for production
- **`locomotion/`**: Production runtime combining inference and HAL server for robot deployment
- **`images/locomotion/`**: Production container that runs on the robot
- **`images/simulation/isaac/`**: IsaacSim HAL server container
- **`images/testing/x86/`**: x86 testing container with inference test runner
- **`images/testing/arm/`**: ARM testing container with inference test runner (for ARM-specific testing)


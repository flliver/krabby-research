# Runtime Architecture Diagram

This document provides a visual representation of the current containerization and communication architecture for the parkour runtime.

## System Overview

The runtime currently uses four main container patterns:

1. **IsaacSim container** (x86): IsaacSim HAL server + inference client in one process (inproc ZMQ)
2. **Testing container (x86)**: mock HAL server + inference test runner in one process (inproc ZMQ)
3. **Testing container (ARM)**: mock HAL server + inference test runner in one process (inproc ZMQ)
4. **Locomotion container** (Jetson/ARM): Jetson HAL server + inference client in one process (inproc ZMQ) - **Production**

The policy runtime (`compute/parkour/`) is shared across simulation, testing, and production. In production, `python -m hal.server.jetson.main` drives the full loop and can also start optional collector and teleop threads.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         Development Machine (x86)                              │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │              IsaacSim Container (inference + HAL server)               │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ IsaacSimHalServer                                                 │  │  │
│  │  │ - Reads simulator sensors                                         │  │  │
│  │  │ - Publishes HAL observation                                       │  │  │
│  │  │ - Receives and applies joint commands                             │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                               │                                         │  │
│  │                               │ inproc://hal_observation /              │  │
│  │                               │ inproc://hal_commands                   │  │
│  │                               ▼                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ ParkourInferenceClient + ParkourPolicyModel                      │  │  │
│  │  │ - Polls HAL observations                                          │  │  │
│  │  │ - Builds model observation tensors                                │  │  │
│  │  │ - Runs inference and sends JointCommand                           │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │           Testing Container - x86 (mock HAL + inference test)          │  │
│  │                         (TESTING / DEVELOPMENT)                         │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ MockHalServer                                                     │  │  │
│  │  │ - Publishes synthetic HardwareObservations                        │  │  │
│  │  │ - Receives JointCommand for validation/stats                      │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                               │                                         │  │
│  │                               │ inproc://hal_observation /              │  │
│  │                               │ inproc://hal_commands                   │  │
│  │                               ▼                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ Inference test runner + ParkourInferenceClient                   │  │  │
│  │  │ - Exercises the same inference loop used in production            │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           Jetson Orin (ARM64)                                 │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │             Locomotion Container (inference + HAL server)              │  │
│  │                              PRODUCTION                                 │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ JetsonHalServer                                                   │  │  │
│  │  │ - Reads real sensors (RGB-D + robot state)                        │  │  │
│  │  │ - Publishes one HAL observation stream                            │  │  │
│  │  │ - Receives and applies joint commands                             │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                               │                                         │  │
│  │                               │ inproc://hal_observation /              │  │
│  │                               │ inproc://hal_commands                   │  │
│  │                               ▼                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ ParkourInferenceClient + ParkourPolicyModel                      │  │  │
│  │  │ - Uses Go2 robot definition for current Jetson runtime            │  │  │
│  │  │ - Runs on CUDA and emits JointCommand                             │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                               │                                         │  │
│  │                               ├─ Optional: HalDataCollector thread      │  │
│  │                               └─ Optional: teleop signaling thread      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │           Testing Container - ARM (mock HAL + inference test)          │  │
│  │                         (TESTING / DEVELOPMENT)                         │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ MockHalServer                                                     │  │  │
│  │  │ - Synthetic observations at control rate                           │  │  │
│  │  │ - Command receive path for ARM validation                           │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  │                               │                                         │  │
│  │                               │ inproc://hal_observation /              │  │
│  │                               │ inproc://hal_commands                   │  │
│  │                               ▼                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐  │  │
│  │  │ Inference test runner + ParkourInferenceClient                   │  │  │
│  │  └───────────────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Communication Layers

### Layer 1: ZMQ Transport Layer

**Purpose**: Low-level transport between server/client components.

**Transports**:
- `inproc://` for same-process runtime (default in Jetson production and most tests)
- `tcp://host:port` for cross-process or cross-machine integration when explicitly configured

**Runtime endpoints**:
- Observation endpoint: `inproc://hal_observation` in Jetson production entrypoint
- Command endpoint: `inproc://hal_commands` in Jetson production entrypoint

### Layer 2: HAL Contract Layer

**Purpose**: Defines message shapes and semantics shared by all runtimes.

**Message types**:

1. **Observation** (PUB/SUB, topic prefix: `"observation"`)
   - Format: single ZMQ frame containing topic prefix + serialized `HardwareObservations`
   - Payload includes required robot state arrays and optional camera/depth data
   - Semantics: latest-only (`SNDHWM=1` on server, `RCVHWM=1` + `CONFLATE=1` on client)

2. **Joint Commands** (PUSH/PULL)
   - Format: serialized `JointCommand` blob
   - Semantics: ordered command stream with backpressure (`HWM=5`)

### Layer 3: Application Layer

**Purpose**: Runtime behavior, model inference, and hardware/simulator adaptation.

**Components**:

1. **HAL server implementations** (`hal/server/*`)
   - `JetsonHalServer` for robot hardware
   - `IsaacSimHalServer` for simulation
   - `MockHalServer` for inference tests

2. **HAL client interface** (`hal/client/client.py`)
   - Polls latest `HardwareObservations`
   - Sends `JointCommand`
   - Shares server context when using inproc

3. **Inference runtime** (`compute/parkour/inference_client.py`)
   - Maps HAL observations to model inputs
   - Runs `ParkourPolicyModel`
   - Maps inference output back to `JointCommand`
   - Loads `.pt` checkpoints (`compute/parkour/policy_interface.py`). The crab-hex policies of record live with
     their sim tasks (forward: `crab_hex_forward_task/policy/`, omni: `crab_hex_omni_task/policy/`); the omni
     depth student's ONNX pair (`policy/f19_9_student/`) is not consumed by `compute/parkour`.

## Container Communication Patterns

### Pattern 1: IsaacSim Container (Simulation Testing)

```
IsaacSim Container (x86)
    ├─ IsaacSimHalServer
    │    │
    │    │ inproc://hal_observation / inproc://hal_commands
    │    ▼
    └─ ParkourInferenceClient + ParkourPolicyModel
```

**Use case**: Simulation validation with the same HAL contract used in production.

### Pattern 2: Testing Container - x86 (Inference Test Loop)

```
Testing Container - x86
    ├─ MockHalServer
    │    │
    │    │ inproc://hal_observation / inproc://hal_commands
    │    ▼
    └─ Inference test runner + ParkourInferenceClient
```

**Use case**: Fast inference validation without IsaacSim or robot hardware.

### Pattern 3: Testing Container - ARM (ARM-Specific Validation)

```
Testing Container - ARM
    ├─ MockHalServer
    │    │
    │    │ inproc://hal_observation / inproc://hal_commands
    │    ▼
    └─ Inference test runner + ParkourInferenceClient
```

**Use case**: ARM-specific runtime checks with synthetic observations.

### Pattern 4: Locomotion Container (Production)

```
Locomotion Container (Jetson)
    ├─ JetsonHalServer
    │    │
    │    │ inproc://hal_observation / inproc://hal_commands
    │    ▼
    └─ ParkourInferenceClient + ParkourPolicyModel
```

**Use case**: Robot deployment with real sensors and actuators.

## Data Flow

### Forward Path (Observation -> Action)

**For production (Jetson) and simulation (IsaacSim):**
```
1. HAL server
   -> publish HardwareObservations on observation channel

2. ParkourInferenceClient
   -> poll latest HardwareObservations
   -> map to model observation tensors

3. ParkourPolicyModel
   -> run forward pass
   -> produce action tensor

4. ParkourInferenceClient
   -> map action tensor to JointCommand
   -> send JointCommand to command channel

5. HAL server
   -> receive JointCommand
   -> apply to simulator or hardware actuators
```

**For testing containers (x86 and ARM):**
```
1. MockHalServer
   -> publish synthetic HardwareObservations

2. Inference test runner / ParkourInferenceClient
   -> poll, infer, and send JointCommand

3. MockHalServer
   -> receive JointCommand for validation/stats
```

### Control Loop Timing

```
┌──────────────────────────────────────────────────────────────────┐
│  Main loop tick (10 ms target, 100 Hz)                          │
│                                                                  │
│  t=0ms    HAL publishes newest observation                       │
│  t=1-3ms  inference client maps + runs model                     │
│  t=3-5ms  command sent and consumed by HAL                       │
│  t=5-10ms scheduler slack (or overrun if system is saturated)    │
└──────────────────────────────────────────────────────────────────┘
```

Actual timings depend on sensor IO, model latency, and device load; the runtime logs lag when loop duration exceeds target.

## High-Watermark and Backpressure

- **Observation HWM = 1**: stale observations are dropped in favor of the latest frame
- **Client CONFLATE = 1**: if multiple observations arrive before polling, only the newest is retained
- **Command HWM = 5**: commands use bounded queueing to preserve ordering without unbounded growth
- **Runtime goal**: keep the command path near real-time while preventing memory/queue buildup

## Security and Isolation

- **Container isolation**: simulation/testing/production images remain separately deployable
- **Process-local default**: production loop uses inproc endpoints within one process
- **Message-only boundaries**: HAL server/client communicate strictly via serialized messages
- **Optional network mode**: TCP endpoints are available for explicit cross-process integration, not default production path


# Runtime Architecture Diagram

This document provides a visual representation of the containerization and communication architecture for the parkour policy runtime system.

## System Overview

The system implements a containerized architecture with four distinct container types:

1. **IsaacSim container** (x86): Combined inference logic + IsaacSim HAL server for simulation testing (inproc ZMQ)
2. **Testing container (x86)**: Combined inference logic + game loop test script for x86 testing (inproc ZMQ)
3. **Testing container (ARM)**: Combined inference logic + game loop test script for ARM-specific testing (inproc ZMQ)
4. **Locomotion container** (Jetson/ARM): Combined inference logic + HAL server for robot deployment (inproc ZMQ) - **Production**

All containers use inproc ZMQ for communication within the same process. The parkour policy inference logic (`compute/parkour/`) is shared across all containers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Development Machine (x86)                        │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              IsaacSim Container (inference + HAL server)          │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  IsaacSimHalServer                                          │ │  │
│  │  │  - Publishes: camera frames, robot state                    │ │  │
│  │  │  - Receives: joint commands                                 │ │  │
│  │  │  - Applies commands to IsaacSim environment                 │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  │                          │                                       │  │
│  │                          │ ZMQ inproc:// (same process)          │  │
│  │                          ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  InferenceRunner + ParkourPolicyModel                       │ │  │
│  │  │  - Polls HAL server for sensor data                         │ │  │
│  │  │  - Builds observation tensor                                │ │  │
│  │  │  - Runs policy inference                                    │ │  │
│  │  │  - Sends joint commands to HAL server                       │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         Testing Container - x86 (inference + game loop)           │  │
│  │                    (TESTING/DEVELOPMENT ONLY)                      │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  Game Loop Test Script                                      │ │  │
│  │  │  - Simulates sensor messages (test data)                    │ │  │
│  │  │  - Publishes: simulated camera frames, robot state           │ │  │
│  │  │  - Receives: joint commands                                 │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  │                          │                                       │  │
│  │                          │ ZMQ inproc:// (same process)          │  │
│  │                          ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  HalClient + ParkourPolicyModel                             │ │  │
│  │  │  - Receives simulated sensor messages                       │ │  │
│  │  │  - Builds observation tensor                                │ │  │
│  │  │  - Runs policy inference                                    │ │  │
│  │  │  - Sends joint commands to game loop                        │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    Jetson Orin (ARM64)                                    │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         Locomotion Container (inference + HAL server)             │  │
│  │                         PRODUCTION                                │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  JetsonHalServer                                            │ │  │
│  │  │  - ZED Mini Camera → depth features                         │ │  │
│  │  │  - IMU/Encoders → robot state                               │ │  │
│  │  │  - Publishes: camera frames, robot state                    │ │  │
│  │  │  - Receives: joint commands                                 │ │  │
│  │  │  - Forwards to: motor controllers                            │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  │                          │                                       │  │
│  │                          │ ZMQ inproc:// (same process)          │  │
│  │                          ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  InferenceRunner                                            │ │  │
│  │  │  - Polls HAL server for sensor data                         │ │  │
│  │  │  - Builds observation tensor                                │ │  │
│  │  │  - Runs policy inference                                    │ │  │
│  │  │  - Sends joint commands to HAL server                       │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  │                          │                                       │  │
│  │                          ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  ParkourPolicyModel (compute/parkour/)                     │ │  │
│  │  │  - Loads checkpoint                                        │ │  │
│  │  │  - Runs inference (100+ Hz)                                 │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │         Testing Container - ARM (inference + game loop)           │  │
│  │                    (TESTING/DEVELOPMENT ONLY)                      │  │
│  │                                                                   │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  Game Loop Test Script                                      │ │  │
│  │  │  - Simulates sensor messages (test data)                    │ │  │
│  │  │  - Publishes: simulated camera frames, robot state           │ │  │
│  │  │  - Receives: joint commands                                 │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  │                          │                                       │  │
│  │                          │ ZMQ inproc:// (same process)          │  │
│  │                          ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────┐ │  │
│  │  │  HalClient + ParkourPolicyModel                             │ │  │
│  │  │  - Receives simulated sensor messages                       │ │  │
│  │  │  - Builds observation tensor                                │ │  │
│  │  │  - Runs policy inference                                    │ │  │
│  │  │  - Sends joint commands to game loop                        │ │  │
│  │  └────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Communication Layers

### Layer 1: ZMQ Transport Layer

**Purpose**: Low-level message transport between containers/processes

**Transports**:
- `inproc://` - In-process communication (for unit tests, same-container deployment)
- `tcp://host:port` - Network communication (for cross-container, cross-machine)

**Endpoints** (configurable via `HAL_BASE_PORT`, default 6000):
- Camera: `tcp://host:6000` (PUB/SUB)
- State: `tcp://host:6001` (PUB/SUB)
- Commands: `tcp://host:6002` (REQ/REP)

### Layer 2: HAL Contract Layer

**Purpose**: Defines message formats and semantics

**Message Types**:

1. **Camera Observation** (PUB/SUB, topic: `"camera"`)
   - Format: Topic-prefixed multipart message
   - Payload: `float32[N]` array (depth features)
   - Rate: 30-60 Hz

2. **State Observation** (PUB/SUB, topic: `"state"`)
   - Format: Topic-prefixed multipart message
   - Payload: `float32[M]` array containing:
     - Base position (3), quaternion (4)
     - Base linear velocity (3), angular velocity (3)
     - Joint positions (ACTION_DIM), joint velocities (ACTION_DIM)
   - Rate: 100+ Hz

3. **Joint Commands** (REQ/REP)
   - Request: `float32[ACTION_DIM]` (desired joint positions)
   - Reply: `b"ok"` (acknowledgement)
   - Rate: 100+ Hz

### Layer 3: Application Layer

**Purpose**: Business logic and policy inference

**Components**:

1. **HalClient** (`locomotion/interfaces/hal/client.py`)
   - Subscribes to camera and state topics
   - Maintains latest-only buffers
   - Sends joint commands via REQ/REP
   - Builds model input dataclass

2. **Policy Model** (`locomotion/runtime/policy_wrapper.py`)
   - Loads trained checkpoint
   - Converts model input → observation tensor
   - Runs policy inference
   - Returns `JointCommand`

3. **Game Loop** (`locomotion/runtime/game_loop.py`)
   - Runs at 100+ Hz
   - Polls HAL for latest data
   - Calls policy inference
   - Sends commands back to HAL

## Container Communication Patterns

### Pattern 1: IsaacSim Container (Simulation Testing)

```
IsaacSim Container (x86)
    ├─ IsaacSimHalServer
    │   │
    │   │ ZMQ inproc:// (same process)
    │   │
    │   ▼
    └─ InferenceRunner + ParkourPolicyModel
```

**Use Case**: Testing inference logic with IsaacSim simulation - inference and HAL server in same container

### Pattern 2: Testing Container - x86 (Game Loop Testing)

```
Testing Container - x86
    ├─ Game Loop Test Script
    │   │
    │   │ ZMQ inproc:// (same process)
    │   │
    │   ▼
    └─ HalClient + ParkourPolicyModel
```

**Use Case**: Testing inference logic with simulated sensor messages (game loop) - inference and game loop in same container

### Pattern 3: Testing Container - ARM (ARM-Specific Testing)

```
Testing Container - ARM (Jetson)
    ├─ Game Loop Test Script
    │   │
    │   │ ZMQ inproc:// (same process)
    │   │
    │   ▼
    └─ HalClient + ParkourPolicyModel
```

**Use Case**: Testing ARM-specific inference behavior with simulated sensor messages - inference and game loop in same container

### Pattern 4: Locomotion Container (Production)

```
Locomotion Container (Jetson Orin)
    ├─ JetsonHalServer (real sensors)
    │   │
    │   │ ZMQ inproc:// (same process)
    │   │
    │   ▼
    └─ InferenceRunner + ParkourPolicyModel
```

**Use Case**: Production deployment on robot - inference and HAL server in same container

## Data Flow

### Forward Path (Observation → Action)

**For Production Container (Locomotion) and IsaacSim Container:**
```
1. HAL Server (IsaacSimHalServer or JetsonHalServer)
   └─> Publishes camera frame (ZMQ PUB inproc, topic: "camera")
   └─> Publishes robot state (ZMQ PUB inproc, topic: "state")

2. InferenceRunner
   └─> Receives camera frame (ZMQ SUB inproc)
   └─> Receives robot state (ZMQ SUB inproc)
   └─> Builds observation tensor

3. ParkourPolicyModel
   └─> Runs policy forward pass
   └─> Returns JointCommand

4. InferenceRunner
   └─> Sends JointCommand to HAL (ZMQ REQ inproc)

5. HAL Server
   └─> Receives command (ZMQ REP inproc)
   └─> Applies to simulator/hardware
```

**For Testing Containers (x86 and ARM):**
```
1. Game Loop Test Script
   └─> Publishes simulated camera frame (ZMQ PUB inproc, topic: "camera")
   └─> Publishes simulated robot state (ZMQ PUB inproc, topic: "state")

2. HalClient
   └─> Receives camera frame (ZMQ SUB inproc)
   └─> Receives robot state (ZMQ SUB inproc)
   └─> Builds model input dataclass

3. ParkourPolicyModel
   └─> Converts model input → observation tensor
   └─> Runs policy forward pass
   └─> Returns JointCommand

4. HalClient
   └─> Sends JointCommand to game loop (ZMQ REQ inproc)

5. Game Loop Test Script
   └─> Receives command (ZMQ REP inproc)
   └─> Logs/validates command
```

### Control Loop Timing

```
┌─────────────────────────────────────────────────────────┐
│  Game Loop Tick (10ms target, 100 Hz)                   │
│                                                           │
│  t=0ms:   Poll HAL for latest camera/state              │
│  t=1ms:   Assemble observation tensor                    │
│  t=2ms:   Policy inference (< 15ms target)             │
│  t=3ms:   Send joint command to HAL                      │
│  t=4-10ms: Wait for next tick                            │
└─────────────────────────────────────────────────────────┘
```

## High-Watermark and Backpressure

- **HWM = 1**: Only latest message is kept, older messages are dropped
- **Latest-only semantics**: Client always gets most recent camera/state frame
- **No queue overflow**: ZMQ automatically drops old messages when HWM is reached
- **Control rate**: 100+ Hz ensures commands are consumed faster than produced

## Security and Isolation

- **Container isolation**: Each container type runs independently
- **Process isolation**: `inproc://` transport for same-container communication (all containers use inproc)
- **No shared state**: All communication is message-based via ZMQ
- **Architecture isolation**: Testing containers are separate from production containers


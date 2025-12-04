# HAL (Hardware Abstraction Layer) Guide

This guide explains how to work with the HAL for publishing observation, subscribing to sensor data, and sending commands.

## Overview

The HAL uses ZMQ (ZeroMQ) for communication with two distinct channels:

1. **Observation** (PUB/SUB) - Topic: `"observation"` - HAL Server → Policy Wrapper (100+ Hz)
   - Hardware observation format containing raw sensor data
   - `KrabbyHardwareObservations` object with joint positions, camera images, depth map, etc.
2. **Joint Commands** (PUSH/PULL) - No topic - Policy Wrapper → HAL Server (100+ Hz)

All channels support both `inproc://` (same process) and `tcp://` (network) transports.

**HWM (High-Watermark)**: HWM=1 means only the latest message is kept in buffers. Old messages are automatically dropped. This ensures real-time control always uses fresh data, prevents queue buildup, and simplifies client code (no need to drain queues).

## Communication Channels

### Observation (PUB/SUB)

- **Topic**: `"observation"`
- **Message**: Hardware observation data as `KrabbyHardwareObservations`
- **Format**: Topic-prefixed multipart message `[topic_string, schema_version_string, ...hw_obs_parts]`
  - Part 0: Topic string `"observation"` (UTF-8 encoded)
  - Part 1: Schema version string (e.g., `"1.0"`, UTF-8 encoded)
  - Parts 2-7: Hardware observation parts (6 parts total)
    - Part 2: Metadata JSON (includes timestamp_ns, shapes, dtypes)
    - Part 3: joint_positions bytes (18 DOF, float32)
    - Part 4: rgb_camera_1 bytes (H×W×3, uint8)
    - Part 5: rgb_camera_2 bytes (H×W×3, uint8)
    - Part 6: depth_map bytes (H×W, float32)
    - Part 7: confidence_map bytes (H×W, float32)
- **Semantics**: Latest-only (HWM=1)
- **Content**: Raw hardware sensor data:
  - Joint positions (18 DOF)
  - RGB camera images (2 cameras)
  - Depth map
  - Confidence map
  - Timestamp (included in metadata)

### Joint Commands (PUSH/PULL)

- **Message**: Joint positions as `float32[18]` array (18 DOF for Krabby robot)
- **Semantics**: PUSH/PULL pattern ensures FIFO ordering with backpressure (HWM=5)

## Hardware Data Structures

### KrabbyHardwareObservations

Raw hardware sensor data from the Krabby robot:
- **joint_positions**: `float32[18]` - Joint positions (18 DOF)
- **rgb_camera_1**: `uint8[H, W, 3]` - First RGB camera image
- **rgb_camera_2**: `uint8[H, W, 3]` - Second RGB camera image
- **depth_map**: `float32[H, W]` - Depth map
- **confidence_map**: `float32[H, W]` - Confidence map
- **timestamp_ns**: Integer (nanoseconds)

This is the format sent by the HAL server and received by the HAL client via `poll()`.

### KrabbyDesiredJointPositions

Desired joint positions for robot control:
- **joint_positions**: `float32[18]` - Target joint positions (18 DOF)
- **timestamp_ns**: Integer (nanoseconds)

This is the format sent to the HAL server via `put_joint_command()`.

## Model-Specific Types

Model-specific types (ParkourObservation, ParkourModelIO, InferenceResponse, etc.) are located in `compute.parkour.parkour_types`. These are used by the policy inference code, not by the HAL client/server directly.

### NavigationCommand

Generic navigation command (in `hal.client.observation.types`):
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **vx**: Float (m/s) - Forward velocity
- **vy**: Float (m/s) - Lateral velocity
- **yaw_rate**: Float (rad/s) - Angular velocity

This is a generic HAL type, not specific to any policy model.

## Coordinate Frame Conventions

All coordinate frames follow the **ROS (REP-103) convention**:

### World Frame (`/world` or `/map`)
- **Origin**: Fixed reference point in the environment
- **X-axis**: Forward (typically East or robot's initial forward direction)
- **Y-axis**: Left (typically North or robot's initial left direction)
- **Z-axis**: Up (opposite to gravity)
- **Units**: Meters

### Robot Base Frame (`/base` or `/base_link`)
- **Origin**: Center of robot base (typically at ground contact point or COM)
- **X-axis**: Forward (robot's forward direction)
- **Y-axis**: Left (robot's left direction)
- **Z-axis**: Up (opposite to gravity, normal to ground)
- **Units**: Meters for position, radians for orientation
- **Quaternion**: (x, y, z, w) format, ROS convention

### Camera Frame (`/camera` or `/camera_link`)
- **Origin**: Camera optical center
- **X-axis**: Right (image right)
- **Y-axis**: Down (image down)
- **Z-axis**: Forward (optical axis, into scene)
- **Units**: Meters
- **Note**: This is the standard camera frame convention (OpenCV/ROS)

### Joint Frame Conventions
- **Joint positions**: Radians, measured from zero position
- **Joint velocities**: Rad/s
- **Joint order**: Must match the robot's joint ordering (typically defined in URDF)
- **For Krabby robot**: 18 DOF (hardware joint positions)
- **Model action dimension**: May differ from hardware (e.g., 12 DOF for some models)

### Navigation Command Frame
- **vx**: Forward velocity in robot base frame (m/s, positive = forward)
- **vy**: Lateral velocity in robot base frame (m/s, positive = left)
- **yaw_rate**: Angular velocity around robot base Z-axis (rad/s, positive = counter-clockwise when viewed from above)

### Observation Data Frame Conventions
- **Base position**: World frame (x, y, z) in meters
- **Base orientation**: World frame quaternion (x, y, z, w)
- **Base linear velocity**: Robot base frame (x, y, z) in m/s
- **Base angular velocity**: Robot base frame (x, y, z) in rad/s
- **Depth features**: Camera frame depth measurements, converted to features matching training format

### Important Notes
- All transformations between frames must be consistent with ROS conventions
- Quaternions use (x, y, z, w) format (not (w, x, y, z))
- Right-handed coordinate systems throughout
- When in doubt, refer to the robot's URDF for joint ordering and frame definitions

## Runtime Type Validation

The HAL implementation includes runtime type validation for all message payloads:

### Observation Validation
- **Type**: Must be `KrabbyHardwareObservations`
- **joint_positions**: Shape `(18,)`, dtype `float32`
- **rgb_camera_1/2**: Shape `(H, W, 3)`, dtype `uint8`
- **depth_map**: Shape `(H, W)`, dtype `float32`
- **confidence_map**: Shape `(H, W)`, dtype `float32`
- **timestamp_ns**: Non-negative integer

### Command Validation
- **Type**: Must be `KrabbyDesiredJointPositions`
- **joint_positions**: Shape `(18,)`, dtype `float32`
- **timestamp_ns**: Non-negative integer

### Error Handling
- Invalid messages are logged and rejected
- Error responses are sent back to clients when validation fails
- Validation errors include detailed information about what failed

## Interface Actions

The HAL client interface supports:

- **Poll for Observation** - `poll()` returns `KrabbyHardwareObservations` if new data is available, `None` otherwise (100+ Hz)
- **Send Joint Command** - `put_joint_command()` sends `KrabbyDesiredJointPositions` to actuators

**Observation Format**: The observation is a `KrabbyHardwareObservations` object containing raw hardware sensor data (joint positions, camera images, depth map, etc.).

**Mapping to Model Format**: The policy inference code uses mappers (`compute.parkour.mappers.hardware_to_model`) to convert `KrabbyHardwareObservations` to model-specific observation formats (e.g., `ParkourObservation`).

### Constraints

- All floats must be `float32` dtype
- Quaternion must be normalized
- Hardware joint arrays are always 18 DOF
- Model action dimension may differ (mappers handle conversion)
- Depth features array must match N (model-specific)
- Timestamps must be monotonically increasing
- Schema versions must be compatible across all components

## Command Interface

The HAL client sends joint commands via `put_joint_command()`, which accepts a `KrabbyDesiredJointPositions` object containing 18-DOF joint positions.

**PUSH/PULL Pattern**: Commands use PUSH/PULL to ensure FIFO ordering with backpressure (HWM=5).

### Constraints

- Joint positions array must be `float32` dtype
- Array shape must be `(18,)` for Krabby robot
- Joint positions typically in range [-π, π] radians
- Commands must be generated within < 10ms from observation timestamp for 100 Hz control

### Validation

**Required checks**:
- Array shape matches (18,)
- Array is float32 dtype
- Timestamp is recent (not stale)

**Optional checks** (typically in actuator layer):
- Joint positions within limits
- Velocity limits (change from previous command)

**Note**: Model-specific types like `InferenceResponse` are in `compute.parkour.parkour_types`. The policy inference code uses mappers (`compute.parkour.mappers.model_to_hardware`) to convert model outputs to `KrabbyDesiredJointPositions` before sending via HAL.

## Endpoints

**TCP Endpoints** (explicit configuration required):
- Observation: Configured via `observation_bind` (e.g., `tcp://*:6001`)
- Commands: Configured via `command_bind` (e.g., `tcp://*:6002`)

**Inproc Endpoints** (same process):
- Observation: Configured via `observation_bind` (e.g., `inproc://hal_observation`)
- Commands: Configured via `command_bind` (e.g., `inproc://hal_commands`)

## HAL Server Workflow

1. Create PUB socket for observation, PULL socket for commands
2. Bind to endpoints (inproc or TCP)
3. Set HWM=1 for latest-only semantics on PUB socket
4. Main loop:
   - Build `KrabbyHardwareObservations` from hardware sensors (100+ Hz)
   - Publish observation: Send as multipart `[topic, schema_version, ...hw_obs_parts]` (8 parts total)
   - Receive command messages (non-blocking): Deserialize to `float32[18]` array, apply to actuators

## HAL Client Workflow

1. Create SUB socket for observation, PUSH socket for commands
2. Connect to endpoints
3. Subscribe to observation topic: `"observation"`
4. Set HWM=1 for latest-only semantics on SUB sockets
5. Main loop:
   - Poll for observation messages (non-blocking with timeout)
   - `poll()` returns `KrabbyHardwareObservations` if new data is available, `None` otherwise
   - If observation received, map it to model format and use for inference
   - Map inference output to `KrabbyDesiredJointPositions`
   - Send joint command via `put_joint_command()`
   - Send command (blocking send for backpressure)

**Example:**
```python
from hal.client import HalClient, HalClientConfig
from hal.client.data_structures.hardware import KrabbyDesiredJointPositions
from hal.client.observation.types import NavigationCommand
from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
from compute.parkour.parkour_types import ParkourModelIO

# Initialize client
config = HalClientConfig(
    observation_endpoint="inproc://hal_observation",
    command_endpoint="inproc://hal_commands",
)
client = HalClient(config)
client.initialize()

# Initialize mappers
hw_to_model_mapper = KrabbyHWObservationsToParkourMapper()
model_to_hw_mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=12)

# Main control loop
nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)
while running:
    # Poll for new hardware observation
    hw_obs = client.poll(timeout_ms=10)
    
    if hw_obs is None:
        # No new data available, skip this iteration
        continue
    
    # Map hardware observation to model format
    parkour_obs = hw_to_model_mapper.map(hw_obs)
    
    # Build model IO
    model_io = ParkourModelIO(
        timestamp_ns=parkour_obs.timestamp_ns,
        schema_version=parkour_obs.schema_version,
        nav_cmd=nav_cmd,
        observation=parkour_obs,
    )
    
    # Run inference
    inference_result = model.inference(model_io)
    
    # Map inference output to hardware joint positions
    joint_positions = model_to_hw_mapper.map(inference_result)
    
    # Send command
    client.put_joint_command(joint_positions)
```

## Latest-Only Semantics

The observation PUB/SUB channel uses HWM=1 (high-watermark=1):
- Only the latest message is kept in buffers
- Old messages are automatically dropped
- Subscribers always receive the most recent observation
- Prevents queue buildup and ensures real-time control uses fresh data
- When `poll()` is called, it returns the latest observation if available, or `None` if no new data

## Synchronization

- **Topic filtering**: Subscribers subscribe to the `"observation"` topic
- **Message ordering**: PUB/SUB has no guaranteed ordering; PUSH/PULL guarantees FIFO ordering
- **Timestamp synchronization**: Observation messages include timestamps in `KrabbyHardwareObservations` metadata
- **Navigation command**: Managed by application code (e.g., `InferenceRunner`), not by HAL client

## Error Handling

- **Connection errors**: Handle ZMQ connection failures with retry logic
- **Timeouts**: Use non-blocking polling with appropriate timeouts (e.g., 10ms for 100 Hz control)
- **Invalid messages**: Validate array shapes and dtypes, reject malformed messages
- **Stale data**: Check message timestamps, reject data older than threshold (e.g., 10ms)

## Best Practices

- **Always set HWM=1**: Ensures latest-only semantics and prevents memory buildup
- **Use non-blocking polling**: Avoid blocking operations in control loops
- **Validate message sizes**: Always check array shapes match expected dimensions
- **Use inproc for same process**: Zero-copy, faster, simpler deployment
- **Use TCP for cross-process/network**: Works across containers and machines

## See Also

- `POLICY_WRAPPER.md` - How policy wrapper uses HAL
- `LOCOMOTION_RUNTIME.md` - Production runtime implementation

---

# HAL Debugging Guide

This section explains how to use the debugging tools for the Hardware Abstraction Layer (HAL).

## hal_dump Tool

The `hal_dump` tool inspects the current state of a HAL server, showing observation data and command endpoint status.

### Basic Usage

**New Format (Unified Observation):**
```bash
python -m hal.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002
```

### Verbose Mode

Show detailed breakdown of observation/state components:

```bash
python -m hal.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002 \
    --verbose
```

### In-Process Endpoints

For debugging in-process communication:

```bash
python -m hal.tools.hal_dump \
    --observation_endpoint inproc://hal_observation \
    --command_endpoint inproc://hal_command
```

### Custom Action Dimension

If your robot has a different number of joints:

```bash
python -m hal.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002 \
    --action_dim 18 \
    --verbose
```

### Output Format

The tool displays:
- **Observation/Observation**: Shape, dtype, statistics (min/max/mean)
- **Detailed Breakdown** (verbose mode):
  - Proprioceptive features (root angular velocity, IMU, joint positions/velocities)
  - Scan features (depth/height measurements)
  - Privileged explicit features (base linear velocity)
  - Privileged latent features (body mass, COM, friction)
  - History buffer
- **Command Endpoint**: Connection status and test response

### Example Output

```
================================================================================
HAL Server State Dump
================================================================================
Timestamp: 2024-01-15 10:30:45

📊 Observation:
  Topic: observation
  Schema Version: 1.0
  Timestamp: 1705315845000000000 ns (1705315845.000000 s)
  Hardware Observation:
    Joint Positions: shape=(18,), dtype=float32
    RGB Camera 1: shape=(480, 640, 3), dtype=uint8
    RGB Camera 2: shape=(480, 640, 3), dtype=uint8
    Depth Map: shape=(480, 640), dtype=float32
    Confidence Map: shape=(480, 640), dtype=float32

  Mapped Model Observation (verbose mode):
    Shape: (753,)
    Dtype: float32
    Stats: min=-1.234, max=2.456, mean=0.123
    Observation Breakdown:
      Total Dimension: 753
      Proprioceptive (53):
        Root angular velocity (body frame): [0.0123, -0.0045, 0.0089]
        IMU (roll, pitch): [0.0012, -0.0023]
        Delta yaw: 0.0456
        ...
      Scan Features (132):
        Min: -1.000, Max: 1.000, Mean: 0.123
        ...

⚙️  Command Endpoint:
  Status: ✅ Connected
  Response: ok
  Test Command Shape: (18,)
  Note: Commands are PUSH/PULL, no history available

================================================================================
```

## Debug Logging

The HAL supports runtime debug logging that can be enabled/disabled without restarting the system.

### Enabling Debug Logging

**In Code:**
```python
from hal.client import HalClient, HalClientConfig
from hal.server import HalServerBase, HalServerConfig

# Enable debug logging on client
hal_client = HalClient(config)
hal_client.initialize()
hal_client.set_debug(True)  # Enable debug logging

# Enable debug logging on server
hal_server = HalServerBase(config)
hal_server.initialize()
hal_server.set_debug(True)  # Enable debug logging
```

**Runtime Toggle:**
```python
# Toggle debug logging at runtime
hal_client.set_debug(not hal_client.is_debug_enabled())
hal_server.set_debug(not hal_server.is_debug_enabled())
```

### Debug Log Output

When enabled, debug logging shows:

**Client (Receiving):**
```
[ZMQ RECV] observation: topic=observation, schema=1.0, num_parts=8
[ZMQ RECV] observation: KrabbyHardwareObservations created successfully
[ZMQ SEND] command: payload_size=72 bytes, joint_positions_shape=(18,), dtype=float32
```

**Server (Sending):**
```
[ZMQ SEND] observation: topic=observation, schema=1.0, num_parts=8, timestamp_ns=...
[ZMQ RECV] command: shape=(18,), dtype=float32, min=-0.5, max=0.5
```

### Debug Log Format

- **Timestamps**: All logs include timestamps from the logging system
- **Structured Format**: Shows shape, dtype, and statistics for arrays
- **Message Type**: Prefixes indicate send/receive and message type
- **Validation**: Errors are logged if data validation fails (NaN/Inf, wrong shape/dtype)

### Performance Considerations

- Debug logging adds overhead (string formatting, logging calls)
- Disable in production for maximum performance
- Enable only when debugging specific issues
- Runtime toggling allows enabling/disabling without restart

### Example: Debugging a Connection Issue

```python
# Enable debug logging
hal_client.set_debug(True)
hal_server.set_debug(True)

# Run your code - you'll see detailed logs
# ...

# Disable when done
hal_client.set_debug(False)
hal_server.set_debug(False)
```

### Example: Checking Data Flow

```python
# Enable debug on both ends
hal_client.set_debug(True)
hal_server.set_debug(True)

# Run a few cycles
for _ in range(10):
    hal_server.set_observation()  # Publish observation
    observation = hal_client.poll(timeout_ms=100)
    if observation is not None:
        print(f"Received observation: shape={observation.shape}")
    # Check logs to see if data is flowing

# Disable when done
hal_client.set_debug(False)
hal_server.set_debug(False)
```

## Troubleshooting

### No Data Available

If `hal_dump` shows "No data available":
1. Check that the HAL server is running and publishing
2. Verify endpoint addresses match (tcp:// vs inproc://)
3. Check firewall/network settings for TCP endpoints
4. Ensure server has published at least one message

### Command Endpoint Timeout

If command endpoint shows timeout:
1. Verify server is running and listening on command endpoint
2. Check endpoint address matches
3. Ensure no other client is holding the REP socket
4. Check for network issues (TCP endpoints)

### Debug Logs Not Appearing

If debug logs don't appear:
1. Verify `set_debug(True)` was called
2. Check logging level is INFO or DEBUG
3. Ensure logging is configured (basicConfig or similar)
4. Check that messages are actually being sent/received

### Wrong Observation Format

If observation deserialization fails:
1. Check that server is sending `KrabbyHardwareObservations` format
2. Verify message has 8 parts (topic, schema, 6 hw_obs parts)
3. Check metadata JSON is valid
4. Verify array shapes match expected (joint_positions: 18, cameras: H×W×3, depth/confidence: H×W)

## Debugging Best Practices

1. **Use hal_dump for Quick Inspection**: Quick way to check if server is running and data is flowing
2. **Enable Debug Logging Selectively**: Only enable when debugging specific issues
3. **Check Verbose Output**: Use `--verbose` to understand data structure
4. **Monitor Performance**: Disable debug logging in production
5. **Use In-Process for Testing**: Use `inproc://` endpoints for faster local testing

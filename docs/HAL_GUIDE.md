# HAL (Hardware Abstraction Layer) Guide

This guide explains how to work with the HAL for publishing telemetry, subscribing to sensor data, and sending commands.

## Overview

The HAL uses ZMQ (ZeroMQ) for communication with three distinct channels:

1. **Camera Telemetry** (PUB/SUB) - Topic: `"camera"` - HAL Server → Policy Wrapper (30-60 Hz)
2. **State Telemetry** (PUB/SUB) - Topic: `"state"` - HAL Server → Policy Wrapper (100+ Hz)
3. **Joint Commands** (REQ/REP) - No topic - Policy Wrapper → HAL Server (100+ Hz)

All channels support both `inproc://` (same process) and `tcp://` (network) transports.

**HWM (High-Watermark)**: HWM=1 means only the latest message is kept in buffers. Old messages are automatically dropped. This ensures real-time control always uses fresh data, prevents queue buildup, and simplifies client code (no need to drain queues).

## Communication Channels

### Camera Telemetry (PUB/SUB)

- **Topic**: `"camera"`
- **Message**: Depth features as `float32[N]` array
- **Format**: Topic-prefixed multipart message `[topic_string, schema_version_string, binary_payload]`
  - Part 0: Topic string `"camera"` (UTF-8 encoded)
  - Part 1: Schema version string (e.g., `"1.0"`, UTF-8 encoded)
  - Part 2: Binary payload (`float32[N]` array serialized as binary)
- **Semantics**: Latest-only (HWM=1)

### State Telemetry (PUB/SUB)

- **Topic**: `"state"`
- **Message**: Robot state as `float32[M]` array containing base position (3), quaternion (4), base linear velocity (3), base angular velocity (3), joint positions (ACTION_DIM), joint velocities (ACTION_DIM)
- **Format**: Topic-prefixed multipart message `[topic_string, schema_version_string, binary_payload]`
  - Part 0: Topic string `"state"` (UTF-8 encoded)
  - Part 1: Schema version string (e.g., `"1.0"`, UTF-8 encoded)
  - Part 2: Binary payload (`float32[M]` array serialized as binary)
- **Semantics**: Latest-only (HWM=1)

### Joint Commands (REQ/REP)

- **Request**: Joint positions as `float32[ACTION_DIM]` array
- **Response**: Acknowledgement string (`"ok"` or error message)
- **Semantics**: Request-response pattern ensures ordering and acknowledgement

## Telemetry Models

Input data models for the Hardware Abstraction Layer (HAL). These represent sensor data and robot state flowing from hardware/simulation to the policy wrapper.

### Overview

Telemetry models capture robot state and sensor observations. They use nested structures for code organization but are flattened to a flat tensor for policy inference.

**Common Reusable Structures**:
- **Position3D**: `{x, y, z}` (float, meters)
- **Quaternion**: `{x, y, z, w}` (float)
- **Vector3D**: `{x, y, z}` (float)

### Model Structures

#### NavigationCommand
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **vx**: Float (m/s) - Forward velocity
- **vy**: Float (m/s) - Lateral velocity
- **yaw_rate**: Float (rad/s) - Angular velocity

#### RobotState
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **base_pos**: Position3D - `{x, y, z}` (meters)
- **base_quat**: Quaternion - `{x, y, z, w}`
- **base_lin_vel**: Vector3D - `{x, y, z}` (m/s)
- **base_ang_vel**: Vector3D - `{x, y, z}` (rad/s)
- **joint_pos**: Array[float] (length ACTION_DIM) - Joint positions (radians)
- **joint_vel**: Array[float] (length ACTION_DIM) - Joint velocities (rad/s)

#### DepthObservation
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **features**: Array[float] (length N, model-specific) - Pre-processed depth features

#### ParkourModelIO
Combined input model aggregating all telemetry for policy inference:
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **nav_cmd**: NavigationCommand (nested)
- **state**: RobotState (nested)
- **depth**: DepthObservation (nested)

### Inference Format

The policy model expects a **flat float32 tensor** with shape `(OBS_DIM,)`. The policy wrapper converts the nested `ParkourModelIO` structure to this flat tensor format. See policy wrapper documentation for the exact conversion order and layout.

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
- **For Unitree Go2**: 12 joints (4 legs × 3 joints per leg: hip_yaw, hip_pitch, knee)

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
- **Type**: Must be `numpy.ndarray`
- **Dtype**: Must be `float32`
- **Shape**: Must be `(OBS_DIM,)` where OBS_DIM = 753
- **Values**: Must be finite (no NaN or Inf)

### Command Validation
- **Payload size**: Must be multiple of 4 bytes (float32)
- **Dtype**: Must be `float32` after deserialization
- **Shape**: Must be 1D array
- **Values**: Must be finite (no NaN or Inf)
- **Action dimension**: Validated against expected ACTION_DIM (typically 12)

### Error Handling
- Invalid messages are logged and rejected
- Error responses are sent back to clients when validation fails
- Validation errors include detailed information about what failed

## Interface Actions

The HAL interface must support:

- **Get NavigationCommand** - Retrieve latest navigation command
- **Get RobotState** - Retrieve latest robot state (100+ Hz)
- **Get DepthObservation** - Retrieve latest depth observation (30-60 Hz)
- **Build ParkourModelIO** - Combine latest NavigationCommand, RobotState, and DepthObservation into a single model

**Synchronization**: All components in ParkourModelIO should have timestamps within < 10ms of each other for 100 Hz control.

### Constraints

- All floats must be `float32` dtype
- Quaternion must be normalized
- Joint arrays must match ACTION_DIM (model-specific, typically 12)
- Depth features array must match N (model-specific)
- Timestamps must be monotonically increasing
- Schema versions must be compatible across all components

## Command Models

Output data models for the Hardware Abstraction Layer (HAL). These represent actuator commands and inference responses flowing from the policy wrapper to hardware/simulation.

### Overview

Command models represent policy inference output - desired actions for robot actuators. They support high-frequency updates (100+ Hz) and include validation/error handling.

### Model Structures

#### JointCommand
- **timestamp_ns**: Integer (nanoseconds)
- **schema_version**: String
- **joint_pos**: Array[float] (length ACTION_DIM) - Desired joint positions (radians)

#### InferenceResponse
- **timestamp_ns**: Integer (nanoseconds)
- **inference_latency_ms**: Float (milliseconds)
- **joint_command**: JointCommand (nested)
- **model_version**: String
- **success**: Boolean
- **error_message**: String (optional, if success=False)

### Interface Actions

The HAL interface must support:

- **Send JointCommand** - Send joint position command to actuators
- **Validate Command** - Validate command shape, dtype, and ranges before sending
- **Get Acknowledgement** - Receive confirmation that command was received/processed
- **Handle Errors** - Process error responses if command is rejected

**Request-Response Pattern**: Commands use request-response to ensure ordering and acknowledgement.

### Constraints

- Array must be `float32` dtype
- Array shape must exactly match ACTION_DIM (model-specific, typically 12)
- Joint positions typically in range [-π, π] radians
- Inference latency should be < 15ms for 100 Hz control (target < 10ms)
- Commands must be generated within < 10ms from observation timestamp

### Validation

**Required checks**:
- Array shape matches ACTION_DIM
- Array is float32 dtype
- Timestamp is recent (not stale)

**Optional checks** (typically in actuator layer):
- Joint positions within limits
- Velocity limits (change from previous command)

## Endpoints

**TCP Endpoints** (configurable via `HAL_BASE_PORT`, default 6000):
- Camera: `tcp://host:6000`
- State: `tcp://host:6001`
- Commands: `tcp://host:6002`

**Inproc Endpoints** (same process):
- Camera: `inproc://hal_camera`
- State: `inproc://hal_state`
- Commands: `inproc://hal_commands`

## HAL Server Workflow

1. Create PUB sockets for camera and state, REP socket for commands
2. Bind to endpoints (inproc or TCP)
3. Set HWM=1 for latest-only semantics on PUB sockets
4. Main loop:
   - Publish camera telemetry (30-60 Hz): Convert depth features to `float32` array, send as multipart `[topic, schema_version, payload]`
   - Publish state telemetry (100+ Hz): Convert robot state to `float32` array, send as multipart `[topic, schema_version, payload]`
   - Receive command requests (non-blocking): Deserialize to `float32` array, apply to actuators, send acknowledgement

## HAL Client Workflow

1. Create SUB sockets for camera and state, REQ socket for commands
2. Connect to endpoints
3. Subscribe to topics: `"camera"` and `"state"`
4. Set HWM=1 for latest-only semantics on SUB sockets
5. Main loop:
   - Poll for telemetry messages (non-blocking with timeout)
   - Receive multipart messages: `[topic, schema_version, payload]`
   - Validate schema version before deserialization
   - Deserialize payload to `float32` arrays (if schema version is compatible)
   - Validate array shapes match expected dimensions
   - Build observation tensor and run inference
   - Send joint command as `float32` array
   - Wait for acknowledgement response

## Latest-Only Semantics

All PUB/SUB channels use HWM=1 (high-watermark=1):
- Only the latest message is kept in buffers
- Old messages are automatically dropped
- Subscribers always receive the most recent telemetry
- Prevents queue buildup and ensures real-time control uses fresh data

## Synchronization

- **Topic filtering**: Subscribers must subscribe to specific topics (`"camera"` or `"state"`)
- **Message ordering**: PUB/SUB has no guaranteed ordering; REQ/REP guarantees ordering
- **Timestamp synchronization**: Camera and state messages should have timestamps within < 10ms of each other (see Telemetry Models section)

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

The `hal_dump` tool inspects the current state of a HAL server, showing telemetry data and command endpoint status.

### Basic Usage

**New Format (Unified Observation):**
```bash
python -m HAL.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002
```

### Verbose Mode

Show detailed breakdown of observation/state components:

```bash
python -m HAL.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002 \
    --verbose
```

### In-Process Endpoints

For debugging in-process communication:

```bash
python -m HAL.tools.hal_dump \
    --observation_endpoint inproc://hal_observation \
    --command_endpoint inproc://hal_command
```

### Custom Action Dimension

If your robot has a different number of joints:

```bash
python -m HAL.tools.hal_dump \
    --observation_endpoint tcp://localhost:6001 \
    --command_endpoint tcp://localhost:6002 \
    --action_dim 18 \
    --verbose
```

### Output Format

The tool displays:
- **Observation/Telemetry**: Shape, dtype, statistics (min/max/mean)
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

📊 Observation Telemetry (New Format):
  Topic: observation
  Schema Version: 1.0
  Timestamp: 1705315845000000000 ns (1705315845.000000 s)
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
  Test Command Shape: (12,)
  Note: Commands are REQ/REP, no history available

================================================================================
```

## Debug Logging

The HAL supports runtime debug logging that can be enabled/disabled without restarting the system.

### Enabling Debug Logging

**In Code:**
```python
from HAL.ZMQ.client import HalClient
from HAL.ZMQ.server import HalServerBase

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
[ZMQ RECV] observation: shape=(753,), dtype=float32, min=-1.234, max=2.456
[ZMQ RECV] observation: ParkourObservation created successfully
[ZMQ SEND] command: shape=(12,), dtype=float32, min=-0.5, max=0.5
```

**Server (Sending):**
```
[ZMQ SEND] observation: shape=(753,), dtype=float32, min=-1.234, max=2.456
[ZMQ RECV] command: shape=(12,), dtype=float32, min=-0.5, max=0.5
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
    hal_server.publish_telemetry()
    hal_client.poll(timeout_ms=100)
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

### Wrong Observation Shape

If observation shape doesn't match expected (753):
1. Check that server is using correct observation format
2. Verify NUM_PROP, NUM_SCAN, etc. match training config
3. Use verbose mode to see detailed breakdown
4. Compare with training observation format

## Debugging Best Practices

1. **Use hal_dump for Quick Inspection**: Quick way to check if server is running and data is flowing
2. **Enable Debug Logging Selectively**: Only enable when debugging specific issues
3. **Check Verbose Output**: Use `--verbose` to understand data structure
4. **Monitor Performance**: Disable debug logging in production
5. **Use In-Process for Testing**: Use `inproc://` endpoints for faster local testing

# HAL ↔ MCU boundary

## Purpose

Defines the stable contract between **`hal/`** (Jetson, Isaac, portal) and **`firmware/`** (MCU transport) when leg joints move from Mega serial to CAN leg boards.

## Components

- **`hal.client`:** `JointCommand`, `HardwareObservations`, `JointCommandSource`
- **`hal.server.robot_definition`:** `RobotDefinition.get_mcu_joints()` — eighteen **`FL_hip_yaw` … `RR_knee`** names for hex
- **Jetson:** `hal/server/jetson/hal_server.py`, `hal/server/jetson/krabby_mcusdk.py`
- **Isaac fullsim:** `hal/server/isaac/firmware_serial_loop.py`, `hal/server/isaac/hal_server.py` (`firmware_transport`)
- **Firmware transports:** `firmware/krabby_mcu.py` (serial), `firmware/stm/mcu_can.py` (CAN), factory in **`firmware/mcu_factory.py`**
- **Callers:** `firmware/gui/app.py`, locomotion `apply_command`, gamepad → portal → HAL (no direct `krabby_mcu` in policy code)

## Critical points

### Stable HAL contracts

- **`JointCommand`:** dict of **radians** keyed by **`robot_definition.get_joint_names()`** (policy joint count); MCU path uses the **18-name** `mcu_joints` subset. Every `apply_command` must include **all** `mcu_joints` keys present in the command dict validation inside `KrabbyMCUSDK.apply_command`.
- **`HardwareObservations`:** shape and dtypes fixed in `hal/client/data_structures/hardware.py`. MCU layer supplies **`joint_positions`** (and optionally **`joint_velocities`** from host differentiation); **does not** fill `privileged_latent`, pack power, or Mega **`IMU`** segments (`ROBOT_SENSORS.md`).
- **`RobotDefinition`:** hex deployment sets **`mcu_joints`** to the eighteen HAL MCU names; empty tuple disables MCU SDK init on Jetson.

### What stays in `firmware/` vs `hal/`

| Concern | `firmware/` | `hal/` |
|---------|-------------|--------|
| Wire protocol | Serial `B`/`J` lines; CAN DBC frames | — |
| Backend selection | `mcu_factory.create_mcu_sdk()` reads **`KRABBY_MCU_STM`** | Passes `mcu_joints`, calls facade only |
| Joint encoding | Radians → PWM / normalized / DBC signals | Passes radians through `JointCommand` |
| 18 → 6 routing | `firmware/stm/ids.py` + dual-bus send order | — |
| Telemetry cache | `KrabbyMCUSDK.joints` dict (`JointTelemetry`) | Reads cache when filling observations on CAN leg path |
| Cameras, ZED IMU | — | `JetsonHalServer` catalog + `zed_imu.py` |
| Observation assembly | — | `set_observation()`, state vector layout |

**Rule:** `hal/` imports **`hal.server.jetson.krabby_mcusdk.KrabbyMCUSDK`** (facade) and **`firmware.mcu_factory`** — not `krabby_mcu` / `mcu_can` directly except inside the facade and GUI factory hook.

### MCU backend abstraction

**Factory:** `firmware/mcu_factory.py`

```text
create_mcu_sdk(mcu_joints, target=None, baud=DEFAULT_BAUD, auto_connect=True) -> McuJointBackend
```

**`McuJointBackend` protocol (behavior-compatible with the Jetson facade):**

| Method / property | Behavior |
|-------------------|----------|
| `connect()` | Idempotent; raises **`RuntimeError`** on failure when `auto_connect` |
| `close()` | Stops reader thread / CAN listeners |
| `is_connected()` | Backend ready for commands |
| `apply_command(JointCommand)` | Validates eighteen keys; sends to hardware |
| `joints` | `Dict[str, JointTelemetry \| None]` keyed by **`mcu_joints`** names (GUI + diagnostics) |

**Backends:**

| ID | Module | When active |
|----|--------|-------------|
| `serial_mega` | `firmware.krabby_mcu.KrabbyMCUSDK` | Default; `KRABBY_MCU_STM` unset |
| `can_legs` | `firmware.stm.mcu_can.KrabbyMCUCANSDK` | `KRABBY_MCU_STM=1` |

**Encoding split (important):**

- **Serial `serial_mega`:** keeps Mega short names (`FLHY`, …) inside `krabby_mcu` via `hal/server/jetson/krabby_mcusdk.py` `_hal_to_firmware_name` mapping.
- **CAN `can_legs`:** uses full **`mcu_joints`** strings end-to-end (`HOST_INTEGRATION.md`); no Mega abbreviations on the bus.

Shared math (`JOINT_LIMIT_RAD`, `JOINT_NEUTRAL`, rad→PWM) moves to **`firmware/joint_command_codec.py`** so Jetson facade and CAN encoder stay aligned.

### Jetson (`hal/server/jetson/hal_server.py`)

- **Init (USB serial):** if `robot_definition.get_mcu_joints()` non-empty → construct **`KrabbyMCUSDK`** with `mcu_target` / `mcu_baud` from `firmware.mcu_port.default_target()`.
- **Init (leg CAN):** `create_mcu_sdk(...)`; when **`KRABBY_MCU_STM=1`**, ignore serial `mcu_target`, open CAN via `firmware/stm/can_transport.py` env (`HOST_INTEGRATION.md`).
- **`apply_command`:** fixed signature; facade delegates to active backend; failures **log + re-raise**.
- **`set_observation` / `_build_state_vector`:** when backend is **`can_legs`** and telemetry fresh, map `joints[name].pos` normalized → radians into **`joint_positions`**; retain command echo fallback only if telemetry missing (bring-up). **Base IMU** still from primary ZED, not MCU.

### Isaac fullsim (`firmware_transport`)

| Value | Command path | Telemetry |
|-------|----------------|-----------|
| `in_process` | Direct host jog on sim core (default) | Sim plant |
| `serial_sdk` | `KrabbyMCUSDK` → PTY → three-board sim (`firmware_serial_loop.py`) | Serial parser |
| `can_vcan` | `KrabbyMCUCANSDK` on **`vcan0`** + virtual leg node (`VIRTUAL_CAN.md`) | DBC decode |

**`can_vcan` jog rule:** `CanCommandLoop.apply_hal_command` encodes **only joints with non-neutral commanded radians**; all other jog PWM fields **zero** in the outgoing `LegCommand` batch (Isaac moves one joint at a time).

Implementation adds **`CanCommandLoop`** parallel to **`SerialSdkCommandLoop`**; `IsaacSimHalServer` selects loop from `firmware_transport`.

### GUI (`firmware/gui/app.py`)

- Uses the same **`create_mcu_sdk`** as HAL when **`--stm`** / **`KRABBY_MCU_STM=1`**.
- UI continues to poll **`sdk.joints`** and jog via backend-specific send APIs hidden behind the facade.
- No `hal/` import in GUI — only `firmware` facade (preserves Tk dependencies).

### Gamepad / AI / `krabby run`

- Controller and policy send **`JointCommand`** into Jetson HAL only; transport switch is **transparent** to callers.
- Locomotion container: **`KRABBY_MCU_STM=1`** in systemd unit for CAN leg robots; USB serial Megas omit it.

### Dual-bus joint router

Inside **`KrabbyMCUCANSDK.apply_command`:**

1. Split `command.to_positions_dict()` to entries whose names ∈ `mcu_joints`.
2. For each name, `ids.joint_to_leg_bus_index(name)` → `(iface, leg_index, joint_index)`.
3. Batch per leg into one DBC `LegCommand` frame per node per tick.
4. Transmit on **`can0`** legs **`FL`, `ML`, `RL`**; on **`can1`** legs **`FR`, `MR`, `RR`** (`CAN_PROTOCOL.md`).
5. Reader thread demuxes **`LegTelemetry`** into `self.joints[name]`.

Ordering: send all legs each HAL tick (~100 Hz); respect **100 ms** stale timeout on boards independently.

### Behavior that must not regress (policy / teleop)

- **`apply_command`** synchronous; no hidden queue in HAL MCU layer.
- Neutral command (|rad| ≤ **`NEUTRAL_RAD_THRESHOLD`**) → jog PWM **0** on that joint (serial and CAN).
- Non-neutral → jog PWM scaled by **`JOINT_LIMIT_RAD`** (`_rad_to_pwm`); position-hold over CAN is optional once pots are trusted on all joints (DBC position fields).
- **`connect()`** on HAL server startup when `mcu_auto_connect=True`.
- **`close()`** on server shutdown.
- Eighteen-actuator hex commands with twelve-DOF robot definition still **raises** with existing hint strings (portal mismatch guard).

### Sensor gaps (MCU layer does not fix)

| Stream | CAN leg path |
|--------|----------------|
| Pack / `BATT` | Not from MCU SDK — deferred (`ROBOT_SENSORS.md`) |
| Mega `IMU` segment | Not from MCU SDK — use ZED / Isaac stub |
| Contact forces | HAL zeros |
| `privileged_latent` | `None` on hardware |

## Reference

### File touch map

| Area | Files |
|------|--------|
| CAN SDK | `firmware/stm/mcu_can.py`, `can_transport.py`, `codec.py` |
| Isaac CAN loop | `hal/server/isaac/firmware_can_loop.py`, `firmware_transport='can_vcan'` |
| Joint routing | Router tests; `ids.py` |
| Backend factory | `firmware/mcu_factory.py`, env wiring in fleet/systemd |
| Jetson facade | `hal/server/jetson/krabby_mcusdk.py` delegate; `hal_server.py` measured joints |
| GUI | `firmware/gui/app.py`, `gui/__main__.py --stm` |

### Isaac `firmware_transport` enum

```text
in_process | serial_sdk | can_vcan
```

### Jetson facade public surface (frozen)

```text
KrabbyMCUSDK(mcu_joints, target=None, baud=DEFAULT_BAUD, auto_connect=True)
  .connect() -> bool
  .close() -> None
  .is_connected() -> bool
  .apply_command(command: JointCommand) -> None
  # .joints via ._mcu.joints after connect
```

Internal `_mcu` is the backend instance from `create_mcu_sdk`; HAL must not branch on concrete type.

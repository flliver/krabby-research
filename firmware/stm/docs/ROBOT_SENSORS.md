# Robot-level sensor streams

## Purpose

Maps every robot-level measurement stream to its physical source and to what the HAL and locomotion stack consume when joints are on six CAN leg boards.

## Components

- **Leg CAN telemetry** — eighteen joints × (`LEG_BOARD_CONTROL.md` fields) on six nodes
- **Host HAL** — `HardwareObservations`, `JointCommand`, `RobotDefinition.mcu_joints`
- **Orin perception** — ZED RGB-D and onboard IMU (`JETSON_SENSOR_CATALOG`)
- **MCU USB serial** — elected leader board **IMU** + **BATT** text segments on the three Mega controllers (not carried on the CAN leg bus)
- **Robot power electronics** — pack shunt / midpoint monitors on the dedicated power board, not on the leg PCB

## Critical points

- **Architecture split:** joint actuation and joint-local sensing move to **six identical leg boards** on **CAN-FD**. **IMU #1** (policy base attitude) comes from the **primary ZED** on the Orin (`zed_imu.py`, `JETSON_SENSOR_CATALOG`). Optional **Qwiic** accessories on **I2C1** (`HARDWARE.md`), **at most one each per robot:** **Qwiic IMU** → **`QwiicImu`** CAN TX from the hosting leg; **Krabby OLED** → **no** CAN TX — that board **soaks** leg-bus frames and renders locally (`DISPLAY.md`). Pack monitoring stays off the leg bus unless added later.
- **Normative joint source (CAN robot):** measured **`position_norm`** (and connection/fault bits) from **`LegTelemetry`** per `CAN_PROTOCOL.md`, aggregated in host `firmware/stm/mcu_can.py` into the same shape as `firmware.interfaces.joint_telemetry.JointTelemetry` keyed by HAL **`mcu_joints`** names (`FL_hip_yaw` … `RR_knee`).
- **HAL joint observation (target):** `HardwareObservations.joint_positions` in **radians** uses **measured** leg positions when CAN telemetry is healthy; **command echo is fallback only** during bring-up (`hal/server/jetson/hal_server.py` placeholder path). **`joint_velocities`** remain **derived on the host** (finite difference of measured positions) until leg firmware publishes explicit joint rates.
- **Policy-critical on hardware today:** primary **RGB-D** + **scan_features**, **`base_quat_w`**, **`base_ang_vel_b`** (ZED IMU when enabled), **`joint_positions`** (18). **`privileged_latent`** is simulation-only (`None` on hardware). **`contact_forces`** stay **zero-filled** on Jetson until a foot sensor stream exists.
- **CLI / operator:** `krabby-firmware --stm telemetry` covers **leg CAN** fields plus **`NodeStatus`**. It does **not** surface **IMU #1** (that is the ZED / HAL path, not MCU CLI telemetry). Pack lines stay absent on the CAN-only leg path until a power-board stream exists (below).

### Stream inventory

| Stream ID | Quantity | Physical source (CAN leg robot) | Host consumer | Availability |
|-----------|----------|-----------------------------------|---------------|------------------|
| `joint_position` | 18 | Leg board pot → `LegTelemetry` | HAL observations, policies, GUI, MCU SDK | **Required** — CAN |
| `joint_connection` | 18 | Leg board pot/current logic | Safety, telemetry display, fault handling | **Required** — CAN |
| `joint_current_raw` | 18 | Leg bridge sense → `LegTelemetry` | Diagnostics, connection inference | **Required** — CAN |
| `joint_hall_edges` | 18 | Off-board Hall on harness when populated | Diagnostics (`LEG_BOARD_CONTROL.md`) | **Optional** — zero when unwired |
| `joint_faults` | 18 | Leg firmware bitmask | Diagnostics, safe-stop policy hooks | **Required** — CAN |
| `node_status` | 6 | Leg board `NodeStatus` | Discovery, `show`, flash | **Required** — CAN |
| `pcb_temperature` | 6 | Leg board `PCB_TEMP_ADC` (`HARDWARE.md`) | Thermal logging | **Bench** — DBC field TBD (`LegTelemetry` vs `NodeStatus`) |
| `base_imu_accel_gyro` (**IMU #1**) | 1 | **Orin:** primary **ZED** onboard IMU (not CAN) | `base_quat_w`, `base_ang_vel_b` in HAL | **Required** when ZED IMU enabled |
| `qwiic_imu` (aux body IMU) | 0–1 | **Optional:** **Qwiic** on **one** leg PCB → **`QwiicImu`** CAN | Diagnostics, future HAL fusion, CLI | **Optional** — no frame when unwired |
| `krabby_oled` (operator UI) | 0–1 | **Optional:** **Qwiic OLED** on **one** leg PCB; feed from **CAN RX mirror** | On-robot display only | **Not** a host stream — see `DISPLAY.md` |
| `base_lin_vel_b` | 1 | **Orin:** ZED visual odometry when tracking enabled | HAL velocity | **Optional** — feature-flagged on ZED |
| `camera_rgb_depth` | N | **Orin:** catalog cameras | Policy scan, collision, GUI | **Orin** — not on leg CAN |
| `pack_voltage_current` | 1 | **Deferred:** **INA228** on robot power board (not on leg PCB) | `BatteryTelemetry`, power UI | **Unavailable** on CAN-only leg path |
| `pack_charge_coulombs` | 1 | **Deferred:** same as pack shunt | Fuel-gauge style logging | **Unavailable** |
| `battery_a_voltage` | 1 | **Deferred:** midpoint monitor on robot power board | Divergence checks | **Unavailable** |
| `battery_b_voltage` | 1 | **Deferred:** inferred second pack tap | Divergence checks | **Unavailable** |
| `pack_voltage_region` | 1 | **Deferred:** firmware policy region enum | Soft/hard cut logic | **Unavailable** |
| `mcu_imu_segment` | 1 | **Unavailable:** was Mega leader **LSM6DSO** serial `IMU` tag | `KrabbyMCUSDK.imu`, serial CLI | **Replaced by ZED path on Orin** |
| `joint_velocity` | 18 | Host derivative of `joint_position` | HAL `joint_velocities` | **Synthesized** on Orin |
| `contact_force` | 5 | No hardware stream | HAL placeholder | **Unavailable** (zeros) |

- **Three-board serial trunk:** one elected **leader** Mega emits **`IMU`** + **`BATT`** once per telemetry tick; all three Megas share one USB serial link to the host. That wire format is **not** carried on the CAN leg bus.
- **Power on the leg PCB:** **`VM`** feed and per-joint **`VM_FUSED_M#`** (`HARDWARE.md`) are **actuator supply only** — no pack current/voltage measurement on the leg controller schematic.
- **Host HAL integration:** CAN transport populates the **joint** cache before `JetsonHalServer.set_observation()`. **IMU #1** stays on the **ZED** publisher. **`QwiicImu`** is decoded in `mcu_can` for CLI/diagnostics; it does **not** replace **`base_quat_w`** unless a future HAL policy wires it in. Pack fields stay **out of** `mcu_can` until a power-board stream exists (do not fake BATT on CAN).
- **Single Qwiic install:** the robot carries **at most one** Qwiic IMU. Host treats **>1** leg with `QwiicImu.valid=1` as a **configuration error** (`show` / telemetry warning).
- **Simulation (`VIRTUAL_CAN.md`):** virtual leg nodes emit **`LegTelemetry`** only. **Base IMU** uses the Isaac HAL stub. Pack streams stay **absent** unless a test fixture injects them.
- **Future pack stream:** add a new row in this inventory when one of these is specified: **power CAN node**, **Orin I2C monitor service**, **GPIO estop-only** — HAL must not treat pack data as authoritative until then.

## Reference

### HAL `mcu_joints` (joint position stream keys)

| Leg | Names (each has `joint_position` + diagnostics) |
|-----|--------------------------------------------------|
| FL | `FL_hip_yaw`, `FL_hip_pitch`, `FL_knee` |
| FR | `FR_hip_yaw`, `FR_hip_pitch`, `FR_knee` |
| ML | `ML_hip_yaw`, `ML_hip_pitch`, `ML_knee` |
| MR | `MR_hip_yaw`, `MR_hip_pitch`, `MR_knee` |
| RL | `RL_hip_yaw`, `RL_hip_pitch`, `RL_knee` |
| RR | `RR_hip_yaw`, `RR_hip_pitch`, `RR_knee` |

### Serial `BATT` segment fields (robot power board / leader Mega; not on CAN leg bus)

| Field | Meaning |
|-------|---------|
| `pack_volts` | Pack bus voltage |
| `pack_current_amperes` | Pack current |
| `pack_power_watts` | Pack power |
| `pack_charge_coulombs` | Coulomb counter |
| `battery_a_volts` | Tap A |
| `battery_b_volts` | Tap B |
| `divergence` | Monitors disagree |
| `pack_region` | `PackVoltageRegion` enum |
| `pack_valid` | Pack INA sample valid |
| `midpoint_valid` | Midpoint INA sample valid |

Host module: `firmware/interfaces/battery_telemetry.py`.

### Serial `IMU` segment fields (leader Mega LSM6DSO; not on CAN leg bus)

| Field | Meaning |
|-------|---------|
| `accel` | m/s², 3 axes |
| `gyro` | rad/s, 3 axes |
| `temp_c` | die temperature |
| `valid` | sample trusted |

Host module: `firmware/interfaces/imu_telemetry.py`. On the CAN leg robot, **base attitude on Orin comes from the ZED IMU** when the primary catalog camera has IMU enabled.

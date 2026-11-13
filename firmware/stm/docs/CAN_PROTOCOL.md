# CAN field bus

## Purpose

CAN-FD link between the Orin host and six leg controller nodes (three joints each).

## Components

- Physical CAN-FD daisy chain per robot bus (up to three leg boards per bus)
- Orin SocketCAN: robot uses `can0` + `can1`; bench uses one interface per mode below
- Per-leg controller node (STM32 + transceiver)
- Host encoder/decoder (`krabby_leg.dbc`, `firmware/stm/` Python)
- Discovery and heartbeat frames per node

## Critical points

- **Link:** CAN-FD, ISO 11898-1; **64-byte** max payload per frame.
- **Bit timing:** **500 kbit/s** arbitration phase; **2 Mbit/s** data phase (nominal — match transceiver and STM32 clock config).
- **IDs:** **29-bit extended** CAN identifiers. **Allocation below is normative;** `../dbc/krabby_leg.dbc` must use the same numeric IDs.
- **Topology:** **Two buses** on the robot; **three leg boards** daisy-chained per bus; **120 Ω** termination at each bus end; host at one end of each chain.
- **Bus assignment:** `can0` → legs **FL, ML, RL**; `can1` → legs **FR, MR, RR** (leg prefix = HAL `RobotDefinition.legs`).
- **Robot:** two SocketCAN devices — **`can0`** and **`can1`** (leg assignment in bus table below).
- **Bench (hardware):** one physical CAN interface, default name **`can0`**; one or more leg nodes on a single chain; same DBC as the robot.
- **Bench (sim/CI):** one virtual interface **`vcan0`** (`VIRTUAL_CAN.md`); same DBC; host treats it as a single bus (map all logical legs onto that interface per sim doc).
- **Nodes:** six logical nodes **`FL`, `FR`, `ML`, `MR`, `RL`, `RR`**; each node drives **`hip_yaw`, `hip_pitch`, `knee`** for that leg (full HAL names in Reference).
- **Commands:** host sends per-joint **jog PWM** and **normalized position** signals (DBC fields per joint) addressed by leg + joint; typical HAL command period **50 Hz**; boards treat commands older than **100 ms** as stale and enter safe state per `LEG_BOARD_CONTROL.md`.
- **Telemetry:** each node publishes joint telemetry at **100 Hz** (aggregated in DBC); host may poll additional status via discovery frames.
- **Discovery:** on boot and on request, each node emits a **NodeStatus** frame: leg ID, firmware build identity string, bootloader vs application mode (identity encoding in Reference).
- **Heartbeat:** periodic **`NodeStatus`** at **1 Hz** while application runs; absence **> 500 ms** marks the node offline on the host (no separate heartbeat message ID).
- **Orin bring-up (robot):** `scripts/stm/can_setup_orin.sh` brings up **`can0`** and **`can1`** as CAN-FD (fd on, nominal/data bit timing as above); both interfaces required before `krabby-firmware --stm`.
- **Leg disconnect:** if one leg’s chain segment is unplugged, other legs on the **same** and **other** bus keep communicating; host stops commanding the missing leg’s joints; when the segment returns, node discovery + heartbeat restores the leg without full-robot power cycle.
- **Host command semantics:** HAL `JointCommand` uses **radians** per `mcu_joints` name; host maps to CAN signals (same joint order as `KRABBY_HEX_DEFINITION.mcu_joints`).

## Reference

### Orin SocketCAN (J4012-class robot)

| Interface | Legs on this robot |
|-----------|-------------------|
| `can0` | **FL**, **ML**, **RL** |
| `can1` | **FR**, **MR**, **RR** |

**Nominal leg-bus load per interface (three nodes):** up to **3 × 100 Hz** `LegTelemetry` + **3 × 50 Hz** `LegCommand` + **3 × 1 Hz** `NodeStatus` ≈ **450 frames/s** before flash or discovery. Bursts (discovery, six-leg flash) should be profiled on robot hardware.

On Jetson Orin, **`can0`** and **`can1`** are SocketCAN names for the on-SoC **MTT CAN** controllers (`mttcan@c310000` and `mttcan@c320000`). NVIDIA’s Orin documentation cites a **peak** rate on the order of **8000 messages/s @ 1 Mbps** per controller (typical sustained average lower); our leg-bus duty cycle is far below that, but high-rate tooling (simultaneous flash, flood tests) should still be measured on hardware. Carrier-specific bring-up (transceiver STBY GPIO, connector pinout, CAN-FD `ip link` examples) is in the Seeed docs below—not Krabby-specific limits.

**Official references (host `can0` / `can1`):**

| Topic | Link |
|-------|------|
| Jetson CAN enable, `mttcan` driver, SocketCAN | [NVIDIA Jetson Linux — Hardware Setup: CAN](https://docs.nvidia.com/jetson/l4t/Tegra%20Linux%20Driver%20Package%20Development%20Guide/hw_setup_jetson_can.html) |
| Orin MTT CAN throughput / IRQ behavior (cites Orin TRM) | [NVIDIA Forums — Maximum CAN interrupt rate](https://forums.developer.nvidia.com/t/maximum-can-interrupt-rate/342084) |
| reComputer Robotics J401/J4012: `can0`/`can1` wiring, GPIO STBY, CAN-FD setup | [Seeed Wiki — J401 getting started (CAN)](https://wiki.seeedstudio.com/recomputer_jetson_robotics_j401_getting_started/) |
| reComputer Robotics user manual (connectors, CAN modes) | [Seeed/DigiKey PDF — reComputer Robotics User Manual](https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/8884/100001302.pdf) |

### Leg index (for ID arithmetic)

| Leg | Index | Bus |
|-----|-------|-----|
| FL | 0 | can0 |
| FR | 1 | can1 |
| ML | 2 | can0 |
| MR | 3 | can1 |
| RL | 4 | can0 |
| RR | 5 | can1 |

### Message IDs (extended, hex)

| Message | CAN ID | Dir | Rate | Payload role |
|---------|--------|-----|------|----------------|
| `LegCommand` | `0x0C000000 + index` | host → leg | ≤50 Hz | jog / position for 3 joints on that leg |
| `LegTelemetry` | `0x0C000100 + index` | leg → host | 100 Hz | position, current, status per joint |
| `NodeStatus` | `0x0C000200 + index` | leg → host | 1 Hz + on boot | leg ID, firmware identity, app/boot mode |
| `QwiicImu` | `0x0C000400 + index` | leg → host | 100 Hz when present | Optional Qwiic IMU on **this** board only (`valid`); silent when unwired |
| `DiscoveryRequest` | `0x0C0007F0` | host → all | on demand | prompt all nodes to emit `NodeStatus` |

Examples: `LegCommand` for **FL** = `0x0C000000`; for **RR** = `0x0C000005`. Flash/bootloader frames (see `FLASH.md`) use the **`0x0C000300`–`0x0C0003FF`** range (defined in `FLASH.md`).

### Firmware build identity (`NodeStatus`)

| Condition | `NodeStatus` identity string |
|-----------|------------------------------|
| Release build (tagged) | Semantic version, e.g. `1.2.3` |
| Otherwise | Full git SHA (40 hex chars) |

### HAL `mcu_joints` (command / telemetry order)

| Leg | `hip_yaw` | `hip_pitch` | `knee` |
|-----|-----------|-------------|--------|
| FL | `FL_hip_yaw` | `FL_hip_pitch` | `FL_knee` |
| FR | `FR_hip_yaw` | `FR_hip_pitch` | `FR_knee` |
| ML | `ML_hip_yaw` | `ML_hip_pitch` | `ML_knee` |
| MR | `MR_hip_yaw` | `MR_hip_pitch` | `MR_knee` |
| RL | `RL_hip_yaw` | `RL_hip_pitch` | `RL_knee` |
| RR | `RR_hip_yaw` | `RR_hip_pitch` | `RR_knee` |

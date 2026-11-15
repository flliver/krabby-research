# Robot-level sensor streams

## Purpose

Where joint and body measurements come from on a six-board CAN leg robot.

## Components

- **Leg CAN (`can0` / `can1`):** eighteen joints — **`TELEMETRY_<leg_id>`** and **`BOARD_INFO_<leg_id>`** (`krabby.dbc`)
- **Orin host:** RGB-D (**ZED**), Orin IMU, and other platform sensors consumed by HAL outside the leg DBC
- **CLI:** `krabby-firmware-stm telemetry` (target) decodes joint fields from leg CAN

## Critical points

- **Joints:** measured pot, hall, duty, **`position`**, fault bits from **`TELEMETRY_*`**; firmware identity from **`BOARD_INFO_*`** at 1 Hz.
- **HAL:** `hal_joint_names.py` maps **`mcu_joints`** ↔ `(leg_id, joint_id)` when filling observations from the CAN SDK.
- **Body IMU / vision:** ZED and Orin IMU feed HAL on the Jetson; they do not use leg-board CAN frames.
- **Pack monitor and leg Qwiic devices:** see `PACK_MONITOR.md` and `DISPLAY.md` when those buses are defined.

### Stream inventory

| Stream | Qty | Source | Consumer |
|--------|-----|--------|----------|
| Joint duty / pot / hall / position | 18 | `TELEMETRY_*` on leg CAN | HAL, GUI, diagnostics |
| Firmware identity | 6 | `BOARD_INFO_*` on leg CAN | CLI, support |
| Base IMU / RGB-D | 1 | Orin (ZED stack) | HAL, policy |
| Pack / auxiliary Qwiic | — | `PACK_MONITOR.md`, `DISPLAY.md` | HAL (when wired) |

## Reference

Joint field semantics: `LEG_BOARD_CONTROL.md`. Wire IDs: `CAN_PROTOCOL.md`.
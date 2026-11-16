# Leg board identity

## Purpose

How each leg board learns **`leg_id` 1…6** for CAN filtering and telemetry IDs.

## Components

- **Node ID switch** on PCB (`HARDWARE.md`, `firmware/stm/uno-v0.3/App/Src/board.c` `Board_GetNodeId()`)
- **`board_ids.h`** — `krabby_leg_code_valid()` for switch **`leg_id` 1…6**
- **NV calibration** pages (joint travel) — separate from identity

## Critical points

- **Primary path:** three-position switch encodes **`leg_id` 1…6** at boot. Invalid encoding → motors stay safe / asleep until fixed.
- **CAN:** board filters **`SET_DUTY`**, accepts **`CALIBRATE`** only on **`0x180 + leg_id`**, transmits on **`0x200 + leg_id`** and **`0x300 + leg_id`** (`CAN_PROTOCOL.md`, `ids.py`).
- **Robot install:** set switch to match physical leg before power-on; no host assignment required.
- **HAL / GUI labels** (`FL`…`RR`) map to **`leg_id`** in `ids.LEG_ID_BY_DBC_SUFFIX` — not on the wire.

| `leg_id` | Bus |
|--------|-----|
| 1, 3, 5 | `can0` |
| 2, 4, 6 | `can1` |

## Host NV provisioning (not on the bus today)

Older plans used **`ProvisionSet`**, **`DiscoveryRequest`**, and host-assigned leg codes over CAN. Those frames are **not** in **`krabby.dbc`**. Use the **switch** (or OpenBLT tools under `firmware/stm/uno-v0.3/`) until the DBC gains provisioning messages.
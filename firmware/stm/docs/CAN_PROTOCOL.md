# CAN field bus

## Purpose

CAN-FD link between the host and six leg controller nodes (three joints each).

## Components

- Two robot buses (`can0`, `can1`); bench sim on single **`vcan0`**
- Schema: **`../dbc/krabby.dbc`** (11-bit IDs, all frames FD)
- Host encode/decode: `firmware/stm/codec.py`, `ids.py`
- Firmware filters: `include/krabby/board_ids.h`

## Critical points

- **Link:** CAN-FD, up to **64-byte** payload; **BRS** on data phase.
- **Bit timing:** **1 Mbit/s** arbitration, **5 Mbit/s** data (`can_setup_orin.sh`, `python-can` `fd=True`, FDCAN2 on board).
- **IDs:** **11-bit standard** only; numeric table below matches `krabby.dbc` and `ids.py`.
- **Topology:** three legs daisy-chained per bus; **120 Ω** at each bus end.
- **Bus assignment:** `can0` → **`leg_id` 1, 3, 5**; `can1` → **2, 4, 6** (`IDENTITY.md`).
- **Host TX:** every **`SET_DUTY`** is sent on **both** `can0` and `can1`; each board applies only its **`leg_id`** slice (six int16 duties in the 36-byte frame).
- **Commands:** one **`SET_DUTY`** (`0x100`) carries all **18** joint duties (scale **0.001**, **−1.0…1.0**). Per-leg **`CALIBRATE`** (`0x180 + leg_id`) starts on-board stall cal for joint **2 or 3** only.
- **Telemetry:** each node **`TELEMETRY_<leg_id>`** (`0x200 + leg_id`) at **50 ms**; **`BOARD_INFO_<leg_id>`** (`0x300 + leg_id`) at **1 Hz** (version + git commit).
- **Command timeout:** firmware **holds last duty** if the host stops sending (no bus-stale safe-stop).
- **Sim:** single **`vcan0`**, MTU **72**; same DBC (`VIRTUAL_CAN.md`).

## Reference

### Orin SocketCAN

| Interface | Legs (`leg_id`) |
|-----------|-----------------|
| `can0` | 1, 3, 5 |
| `can1` | 2, 4, 6 |

### Message IDs (hex)

| DBC name | CAN ID | Dir | Rate | Role |
|----------|--------|-----|------|------|
| `SET_DUTY` | `0x100` | host → all | per host tick | all 18 duties |
| `CALIBRATE_<n>` | `0x180 + n` | host → leg *n* | on command | joint 2 or 3 |
| `TELEMETRY_<n>` | `0x200 + n` | leg *n* → host | 50 ms | hall, duty, pot, position, flags |
| `BOARD_INFO_<n>` | `0x300 + n` | leg *n* → host | 1 s | major, minor, patch, commit |

*`n` = **`leg_id` 1…6** (not `leg_index`).*

### Leg / joint on wire

| `leg_id` | `leg_index` | Bus |
|--------|-------------|-----|
| 1 | 0 | can0 |
| 2 | 1 | can1 |
| 3 | 2 | can0 |
| 4 | 3 | can1 |
| 5 | 4 | can0 |
| 6 | 5 | can1 |

Joint **1 / 2 / 3** = **Y / H / K** on every node (`IDENTITY.md`). Signals **`L1_J1`…`L6_J3`** in `SET_DUTY`; per-leg telemetry uses **`J1_*`…`J3_*`**.

### Host observation path

HAL **`mcu_joints`** names stay on the Jetson edge (`hal_joint_names.py`). Policy may send **duty** on the `--stm` path (not radians on the wire). See `HAL_MCU_LAYER.md`.
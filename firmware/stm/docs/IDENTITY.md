# Leg and joint identity (firmware / CAN)

## Purpose

Single numbering scheme for leg boards, CAN, CLI, and firmware structs. Human labels (“front left hip yaw”, `FLHY`, `FL_hip_yaw`) are **not** used below the HAL adapter.

## Components

- **`leg_id`** — **1…6** (switch value, operator docs, CLI integers)
- **`joint_id`** — **1…3** on each leg: **1 = Y** (yaw), **2 = H** (hip pitch), **3 = K** (knee)
- **`leg_index`** — **0…5** runtime / CAN ID math only: **`leg_index = leg_id - 1`**
- **`joint_index`** — **0…2** array slot on the board: **`joint_index = joint_id - 1`**
- **Python:** `firmware/stm/ids.py` — CAN routing, buses, conversions
- **C:** `firmware/stm/include/krabby/board_ids.h` — `leg_index` enums and CAN bases
- **HAL / policy names:** `firmware/stm/hal_joint_names.py` — maps `(leg_id, joint_id)` ↔ `RobotDefinition.mcu_joints` strings **only** at the Jetson/policy edge

## Critical points

### Leg map

| `leg_id` | Side | Bus | Switch | Legacy corner label (GUI only) |
|--------|------|-----|--------|--------------------------------|
| 1 | left | `can0` | 1 | FL |
| 2 | right | `can1` | 2 | FR |
| 3 | left | `can0` | 3 | ML |
| 4 | right | `can1` | 4 | MR |
| 5 | left | `can0` | 5 | RL |
| 6 | right | `can1` | 6 | RR |

**Left legs:** **1, 3, 5** on **`can0`**. **Right legs:** **2, 4, 6** on **`can1`**.

### Joint map (every leg PCB)

| `joint_id` | Letter | Board motor | `joint_index` |
|-----------|--------|-------------|---------------|
| 1 | **Y** | M1 | 0 |
| 2 | **H** | M2 | 1 |
| 3 | **K** | M3 | 2 |

### What is dead on the leg path

| Retired | Replaced by |
|---------|-------------|
| Mega **`FLHY`**, **`FRHL`**, … | **`(leg_id, joint_id)`** or duty triplet **`(leg_id, y, h, k)`** |
| Firmware / CAN use of **`FL_hip_yaw`** strings | **`leg_id` + `joint_id`** in structs; HAL strings only in **`hal_joint_names.py`** |
| Corner strings in STM32 structs | **`uint8_t leg_id`**, **`uint8_t joint_id`** (or fixed **3** slots per node) |

### CAN and DBC

- CAN IDs still use **`leg_index` 0…5** (`0x0C000000 + leg_index`, … — `CAN_PROTOCOL.md`).
- DBC message names **`LegCommand_FL`** … **`_RR`** are **wire labels** tied to **`leg_id` 1…6** (table above); signals per joint remain **`hip_yaw_*` / `hip_pitch_*` / `knee_*`** until a schema edit renames them **Y/H/K**. Layout order is always **joint 1, 2, 3**.

### Host boundaries

| Layer | Identity |
|-------|----------|
| Leg firmware, OpenBLT, DBC encode/decode | **`leg_index`**, **`joint_index`**, no HAL strings |
| **`krabby-firmware --stm` CLI** | **`leg_id` 1…6**, **`joint_id` 1…3** (or Y/H/K flags) |
| **`KrabbyMCUCANSDK` / policy** | Radians keyed by HAL **`mcu_joints`** via **`hal_joint_names.mcu_joint_name(leg_id, joint_id)`** |
| **GUI / fleet UI** | Full English strings; never required on device |

## Reference

### CLI examples (intent)

```text
# Jog leg 1, joint 1 (Y) — not "FLHY"
duty 1 1 0.4 0 0 0    # grant-style: leg 1, Y duty 0.4, H/K hold
```

### Unassigned

| Condition | `leg_id` | `leg_index` | NV `leg_code` |
|-----------|----------|-------------|----------------|
| Provisioned | 1…6 | 0…5 | 0…5 |
| Unknown | — | **6** | **255** |

Switch encoding: **1…6** equals **`leg_id`** when valid (`PROVISIONING.md`).

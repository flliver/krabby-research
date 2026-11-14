# Leg and joint identity (firmware / CAN)

## Purpose

Single numbering scheme for leg boards, CAN, CLI, and firmware. Corner strings (`FL`, `FLHY`, …) are **host/GUI only**.

## Components

- **`leg_id`** — **1…6** (node switch, CLI)
- **`joint_id`** — **1…3**: **1 = Y**, **2 = H**, **3 = K**
- **`leg_index`** — **0…5**: **`leg_index = leg_id - 1`**
- **`joint_index`** — **0…2**: **`joint_index = joint_id - 1`**
- **Python:** `firmware/stm/ids.py`
- **C:** `firmware/stm/include/krabby/board_ids.h`
- **HAL:** `firmware/stm/hal_joint_names.py` — `mcu_joints` strings at policy edge only

## Critical points

### Leg map

| `leg_id` | Side | Bus | Switch |
|--------|------|-----|--------|
| 1 | left | `can0` | 1 |
| 2 | right | `can1` | 2 |
| 3 | left | `can0` | 3 |
| 4 | right | `can1` | 4 |
| 5 | left | `can0` | 5 |
| 6 | right | `can1` | 6 |

### Joint map (every leg PCB)

| `joint_id` | Letter | Motor | `joint_index` |
|-----------|--------|-------|---------------|
| 1 | Y | M1 | 0 |
| 2 | H | M2 | 1 |
| 3 | K | M3 | 2 |

### CAN and DBC

- IDs use **`leg_id` 1…6** in arithmetic: **`0x200 + leg_id`**, etc. (`CAN_PROTOCOL.md`).
- **`SET_DUTY`** signals **`L1_J1`…`L6_J3`**; per-leg **`TELEMETRY_<n>`** uses **`J1_*`…`J3_*`**.

### Host boundaries

| Layer | Identity |
|-------|----------|
| Firmware, DBC | **`leg_id`**, **`joint_id`**, no HAL strings |
| **`krabby-firmware-stm`** | **`duty <leg> <j1> <j2> <j3>`** (float duties) |
| HAL / GUI with `--stm` | Duty **−1…1** after mapper; wire via **`SET_DUTY`** |

## Reference

```text
krabby-firmware-stm --virtual duty 1 0.5 0 0   # leg 1, Y duty 0.5, H/K hold
```

Switch **1…6** equals **`leg_id`** when valid (`PROVISIONING.md`).
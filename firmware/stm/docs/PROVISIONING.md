# Leg board provisioning

## Purpose

Assigns each identical leg controller PCB a stable leg identity (`FL` … `RR`) that drives CAN addressing, HAL joint names, and calibration NV.

## Components

- **Six leg codes:** **`FL`, `FR`, `ML`, `MR`, `RL`, `RR`** (one board each on the robot)
- **Node ID switch:** three switches on the leg PCB encode **1–6** at boot (`HARDWARE.md`)
- **On-chip NV** leg byte (`LEG_BOARD_CONTROL.md` layout) — survives application reflash; used when switch invalid or bench provisioning
- **CAN `NodeStatus`** — publishes current leg code and firmware identity (`CAN_PROTOCOL.md`)
- **Host tools** — `krabby-firmware --stm show`, `krabby-firmware --stm provision`
- **Shared ID map** — `firmware/stm/board_ids.h` + `firmware/stm/ids.py`

## Critical points

- **Identical PCBs:** every leg board ships the same BOM and firmware image; **no** leg-specific assembly variant. Physical mounting position does **not** imply identity — only stored leg code does.
- **Source of truth at boot:**
  1. Read **node ID switch**. If encoded value is **1–6**, map to **leg index 0–5** and **leg code** byte (table below). Switch wins over NV when valid.
  2. Else use NV byte at offset **`0x0005`** (`leg_code`) if valid (**0–5**).
  3. Else **`LEG_UNASSIGNED`** (**255** / **`UNK`** / index **6**).
- Firmware copies resolved identity into runtime config; CAN **`NodeStatus`** and **`LegCommand` / `LegTelemetry`** IDs use **leg index** (`CAN_PROTOCOL.md`).

| `leg_id` | Switch | `leg_code` byte | `leg_index` | Bus | GUI label (not on bus) |
|--------|--------|-----------------|-------------|-----|----------------------|
| 1 | 1 | 0 | 0 | `can0` | FL |
| 2 | 2 | 1 | 1 | `can1` | FR |
| 3 | 3 | 2 | 2 | `can0` | ML |
| 4 | 4 | 3 | 3 | `can1` | MR |
| 5 | 5 | 4 | 4 | `can0` | RL |
| 6 | 6 | 5 | 5 | `can1` | RR |

- **Switch encoding:** each closed switch pulls its line so the pin reads **1**; **`NODE_ID0`** is bit 0 (LSB). Value **0** or **7** (invalid encoding) ⇒ treat as unassigned — rapid **`STATUS_LED`** blink, motors stay asleep until identity is valid.

- **Factory default:** `leg_code = 255` (**`LEG_UNASSIGNED`**). Board boots application, drives joints only after host provisioning; on CAN it emits **`NodeStatus`** with leg label **`UNK`** and index **`6`** (reserved — no `LegCommand` / `LegTelemetry` IDs until assigned). Joint calibration NV may be missing (`cal status = 0`).
- **Bench unknown board:** single node on one interface with **`LEG_UNASSIGNED`** is normal. Operator runs **`provision`** once; host must see exactly **one** unassigned node before writing (multi-node bench requires assigning by listed CAN interface + unique bus address / serial field in `NodeStatus`).
- **Host assignment (default path):** `krabby-firmware --stm provision --leg <code>` selects targets per rules below, sends **`ProvisionSet`** (DBC), board verifies CRC, writes NV **`leg_code`**, responds with **`NodeStatus`**, then resets into application using the new index. **OpenBLT application updates do not erase** NV (`FLASH.md`).
- **Robot install:** set switch to match mounting leg **before** power-on; no **`provision`** required when switch is valid.
- **Bench / rework:** leave switch invalid or **`UNK`** NV, then use host **`provision`** (same CAN-FD **`ProvisionSet`** flow).
- **Duplicate detection:** after discovery, if two nodes report the same leg code **other than `UNK`**, host **`show`** prints **`CONFLICT`** and **`provision`** refuses until the duplicate leaves the bus (power off, rewire, rework). Firmware never accepts a `ProvisionSet` that matches an in-use code already on the same CAN interface.
- **Tie to CAN IDs:** provisioned leg index **`i`** maps to **`LegCommand` `0x0C000000 + i`**, **`LegTelemetry` `0x0C000100 + i`**, **`NodeStatus` `0x0C000200 + i`**. **`board_ids.h` / `ids.py`** export the same table for firmware and Python (single enum).
- **HAL joint names:** after provision, firmware reports telemetry under **`{leg}_hip_yaw`**, **`{leg}_hip_pitch`**, **`{leg}_knee`** matching `RobotDefinition.mcu_joints` (`ROBOT_SENSORS.md`).

### `krabby-firmware --stm show` (operator flow)

1. Bring up the CAN interface (`scripts/stm/can_setup_orin.sh` on robot; **`can0`** on bench hardware; **`vcan0`** in sim — modes in `CAN_PROTOCOL.md`).
2. Run **`krabby-firmware --stm show`** (no args).
3. Host sends **`DiscoveryRequest`** (`0x0C0007F0`), listens **500 ms** on the active interface(s).
4. Print one row per **`NodeStatus`** received (see table below).
5. Exit code **0** when six provisioned legs are present on the robot; **non-zero** on missing leg, **`UNK`** on a robot bus, **CONFLICT** set.

### `krabby-firmware --stm provision` (operator flow)

1. Same interface bring-up as **`show`**.
2. Run **`krabby-firmware --stm provision --leg <code>`** with **`<code>`** ∈ **`FL`, `FR`, `ML`, `MR`, `RL`, `RR`**.
3. **Target selection:**
   - **Bench:** exactly one **`UNK`** node on the bus → target it.
   - **Robot:** target **`UNK`** if present; otherwise require **`--replace`** plus operator confirmation to overwrite an existing **`code`** (factory rework only).
4. Host sends **`ProvisionSet`** payload: desired leg byte, operator token, CRC.
5. On ACK, host re-runs discovery and prints updated **`show`** line for that node.

### `show` output columns

| Column | Source |
|--------|--------|
| `bus` | SocketCAN interface name (`can0`, `can1`, `vcan0`) |
| `leg` | `FL` … `RR`; **`UNK`** when `LEG_UNASSIGNED` |
| `index` | **0–5** provisioned; **6** unassigned |
| `firmware` | `NodeStatus` identity string |
| `mode` | `application` / `bootloader` |
| `cal` | NV cal status 0/1 |
| `last_seen_ms` | Host RX age |

## Reference

### `ProvisionSet` (host → one node)

| Field | Size | Value |
|-------|------|--------|
| CAN ID | — | **`0x0C0007E0`** (extended); unprovisioned nodes listen; provisioned nodes accept only when `--replace` session active |
| `new_leg_code` | 1 B | 0–5 per table above |
| `session_nonce` | 4 B | Host random; replay rejected |
| `crc32` | 4 B | Over preceding bytes |

DBC signal names follow `krabby_leg.dbc`; numeric CAN ID is normative here.

### NV leg byte persistence

| Event | `leg_code` in NV |
|-------|------------------|
| Factory flash | **255** (`LEG_UNASSIGNED`) |
| Successful `ProvisionSet` | 0–5 per leg |
| Application `update` over CAN | preserved |
| Bootloader update (`FLASH.md`) | preserved |
| Full chip erase (manufacturing) | **255** |

### CLI quick reference

| Command | Effect |
|---------|--------|
| `krabby-firmware --stm show` | Discovery table |
| `krabby-firmware --stm provision --leg FL` | Assign **`FL`** to selected **`UNK`** node |
| `krabby-firmware --stm provision --leg ML --replace` | Rework: rewrite **`ML`** on targeted node |

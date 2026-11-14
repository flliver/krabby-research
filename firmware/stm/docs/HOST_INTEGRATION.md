# Host integration (leg CAN)

## Purpose

Python package layout, **`krabby-firmware-stm`** CLI, and CAN SDK on the Orin or sim host.

## Components

- **`krabby-firmware`** PyPI package (`firmware/pyproject.toml`)
- Optional **`stm`** deps: `python-can`, `cantools`
- Console script **`krabby-firmware-stm`** → `firmware/stm/cli/`
- **`ids.py`**, **`codec.py`**, **`can_transport.py`** (SDK wiring in progress)
- Mega serial unchanged: `firmware/cli.py`, `firmware/krabby_mcu.py`

## Critical points

- **Two stacks:** default **USB Mega**; leg path via **`krabby-firmware-stm`** or **`krabby-firmware --stm`** when unified entry ships.
- **DBC:** `firmware/stm/dbc/krabby.dbc` only.
- **Connect:**
  - Robot: **`can0`** + **`can1`**, CAN-FD **1M/5M**, BRS.
  - Sim: **`vcan0`**, MTU **72**, `virtual=True`.
- **SDK (target):** `set_duty(leg, j1, j2, j3)` / `set_duties(18 values)` → one **`SET_DUTY`** per TX on **each** open bus; background thread decodes **`TELEMETRY_*`** / **`BOARD_INFO_*`**.
- **CLI (target):**
  - `duty <leg> <y> <h> <k>` — duties **−1…1**
  - `calibrate <leg> <joint>`
  - `telemetry`
  - `--virtual` — `vcan0` / sim boards
- **CLI still to wire:** `show`, host `provision`, `update` (OpenBLT), `sweep_18_joints` — see `FLASH.md`, `PROVISIONING.md`.

### `firmware/stm/` layout

| Module | Role |
|--------|------|
| `ids.py` | `leg_id`, CAN ID helpers |
| `codec.py` | cantools load `krabby.dbc` |
| `can_transport.py` | SocketCAN FD open |
| `cli/` | `krabby-firmware-stm` |
| `include/krabby/board_ids.h` | Firmware ID helpers |

### Environment

| Variable | Purpose |
|----------|---------|
| `KRABBY_MCU_STM` | `1` → CAN SDK in HAL/GUI when wired |
| `KRABBY_CAN_IFACES` | Robot: `can0,can1` |
| `KRABBY_CAN_IFACE` | Single iface / `vcan0` |

## Reference

```bash
make -C firmware stm-dbc
krabby-firmware-stm --virtual duty 1 0.5 0 0
```

HAL binding: `HAL_MCU_LAYER.md`. Orin link setup: `scripts/stm/can_setup_orin.sh` (when present).
# Leg controller documentation

## Purpose

Specs for six CAN-FD leg boards, host tools, and simulation.

## Components

- Leg-board firmware (STM32, three joints per board)
- **`../dbc/krabby.dbc`** and `ids.py` / `board_ids.h`
- **`krabby-firmware-stm`** CLI and CAN SDK (in progress)
- HAL MCU boundary (Jetson, Isaac, GUI)
- Virtual CAN (`vcan0`) for Isaac sim

## Index

| Path | Scope |
|------|--------|
| [IDENTITY.md](./IDENTITY.md) | Leg **1–6**, joint **1–3** (Y/H/K); HAL names at host edge only |
| [CAN_PROTOCOL.md](./CAN_PROTOCOL.md) | CAN-FD, bit rates, message IDs, dual bus |
| [LEG_BOARD_CONTROL.md](./LEG_BOARD_CONTROL.md) | Duty drive, telemetry, on-board calibration |
| [HARDWARE.md](./HARDWARE.md) | DRV8262, DAC7574, FDCAN2, node switch |
| [HOST_INTEGRATION.md](./HOST_INTEGRATION.md) | Package layout, `krabby-firmware-stm`, SDK |
| [HAL_MCU_LAYER.md](./HAL_MCU_LAYER.md) | HAL ↔ CAN SDK, `SET_DUTY`, `--stm` duty path |
| [VIRTUAL_CAN.md](./VIRTUAL_CAN.md) | `vcan0`, six sim boards, Isaac |
| [PROVISIONING.md](./PROVISIONING.md) | Node ID switch **1–6** |
| [ROBOT_SENSORS.md](./ROBOT_SENSORS.md) | Joint streams on CAN; Orin vision/IMU |
| [FLASH.md](./FLASH.md) | OpenBLT (board); Krabby `update` not in DBC yet |
| [DISPLAY.md](./DISPLAY.md) | Qwiic OLED — not in DBC |
| [PACK_MONITOR.md](./PACK_MONITOR.md) | Qwiic pack monitor — not in DBC |
| [`../dbc/krabby.dbc`](../dbc/krabby.dbc) | CAN-FD schema (`../dbc/krabby.sha256`) |
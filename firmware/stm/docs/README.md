# Leg controller documentation

## Purpose

Condensed specs for six CAN-FD leg controller boards, host tools, simulation, and flash.

## Components

- Leg-board firmware (STM32, three joints per board)
- CAN-FD bus protocol and `krabby_leg.dbc`
- Host `krabby-firmware --stm` and `firmware/stm/` Python modules
- HAL MCU boundary (Jetson, Isaac, GUI)
- OpenBLT flash over CAN
- Virtual CAN (`vcan0`) simulation
- Operator verification scenarios

## Index

| Path | Scope |
|------|--------|
| [CAN_PROTOCOL.md](./CAN_PROTOCOL.md) | CAN-FD, two buses, timing, discovery, leg-drop, Orin interfaces |
| [LEG_BOARD_CONTROL.md](./LEG_BOARD_CONTROL.md) | Three joints per board, jog/position, safe state, faults, NV, telemetry |
| [HARDWARE.md](./HARDWARE.md) | Leg controller PCB: DRV8262, DAC7574, FDCAN2, switches, schematic pin map |
| [ROBOT_SENSORS.md](./ROBOT_SENSORS.md) | Robot streams: joints on CAN, Orin vision/IMU, deferred pack power |
| [DISPLAY.md](./DISPLAY.md) | Optional Qwiic Krabby OLED: auto-detect, CAN bus mirror, render |
| [PROVISIONING.md](./PROVISIONING.md) | Node ID switch 1–6, NV fallback, CAN indices, show/provision CLI |
| [HOST_INTEGRATION.md](./HOST_INTEGRATION.md) | `krabby-firmware --stm` package layout, CLI, update, fleet/GUI hooks |
| [HAL_MCU_LAYER.md](./HAL_MCU_LAYER.md) | HAL ↔ firmware MCU facade, backends, Jetson/Isaac/GUI |
| [FLASH.md](./FLASH.md) | OpenBLT over CAN, memory map, session IDs, `update` / power-loss |
| [VIRTUAL_CAN.md](./VIRTUAL_CAN.md) | `vcan0`, six virtual leg nodes, Isaac `can_vcan`, DBC parity |
| [`../dbc/krabby_leg.dbc`](../dbc/krabby_leg.dbc) | CAN-FD message/signal schema (see `../dbc/README.md`, `krabby_leg.sha256`) |

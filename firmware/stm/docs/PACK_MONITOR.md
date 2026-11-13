# Qwiic pack monitor (leg board)

## Purpose

Optional **Qwiic INA228** cluster on **one** leg PCB measures pack and midpoint rails, then **publishes battery telemetry on CAN-FD** — same operator fields as Mega serial **`BATT`**, same install model as **Qwiic IMU** and **Krabby OLED**.

**Implementation:** port Arduino **`ina228_adapter` / `power_monitor`** to STM32 **I2C1** when pack CAN is added (not required for initial leg-bus joint bring-up). This file is the design contract.

## Components

- **Hardware:** up to two **INA228** on the Qwiic chain — pack **`0x40`**, midpoint **`0x41`** (shipping Mega map in [`firmware/sim/I2C_SIM_SCOPE.md`](../../sim/I2C_SIM_SCOPE.md))
- **Bus:** **I2C1** with **DAC7574** (`0x4C`), optional IMU, OLED, other Qwiic (`HARDWARE.md`)
- **Host parser:** `firmware/interfaces/battery_telemetry.py` (`BatteryTelemetry` — unchanged field set)
- **Arduino reference:** `firmware/arduino/arduino.ino` leader power monitor init + `appendBatteryTelemetry`; adapters under `firmware/arduino/src/power_monitor/`
- **Planned CAN message:** **`QwiicBattery`** — `0x0C000500 + leg_index` (`CAN_PROTOCOL.md`); **`krabby_leg.dbc`** entry added when firmware ships

## Critical points

- **Single install per robot:** at most **one** leg board hosts the Qwiic shunt/midpoint hardware. Other nodes do **not** emit **`QwiicBattery`** when probes fail (no spam with `valid=0`).
- **Auto-detect:** after **DAC7574** init, probe **`0x40`** / **`0x41`**. If pack INA ACKs, start periodic reads; midpoint optional but **both** addresses match Arduino when populated.
- **Forward on CAN:** hosting leg encodes the same logical fields as serial **`BATT`** (pack V/I/P, coulombs, A/B taps, divergence, region, per-monitor validity) into **`QwiicBattery`** at a modest rate (target **~10–20 Hz** — TBD in implementation; well below joint telemetry).
- **Host path:** `mcu_can` decodes into **`BatteryTelemetry`** for CLI, OLED mirror, and future HAL fuel-gauge hooks. Do **not** synthesize pack data on boards without INA hardware.
- **OLED interaction:** a board with **only** the display soaks **`QwiicBattery`** from the bus mirror for battery chrome (`DISPLAY.md`). A board with **both** OLED and INA may use local reads for the panel and still **forward** on CAN for the Orin.
- **Policy / estop:** region and validity bits are authoritative **only** from a live **`QwiicBattery`** with **`pack_valid`**; leg firmware must not guess pack state from **`VM`** actuator feed.
- **Failure isolation:** INA I2C faults disable **`QwiicBattery`** TX for that cycle; joint control and DAC programming continue.

## Reference

### Planned payload fields (align with `BatteryTelemetry`)

| Field | Notes |
|-------|--------|
| `pack_volts`, `pack_current_amperes`, `pack_power_watts`, `pack_charge_coulombs` | Pack INA228 |
| `battery_a_volts`, `battery_b_volts` | Midpoint INA228 |
| `divergence` | Same threshold semantics as Arduino `BATTERY_DIVERGENCE_THRESHOLD` |
| `pack_region` | `PackVoltageRegion` enum |
| `pack_valid`, `midpoint_valid` | Per-monitor read success |

### STM port notes

- Reuse shunt constants and register sequencing from **`ina228_constants.h`** / **`ina228_adapter.h`**.
- Run reads in the **1 kHz** loop or a dedicated slow tick; **never** block FDCAN TX slots used for **`LegTelemetry`**.
- Coulomb integration policy: match Arduino leader tick (`power_monitor` integration).

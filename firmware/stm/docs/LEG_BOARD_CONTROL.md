# Leg board joint control

## Purpose

Per-leg STM32 firmware drives three linear joints and reports state on CAN.

## Components

- **`joint_id` 1/2/3** (Y/H/K) — motors M1–M3 (`IDENTITY.md`)
- **DRV8262** bridges, **DAC7574** current limit (`HARDWARE.md`)
- Potentiometer + signed Hall per joint when routed
- **`krabby.dbc`:** `SET_DUTY`, `CALIBRATE`, `TELEMETRY_*`, `BOARD_INFO_*`
- Stall-based on-board **`calibration.c`** (joints 2–3; yaw ignored)
- Polar semantics from `firmware/arduino/actuator_manager.h` (duty sign, stall timing)

## Critical points

- **Control loop:** joint sampling at **1 kHz**; CAN handled in the same loop.
- **Duty command:** each joint duty **−1.0…1.0** from this leg’s bytes in **`SET_DUTY`**; negative retracts, positive extends; magnitude → PWM **0…1000** (`send_pwm`).
- **No host position mode:** normalized position appears in **telemetry** only (computed from cal); host does not send position setpoints on the bus.
- **No command timeout:** if the host stops sending, joints **keep last duty** (same as Mega jog hold).
- **Calibration:** host sends **`CALIBRATE`** with joint **2 or 3**; firmware runs extend/retract stall capture, NV save (`calibration.c`). Joint **1 (yaw)** ignores cal; **`position`** stays NaN.
- **Fault:** duty **0** on a faulted joint may **`clear_fault()`** via bridge sleep toggle (DRV8262).
- **Telemetry (50 ms):** per joint — raw hall, applied duty, pot (12-bit), fault / calibrating / calibrated bits, **`position`** 0…1 (or NaN). See `krabby.dbc` **`TELEMETRY_<leg_id>`**.
- **Board info (1 s):** **`BOARD_INFO_<leg_id>`** — semantic version + commit (`core/version.h`).
- **Startup:** **`M#_SLEEP` low** → DAC7574 → IWDG → read **`node_id`** switch → **`M#_SLEEP` high** → no motion until **`SET_DUTY`**.
- **Hall:** both edges on both hall lines; signed 32-bit total; not comparable to Mega counts (`OVERVIEW` FAQ).
- **Not in `krabby.dbc`:** Qwiic IMU, OLED, pack monitor.

## Reference

| `joint_id` | Letter | `joint_index` |
|-----------|--------|---------------|
| 1 | Y | 0 |
| 2 | H | 1 |
| 3 | K | 2 |

Drive polarity and profile details remain in `HARDWARE.md` and `firmware/stm/uno-v0.3/`.
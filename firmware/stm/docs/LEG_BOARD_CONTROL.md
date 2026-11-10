# Leg board joint control

## Purpose

Per-leg STM32 firmware closes the loop on three linear joints and exposes their state on CAN.

## Components

- Three joint channels per PCB: **`hip_yaw`**, **`hip_pitch`**, **`knee`** (HAL names prefixed by provisioned leg code)
- Motor bridges: **profile B (DRV8262)** — **`M#_PWM`** (EN), **`M#_DIR`** (PH), **`M#_SLEEP`**; **profile A (STSPIN9P2x)** dual PWM + EN when that silicon is populated (`HARDWARE.md`)
- **DAC7574** programs per-channel current limit before **`M#_SLEEP`** high
- Position feedback: potentiometer ADC per joint; optional Hall edge counter per joint when routed on the PCB (`HARDWARE.md`)
- Current sense ADC per joint (connection and stall evidence)
- `LegCommand` / `LegTelemetry` encoding (`CAN_PROTOCOL.md`, `../dbc/krabby_leg.dbc`)
- On-chip non-volatile storage for leg identity and joint travel limits
- Control port of `firmware/arduino/actuator_manager.h` / `LinearActuator` semantics (`firmware/arduino/`)

## Critical points

- **Joint set:** fixed order on the board — **`hip_yaw`** (index 0), **`hip_pitch`** (index 1), **`knee`** (index 2). HAL names are `{leg}_hip_yaw`, `{leg}_hip_pitch`, `{leg}_knee` with `{leg}` ∈ **`FL`, `FR`, `ML`, `MR`, `RL`, `RR`** from provisioning.
- **Control loop:** run joint `update()` at **1 kHz** (sensor filtering, position loop, ramping). CAN RX and command freshness run in the same loop. Publish **`LegTelemetry`** at **100 Hz** per `CAN_PROTOCOL.md`.
- **Position command:** normalized setpoint **0.0–1.0** maps linearly between stored **`minStop`** and **`maxStop`** raw pot limits for that joint. Activates position hold (`hasTarget = true`) on the joint. P gain, PWM ramp, and deadbands match the ported `ControlConfig` defaults unless tuned in firmware constants.
- **Jog command:** signed PWM **−255…255** per joint (`manualDrive`). Non-zero jog drives the bridge immediately (jog deadband bypass). Jog **0** stops motion on that joint but does **not** clear an active position target.
- **Command precedence (per joint):** if the latest non-stale `LegCommand` has **non-zero jog**, jog wins; else if position field is **valid** (host marks update), apply position target; else hold previous mode for that joint until stale.
- **Stale CAN / idle bus:** if no valid **`LegCommand`** for this node for **> 100 ms**, enter **safe state**: all joints `stopMotor()` — **EN low**, both PWM outputs zero, **`hasTarget = false`** on every joint. Resume normal control on the next fresh command without reboot.
- **Safe startup:** **`M#_SLEEP` low** through reset pull-downs → read leg identity → **program DAC7574** (I2C **`0x4C`**, channels A/B/C → motors 0/1/2; start ~**2 A** limit, tune in constants) → enable **IWDG** (~**100 ms**) → **`M#_SLEEP` high** → init joints, no motion until host **`LegCommand`**.
- **Drive rules (profile B):** positive jog/PWM = extend, negative = retract; magnitude maps to **`M#_PWM`** duty; **`M#_DIR`** selects direction. **Power 0** holds **`M#_PWM`** low (brake). Clear **`M#_FAULT`** by pulsing **`M#_SLEEP`** low when fault latched and command is zero (DRV8262 datasheet timing).
- **Drive rules (profile A):** never assert retract and extend PWM together; **EN** HIGH only while commanding non-zero PWM (STSPIN path).
- **Host command path (production):** Jetson HAL sends **open-loop jog** (radians → PWM) today; **`LegCommand` jog fields** on CAN-FD are normative. Normalized **position** and **autocal** are **optional** when NV cal is valid — Orin may own calibration before those firmware paths are enabled.
- **Pot validity:** exponential average + slew check + periodic open-circuit probe (ported `ActuatorPotTracker` logic). Invalid pot ⇒ normalized position **NaN** on the host path; joint **`connection_state = Disconnected`**.
- **Current sense:** smoothed ADC while driving; ported `ActuatorCurrentTracker` debounce. Evidence **CurrentAbsent** with valid pot ⇒ **Disconnected**; driving with **CurrentPresent** ⇒ **Connected**; otherwise **Unknown** until debounced.
- **Fault flags (per joint, in telemetry):** `pot_invalid`, `disconnected`, `calibration_missing`, `stall_timeout` (motion command active, position unchanged for **250 ms**), `bridge_disable` (driver fault line asserted when wired). Set `stall_timeout` during autocal only unless host policy promotes it to a held fault.
- **Hall:** encoder mode on **TIM2** / **TIM3** / **TIM5** — count **both edges** of **both** hall lines; **TIM3** is 16-bit — accumulate signed wrap into **32-bit** totals each tick; **motor 1** inverts count so extend increases count on all three joints. Hall totals are not comparable to legacy Mega serial counts.
- **Autocal (optional):** host raises **`autocal_start`** in `LegCommand` when the firmware build enables it; stall-based min/max per joint, NV write, **`CAL_IDLE`**. Early application images may ship **jog-only**; autocal may run from Orin policy or a later firmware enable.
- **Manual cal:** host sets `minStop`/`maxStop` via provisioning/CLI when position mode is active (`PROVISIONING.md`).
- **Host-facing telemetry (each joint, `LegTelemetry`):** `position_norm` (0–1 when pot valid; invalid sentinel when pot invalid), `pot_raw`, `current_raw`, `en_active`, `pwm_retract`, `pwm_extend`, `hall_edges`, `connection_state` (`Unknown` = 0, `Connected` = 1, `Disconnected` = 2), `faults` (bitmask above). Node aggregates three joints in DBC layout consistent with HAL order in `CAN_PROTOCOL.md`.

## Reference

### Board joint index

| Index | Joint on leg | Example HAL name (leg FL) |
|-------|----------------|---------------------------|
| 0 | `hip_yaw` | `FL_hip_yaw` |
| 1 | `hip_pitch` | `FL_hip_pitch` |
| 2 | `knee` | `FL_knee` |

### Default control tuning (initial port)

| Parameter | Value | Role |
|-----------|-------|------|
| `pwmRampStep` | 5 | PWM step per ramp tick |
| `rampIntervalMs` | 10 | ms between ramp steps |
| `pwmDeadband` | 20 | stop PWM below this when holding |
| `pwmErrDeadband` | 10 | raw pot error deadband in position mode |
| `Kp` | 2.0 | position loop gain |
| `alphaPot` | 0.15 | pot EMA |
| `alphaIS` | 0.10 | current EMA |

### Non-volatile layout (per leg board)

All regions below survive application update; bootloader must not erase them (`FLASH.md`).

| Offset (conceptual) | Size | Content |
|-------------------|------|---------|
| `0x0000` | 4 | Magic **`0x4B524242`** (“KRBB”) |
| `0x0004` | 1 | Layout schema version (**`1`**) |
| `0x0005` | 1 | Leg code enum: **0=FL, 1=FR, 2=ML, 3=MR, 4=RL, 5=RR** |
| `0x0006` | 1 | Cal status: **0=missing, 1=valid** |
| `0x0007` | 1 | Reserved |
| `0x0010` | 12 | **`minStop[3]`** int16, joints 0–2 |
| `0x001C` | 12 | **`maxStop[3]`** int16, joints 0–2 |
| `0x0028` | 16 | Reserved for future per-joint trim |

Factory default when NV header fails validation (bad magic, cal status not valid): **`minStop = 0`**, **`maxStop = 4095`** (12-bit ADC assumption; match STM32 ADC resolution in implementation).

### Autocal state sequence (this board only)

| Step | Joint index | Motion | On stall save |
|------|-------------|--------|----------------|
| 1 | 0 `hip_yaw` | retract | `minStop[0]` |
| 2 | 0 `hip_yaw` | extend | `maxStop[0]` |
| 3 | 1 `hip_pitch` | retract | `minStop[1]` |
| 4 | 1 `hip_pitch` | extend | `maxStop[1]` |
| 5 | 2 `knee` | retract | `minStop[2]` |
| 6 | 2 `knee` | extend | `maxStop[2]` |
| 7 | — | stop all, write NV, `CAL_IDLE` | — |

Autocal drive PWM magnitude: **150** for yaw-class joint (index 0), **200** for indices 1–2 (same stall timing **250 ms** as Mega autocal helper).

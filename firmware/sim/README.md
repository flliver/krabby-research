# Firmware <-> IsaacSim pin simulation

Host-side simulation binds Mega2560 drive pins to Isaac Lab articulation joints. Pin numbers for **REV 3** (`KRABBY_PIN_REV=3`, Krabby Uno v0.2) match `firmware/arduino/board_pins.h` and the front-board wiring in `firmware/SETUP.md` section 1.

## FRONT board - six slots (`ACT_LIST_FRONT`)

When the MCU role is **FRONT** (or UNKNOWN with front actuators), `arduino.ino` drives FL + FR legs on slots S0-S5:

| Slot | PWM_R | PWM_L | EN (Rev 3) | POT | IS | Firmware ID | Leg segment | Isaac USD joint (`find_joints` name) |
|------|-------|-------|------------|-----|----|-------------|-------------|--------------------------------------|
| S0 | D2 | D3 | D22 | A0 | A6 | `FLHY` | FL hip yaw | `FL_Body_Hip_RevoluteJoint` |
| S1 | D4 | D5 | D24 | A1 | A7 | `FLHL` | FL hip pitch | `FL_Hip_Femur_RevoluteJoint` |
| S2 | D6 | D7 | D26 | A2 | A8 | `FLKL` | FL knee | `FL_Femur_Tibia_RevoluteJoint` |
| S3 | D8 | D9 | D23 | A3 | A9 | `FRHY` | FR hip yaw | `FR_Body_Hip_RevoluteJoint` |
| S4 | D10 | D11 | D25 | A4 | A10 | `FRHL` | FR hip pitch | `FR_Hip_Femur_RevoluteJoint` |
| S5 | D12 | D13 | D27 | A5 | A11 | `FRKL` | FR knee | `FR_Femur_Tibia_RevoluteJoint` |

**Order:** Same as `LinearActuator* ACT_LIST_FRONT[]` in `firmware/arduino/arduino.ino` (FL yaw -> knee, then FR yaw -> knee).

**Isaac resolution:** Prefer exact USD names above (see `assets/crab_simple.usda`). Per-leg regex examples: `FL_.*_RevoluteJoint` returns three joints; exclude `*_Body_CamShaft_*` if a broad pattern is used - the table names are the driven DOFs aligned with `hal/server/robot_definition_krabby_hex.py` `mcu_joints` for FL/FR.

**PWM semantics:** `LinearActuator::driveActuator` - extend = `PWM_R`, retract = `PWM_L`, motion gated by `EN` HIGH (`firmware/arduino/actuator_manager.h`).

**Programmatic config:** `SimulatedRobotConfig.front_board_rev3()` in `simulated_robot_config.py` encodes this table for sim and unit tests.

## Pin state (`pin_state.py`)

The host Arduino shim calls `PinStateModel.pin_mode`, `digital_write`, and `analog_write` to record Mega pin state. Each pin stores mode (`INPUT` / `OUTPUT`), last digital level (`LOW` / `HIGH`), and PWM duty (0-255). Actuator code reads `signed_pwm_command(PWM_R, PWM_L, EN)` as `(EN high) * (PWM_R - PWM_L)`, matching `LinearActuator::driveActuator`.

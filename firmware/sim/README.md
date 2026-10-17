# Firmware <-> IsaacSim pin simulation

Architecture (flat stack, host vs AVR builds): [ARCHITECTURE.md](ARCHITECTURE.md).

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

**PWM / drive:** `LinearActuator::driveActuator` - extend = `PWM_R`, retract = `PWM_L`, motion gated by `EN` HIGH. Signed motor command: `(EN high) * (PWM_R - PWM_L)` (`firmware/arduino/actuator_manager.h`).

Multi-board role election on the host sim is documented in [ROLE_BOOT.md](ROLE_BOOT.md). Optional single-leader 18-joint SDK shortcut: `KRABBY_SIM_SINGLE_LEADER_18=1` (see ROLE_BOOT.md).

I2C sim scope (shipping LSM6DSO + SSD1306, INA228 blockers): [I2C_SIM_SCOPE.md](I2C_SIM_SCOPE.md).

## Virtual serial (SDK / GUI on sim USB)

Linux/macOS use a PTY; Windows uses a localhost `socket://` TCP bridge or an optional com0com pair. See [VIRTUAL_SERIAL.md](VIRTUAL_SERIAL.md). Override with `KRABBY_MCU_PORT`.

## Core layout

- `simulated_robot_config.py` - FRONT-board pin and Isaac joint table in code.
- `robot_core.py` - the loop: firmware step, pin readback, joint targets, sim advance.
- `host/` - host `Arduino.h`, shipping `actuator_manager.h` build, `libkrabby_host`, and `host_bridge.py`.
- [ROLE_BOOT.md](ROLE_BOOT.md) - three-board SYNC election, EEPROM hints, and host APIs.

Build and test commands live in `firmware/Makefile` (`host-actuator`, `test-native`, `compile-firmware`).

## Isaac integration tests

Isaac Sim tests are opt-in (`RUN_FIRMWARE_ISAAC=1`, `@pytest.mark.isaacsim`). Run them in the **`krabby-isaacsim`** Docker image (`make build-isaacsim-image` from repo root), not on host Python unless you have a full Isaac Lab + Sim install:

```bash
./images/isaacsim/run_firmware_isaac_pytest.sh
# or: make -C firmware test-firmware-isaac
```

Pass/fail comes from `build/firmware-isaac-pytest/outcomes.jsonl` (see [images/isaacsim/README.md](../../images/isaacsim/README.md) — Firmware simulation tests).

GUI **Cur** column over sim MCU: [firmware/gui/CURRENT_SENSE_SIM_DEMO.md](../gui/CURRENT_SENSE_SIM_DEMO.md) (`run_gui_sim_demo` + `python -m firmware.gui`).

## Current-sense tuning (manual)

Tune wrench → `pinIS` breakpoints in `current_sense_mapping.py` (`CurrentSenseBreakpoints`: ADC levels 0 / ~2 / ~4 / ~30, force thresholds, and optional `effort_nm_to_load_n` cross-check against `applied_torque` / `ParkourDCMotor.applied_effort`). Run a **windowed** single-env viewer from the repo root (Isaac Lab + GPU; omit `--headless` on `AppLauncher` args to see the scene):

```bash
make -C firmware host-actuator
cd /path/to/krabby-research
python -m firmware.sim.wrench_tuning_viewer
# optional: --actuator FLHL --jog-pwm 200 --device cuda:0
```

Focus the terminal for keys: **e** / **r** jog extend/retract, **space** coast, **q** quit. Each printed line shows effective load (N) with wrench vs effort breakdown when both are available, mapped ADC, host `analogRead` shim value, joint velocity, and EN — adjust constants until idle / swing / contact / lift match ~0, ~2, ~4, and ~30.

## Hall encoder scale (`saf`)

Simulated hall edges per stop-to-stop travel live in `hall_encoder_scale.json` (loaded by `hall_encoder.py` and mirrored in `host/hall_encoder_scale.hpp`). Until a physical stop-to-stop measurement is committed, the file keeps the documented placeholder **3200**. After measuring `saf` on real hardware (one joint, mechanical min to max), update `edges_per_full_travel` in the JSON and set the same value on `kHallEdgesPerFullTravel` in `host/hall_encoder_scale.hpp`.

Override at runtime without editing JSON: `KRABBY_HALL_EDGES_PER_FULL_TRAVEL=<n>`.

## Fleet + Isaac fullsim

Fleet teleop on a GPU host uses Isaac HAL **fullsim** separately from this pin/firmware sim
tree. Procedure and known gaps: [`fleet/FULLSIM-FLEET-TELEOP-SMOKE.md`](../../fleet/FULLSIM-FLEET-TELEOP-SMOKE.md).

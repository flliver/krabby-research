# Jetson HAL bench session — motor failure investigation

*Date: 2026-05-26*

## Summary

During an end-to-end gamepad-control test of PR #4 on the Jetson, two leg motors became unresponsive — one stuck fully extended, one stuck fully retracted. The motors are on different Krabby boards. Neither returned to neutral after a 12V power cycle followed by commanding `0.5` on all 18 joints. Software-side investigation finds a configuration root cause that fully explains the failure mode, plus three additional firmware-level issues surfaced along the way.

## Test setup

- Image: `public.ecr.aws/t7t7b3i3/krabby-locomotion:mainline-latest`
- New `hal/server/jetson/main.py` from PR #4 branch `m14-gamepad-consolidation` bind-mounted over the installed module (`-v /tmp/main.py:/usr/local/lib/python3.12/dist-packages/hal/server/jetson/main.py:ro`)
- Mode: `krabby-hal-server-jetson --control-source gamepad` (HAL bound to `tcp://*:6001` / `tcp://*:6002`)
- Robot definition: `KRABBY_HEX_DEFINITION` (18 joints, MCU-controlled)
- Boards reported by `krabby firmware show`:
  - `/dev/ttyACM0`  primary: `0.2.9 (release/0.2.9 05e5e3e)`
  - `/dev/ttyUSB0`  left:    `0.2.9 (release/0.2.9 05e5e3e)`
  - `/dev/ttyUSB1`  right:   `0.2.9 (release/0.2.9 05e5e3e)`
- Controller: Nintendo Switch Pro Controller, USB. SDL2 detected it cleanly; `/dev/input/js0`, `/dev/input/event1` both present.
- Pre-test state: only two motors had been previously verified working; the third was already non-functional (pre-existing). Other 15 motors not exercised today.

## Failure timeline

1. Server started in gamepad mode. Logs confirm:
   - `HAL server initialized: observation=tcp://*:6001, command=tcp://*:6002`
   - `KrabbyMCUSDK initialized for JetsonHalServer`
   - `[MCU] Krabby Ready PINS_REV2_UNO_V01. FLHY`
   - 100 Hz production loop entered.
2. `krabby uno` connected via `--network=container:krabby`. SDL2 picked up the controller. Commands flowed at ~50 Hz from uno → HAL server → MCU.
3. Joystick exercise. Non-neutral values observed for multiple joints, e.g. `FLHY=0.5127, FLHL=0.2000, FLKL=0.4093` and `FLHL=0.4826, FLKL=0.7991`. Within the mapper's ±0.3 rad clamp.
4. During the session, one joint reached fully extended and stopped responding; a second joint (different board) reached fully retracted, same.
5. Stopped server, stopped uno.
6. **Power-cycled 12V supply** (off ~60 s, then back on).
7. Restarted server with no stick input. `krabby uno` sent neutral (`0.5000`) to all 18 joints; logs confirm `KrabbyMCUSDK: Applied joint command (...) FLHY=0.5000, FLHL=0.5000, ... RRKL=0.5000`.
8. Two affected motors did **not** move toward neutral. They remained pinned at their respective extremes.

This rules out a recoverable BTS7960 over-current/thermal trip (those clear when 12V is removed).

## Confirmed software-pipeline health

- All three boards enumerated and addressable.
- Serial comms working (joint commands logged at the MCU SDK layer match what the host sends).
- MCU reports ready and accepts commands.
- 100 Hz main loop runs (some 12-13 ms frame-time warnings; pre-existing, unrelated).
- End-to-end command path verified working with the joints that *did* respond during the session.

So the fault is not "commands aren't reaching the boards."

## Root cause (confirmed)

Raw serial capture of MCU boot output (read by toggling DTR on each port and dumping the first ~4 s; full transcript in Appendix A) shows **identical lines on all three boards:**

```
--- SYNC ---
ROLE_HINT: FRONT      (or LEFT or RIGHT)
ROLE: UNKNOWN (front actuators)
No EEPROM calibration found. Using defaults.
Krabby Ready PINS_REV2_UNO_V01. FLHY
```

`No EEPROM calibration found. Using defaults.` is emitted by `ActuatorManager::loadCalibration()` (`firmware/arduino/actuator_manager.h:531-548`) when the EEPROM magic does not match `0xDEADBEEF`. When this happens, every `LinearActuator` retains its initialized defaults:

```cpp
int minStop = 0;
int maxStop = 1023;   // firmware/arduino/actuator_manager.h:37-38
```

i.e. the full ADC range, not the joint's actual mechanical range.

The closed-loop control in `LinearActuator::update()` (same file, lines 131-165) is:

```cpp
currentTarget = minStop + (int)(val * (maxStop - minStop));   // setTarget(), line 102
...
int error = currentTarget - getRawPos();
int desiredPwm = (int)(error * controlConfig.Kp);             // Kp = 2.0, line 17
desiredPwm = constrain(desiredPwm, -255, 255);
```

`update()` has **no stall detection.** `isStalled(timeout)` exists (lines 169-190) but is only invoked inside `updateCalibration()` (lines 371-516) — the calibration state machine. In normal operation, a position-error that can never close (because the pot can't reach the commanded ADC value) leaves `desiredPwm` saturated indefinitely.

So with `minStop=0, maxStop=1023` defaults:
- Stick at, say, hip-pitch full deflection (the mapper's −LY axis, scaled by `DEFAULT_HIP_UP_DOWN_SCALE = 0.3` rad in `controller/mappers/gamepad_to_krabby_hal_mapper.py:44`) → mapper emits ±0.3 rad.
- MCU SDK normalization (`hal/server/jetson/krabby_mcusdk.py:36`): `n = max(0.0, min(1.0, (r / JOINT_LIMIT_RAD) * 0.5 + JOINT_NEUTRAL))`. With `JOINT_LIMIT_RAD = JOINT_NEUTRAL = 0.5` and `r = +0.3`, `n ≈ 0.8`.
- Firmware: `currentTarget = 0 + 0.8 × 1023 ≈ 820` ADC counts.
- If the mechanical end-stop limits pot travel to (e.g.) 700, `error = 120`, `desiredPwm = 240`, applied continuously to a stalled motor.

This matches the observed failure: motor / H-bridge cooks over seconds-to-minutes against a hard stop, with no firmware-side cutoff.

## Additional firmware-level issues surfaced

These are independent of the immediate motor failure but came up during the investigation. Listing them here because they all affect what a recovery plan should do.

### 1. `CalData::magic` is a 16-bit type holding a 32-bit literal

`actuator_manager.h:356-362`:

```cpp
struct CalData
{
    int minVals[6];
    int maxVals[6];
    int magic; // 0xDEADBEEF to check validity
};
```

Build target is `arduino:avr:mega` (`firmware/Makefile:13`) — ATmega2560, AVR architecture, **`int` is 16-bit**. The literal `0xDEADBEEF` (32-bit `unsigned long`) silently narrows on assignment to `0xBEEF`. On read-back, the comparison `if (data.magic == 0xDEADBEEF)` (line 535) compares the sign-extended 16-bit value (`0xFFFFBEEF` after the usual arithmetic conversions) against `0xDEADBEEF` — never equal.

Net effect: **even if calibration is successfully run, `loadCalibration()` will reject the data on the next boot and fall back to defaults.** So we can't distinguish "calibration was never run" from "calibration was run but the magic check always fails."

Fix is one line: change `int magic;` to `uint32_t magic;`. The deliberate 26-byte size of `CalData` (per the offset-32 layout comment in `arduino.ino:25-28`) would grow to 28 bytes; the role bytes at offset 32-33 still don't overlap. Or change to a 16-bit sentinel: `int magic; ... data.magic = 0x1234;`.

### 2. `ROLE: UNKNOWN` on all three boards, despite valid `ROLE_HINT:` cache

The boot transcripts show each board correctly reading its cached role from EEPROM offset 32-33 and printing it as `ROLE_HINT:`, but then falling through to `ROLE: UNKNOWN (front actuators)`. Cause: `loadRole()` (`arduino.ino:36-44`) is called inside `determineRole()` *only to print the hint* (`arduino.ino:155-161`). It does **not** set `currentRole`. Live role is always re-elected by the SYNC_TOKEN handshake over Serial1/Serial2 (`arduino.ino:171-225`). If that handshake times out — 3 second window — the board defaults to `ROLE_UNKNOWN` (line 228), regardless of the cached value.

For this to time out on all three boards simultaneously, the inter-board Serial1/Serial2 cabling between primary and the two followers would need to be missing or non-functional. Worth checking the physical inter-board harness.

Note: the role-storage commit (`05e5e3e M14 Task 1 Step 9`) itself works as designed — the EEPROM bytes 32-33 do hold the previously-saved role, which is exactly what's being printed as `ROLE_HINT:`. The gap is that the cache isn't used as a fallback when the live election fails. A small change would close this: in the fall-through path at `arduino.ino:228`, set `currentRole = loadRole()` instead of unconditionally `ROLE_UNKNOWN`, and pick `actuatorManager` accordingly.

### 3. Practical impact of `ROLE_UNKNOWN` on command routing

When `currentRole == ROLE_UNKNOWN`, the code at `arduino.ino:228-233` defaults `actuatorManager = new ActuatorManager(ACT_LIST_FRONT, ...)` and `mainSerial = &Serial` (the USB port) on **every board**. That means each of the three boards thinks it owns the FL/FR joints (`FLHY/FLHL/FLKL/FRHY/FRHL/FRKL`) and accepts T-commands for those joint names over USB.

Each board therefore executes the same T-command on its own physically wired-to-pins motors. Whether this is harmful depends on which physical motors are wired to those pins on each of the three boards — but it does mean a single `T FLHY 0.8` command from the host is being processed and applied by three boards in parallel, each driving whatever is wired to that board's `PIN_S0_*` pins. If two boards have working motors at the same logical-joint-name's pin slot, two motors get the same command per stick deflection.

### 4. No headless calibration command

`startAutoCalibration()` is reachable in three ways from the host:
- `firmware/krabby_mcu.py:239-242` — `send_command_calibrate()` writes `b"C\n"` over serial.
- `firmware/__main__.py:121,140-143` — interactive menu, key `9`.
- Raw serial: send `C\n` to `/dev/ttyACM0` directly.

`krabby firmware` (no subcommand) drops into the interactive menu, but `firmware_cmd` in `krabby/_docker.py` doesn't pass `-it` to `docker run`, so it isn't usable headless from SSH today. A `krabby firmware calibrate` subcommand (or an `-it` flag on the docker invocation) would close this gap.

## What was ruled out

- **Single-board hardware fault**: failures are on different boards.
- **Recoverable H-bridge trip**: 12V power cycle did not restore response.
- **Command-pipeline software bug**: commands reach the MCU and are applied (logs confirm).
- **Mapper saturation**: mapper *is* clamped at ±0.3 rad (`hip_up_down_scale=0.3`, `knee_out_in_scale=0.3`, `hip_yaw_scale=0.2` in `controller/mappers/gamepad_to_krabby_hal_mapper.py:44-46`). The MCU SDK clamps further to `[0.0, 1.0]`. The mapping is bounded; the issue is downstream of the bound.
- **EEPROM region collision between role storage and calibration**: deliberately disjoint per the layout comment in `arduino.ino:25-28` (calibration 0-25, role 32-33). The role-storage change did not corrupt calibration data.

## Open questions (can't be resolved from software alone)

1. **Which component failed** in each of the two stuck joints — motor windings, H-bridge MOSFETs, or wiring? Diagnosis needs a multimeter: continuity across motor terminals (winding integrity) and voltage at the H-bridge output while commanding non-neutral.
2. **Whether the joystick session is what caused the damage**, vs the motors being marginal pre-test. The motors were known-working pre-session, but only those two had been exercised at all today.
3. **Inter-board Serial1/Serial2 harness**: live role election timing out on all three boards strongly suggests the leader-to-followers serial cabling isn't carrying SYNC_TOKEN. Worth a continuity check.

## Suggested follow-ups (in priority order)

1. **Inspect the two failed joints with a multimeter** to confirm whether the damaged component is motor or H-bridge, before deciding on replacement parts. Calibration would re-energize whichever path is damaged and could complete the failure on a marginal H-bridge.
2. **Fix `CalData::magic` to 32-bit (`uint32_t`)**. Without this, calibration can never persist regardless of how often `C\n` is sent — `loadCalibration()` will always fall back to defaults on next boot. One-line change.
3. **Add stall protection to `LinearActuator::update()`** in firmware. The same logic that `isStalled()` already implements (no position change > 2 ADC for ≥250 ms while PWM applied) could de-energize the motor and clear `hasTarget` in normal operation, not just during calibration. Optionally use `avgIS` (current sense, already smoothed) as an additional cutoff threshold. This would prevent the failure class even if calibration limits are stale or wrong.
4. **Either fix live role election (check Serial1/Serial2 harness) or fall back to `loadRole()` in `determineRole()`** at line 228 when the live handshake times out. Today the cached role is read only for hint-printing, not as a fallback.
5. **Run auto-calibration** *after* #1, #2, and ideally #3. The mechanism exists (`C\n` to primary, broadcasts to followers via `arduino.ino:347-353`). State machine drives each joint until stalled, saves min/max per joint to EEPROM. Also add a headless `krabby firmware calibrate` subcommand or pass `-it` on the firmware docker invocation.

## Appendix A — raw MCU boot transcripts

Captured with a small Python helper that toggles DTR on each `/dev/tty*` and reads ~4 s of output. Pot/current-sense readings are non-zero on most joints, confirming the boards see their analog inputs.

```
===== /dev/ttyACM0 =====
--- SYNC ---
ROLE_HINT: FRONT
ROLE: UNKNOWN (front actuators)
No EEPROM calibration found. Using defaults.
Krabby Ready PINS_REV2_UNO_V01. FLHY
UNKWN; FLHY 0.000 0 0 0 0 0 0 0;FLHL 0.000 0 0 0 0 0 0 0;FLKL 0.000 0 0 0 0 0 0 0;
       FRHY 0.182 186 215 0 0 0 0 0;FRHL 0.199 204 211 0 0 0 0 0;FRKL 0.189 193 215 0 0 0 0 0

===== /dev/ttyUSB0 =====
--- SYNC ---
ROLE_HINT: LEFT
ROLE: UNKNOWN (front actuators)
No EEPROM calibration found. Using defaults.
Krabby Ready PINS_REV2_UNO_V01. FLHY
UNKWN; FLHY 0.239 244 205 0 0 0 0 0;FLHL 0.203 208 195 0 0 0 0 0;FLKL 0.197 202 200 0 0 0 0 0;
       FRHY 0.210 215 197 0 0 0 0 0;FRHL 0.198 203 202 0 0 0 0 0;FRKL 0.197 202 197 0 0 0 0 0

===== /dev/ttyUSB1 =====
--- SYNC ---
ROLE_HINT: RIGHT
ROLE: UNKNOWN (front actuators)
No EEPROM calibration found. Using defaults.
Krabby Ready PINS_REV2_UNO_V01. FLHY
UNKWN; FLHY 0.212 217 168 0 0 0 0 0;FLHL 0.196 201 175 0 0 0 0 0;FLKL 0.169 173 176 0 0 0 0 0;
       FRHY 0.178 182 170 0 0 0 0 0;FRHL 0.173 177 179 0 0 0 0 0;FRKL 0.161 165 160 0 0 0 0 0
```

Note the `FLHY/FLHL/FLKL=0.000` triple on ACM0 (primary) — those three look unwired/floating at the pot, in contrast to the other 15 which report non-zero positions. May be incidental or may be relevant; flagging.

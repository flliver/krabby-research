# Krabby-Uno Task 2: Six-Axis Leg Controller

## Overview

This firmware drives a full leg pair (Left & Right) consisting of **6 Motors**.

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560
  - **6x** BTS7960 43A H-Bridge Drivers
  - **12x** Resistors (10kΩ) for Current Sense protection
  - 12V Power Supply
- **Software:**
  - Python 3
  - Libraries: `pip install pyserial` (the interactive menu uses the stdlib `termios`/`select`, so it works headless over SSH — no `keyboard`/`pynput`/X11 needed)
  - Arduino IDE

---

## 1. Hardware Wiring (Rev 3 — Krabby Uno v0.2)

**Polarity Note:**
* **RPWM / R_EN** = Right (Extend/Forward).
* **LPWM / L_EN** = Left (Retract/Reverse).

| Board     | Joint         | PWM (R, L)       | EN    | Pot / Hall B  | Current Sense | HallA  |
| :-------- | :------------ | :--------------- | :---- | :------------ | :------------ | :----- |
| **FL**    | Yaw (LHY)     | D2, D3           | D22   | A0            | A6            | D50    |
|           | Hip (LHL)     | D4, D5           | D24   | A1            | A7            | D51    |
|           | Knee (LKL)    | D6, D7           | D26   | A2            | A8            | D52    |
| **FR**    | Yaw (RHY)     | D8, D9           | D23   | A3            | A9            | A12    |
|           | Hip (RHL)     | D10, D11         | D25   | A4            | A10           | A13    |
|           | Knee (RKL)    | D12, D13         | D27   | A5            | A11           | A14    |

**Sensors on this robot:** the hips and knees are **hall-encoder actuators** (quadrature: hall A on the HallA pin, hall B on the analog pin in the "Pot / Hall B" column). The analog pins A0-A5 only act as a potentiometer input on an actuator that has a pot. Each joint's kind is auto-detected during calibration (see Feature 1). The hall inputs are connected on hips and knees; the yaws are not calibrated.

**Note:** Ensure all Enable (EN) pins are connected and driven HIGH when driving, otherwise calibration will get 'lost' as it will not know where joint positions are.

---

## 2. Installation

### 2.1 Serial RX buffer (leader board, 3-board setup)

When using the **leader** board that forwards telemetry from left/right followers, a small serial RX buffer can overflow and drop bytes (corrupt or missing actuators in telemetry, "can't keep up" on the host). The leader needs a **256-byte** RX buffer for Serial1/Serial2 so it can hold a full ~200-byte forwarded line from each follower while it services USB and the actuator update.

The Makefile passes this define on every build, so you usually don't have to do anything. `make compile-firmware` / `make upload-firmware` bake `-DSERIAL_RX_BUFFER_SIZE=256` into the `arduino-cli compile` invocation unconditionally (see `firmware/Makefile` `BUILD_PROPS`), exactly as CI (`.github/workflows/publish-firmware.yml`) does. `firmware/install.py`'s `platform.local.txt` write is now a **belt-and-suspenders backup for IDE builds, not a requirement** — a `make`-built or CI-built binary already has the 256-byte buffer regardless of whether `install.py` ran or which AVR core version is installed. (Note: IDE builds also need the fetched-library symlink from the "Fetched libraries" section below, or they won't find the LSM6DSO header.) (Some core versions, e.g. 1.8.7, already default the Mega's RX buffer to 256; passing the define guarantees it on every core version and board variant.)

The manual edits below are only needed if you build the sketch **directly from the Arduino IDE** without the `platform.local.txt` override.

**You do not flash the core separately.** The Arduino “core” is just C++ source that is compiled *with* your sketch into a single firmware image. Change the buffer size, then build and upload as usual.

**Arduino IDE**

- **Option A – One-time edit (survives until you update the AVR board package):**  
  Open the core file (path similar to):
  - Windows: `%LOCALAPPDATA%\Arduino15\packages\arduino\hardware\avr\1.8.7\cores\arduino\HardwareSerial.h`
  - macOS: `~/Library/Arduino15/packages/arduino/hardware/avr/1.8.7/cores/arduino/HardwareSerial.h`  
  Find the block that sets `SERIAL_RX_BUFFER_SIZE` (e.g. `#define SERIAL_RX_BUFFER_SIZE 64`) and change **64** to **256**. Save. Then compile and upload your sketch as usual.

- **Option B – Build flag via platform override:**  
  In the same `avr` package folder (e.g. `.../packages/arduino/hardware/avr/1.8.7/`), create or edit `platform.local.txt` and add:
  ```text
  compiler.c.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256
  compiler.cpp.extra_flags=-DSERIAL_RX_BUFFER_SIZE=256
  ```
  so the define is applied when the core and your sketch are compiled. Then build/upload as usual.

**PlatformIO**

In `platformio.ini` for the board that acts as the leader, add:

```ini
build_flags = -DSERIAL_RX_BUFFER_SIZE=256
```

Then build and upload. No core file edit needed.

**Follower-only boards** do not need this change; only the board that runs `forwardFullLines` (the leader on USB) benefits from the larger buffer.

### 2.2 Telemetry format (wire protocol)

Telemetry is sent as **newline-terminated lines** over serial. The Python side parses each line into a **dict of joint id → values** using `JointTelemetry` in `interfaces/joint_telemetry.py`.

- **Line format:** `<ROLE>; <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf>; <name> ...; ...`
- **Role prefix:** One of `FRONT`, `UNKNOWN`, `LEFT`, `RIGHT` (no semicolon inside the role).
- **Segment format:** Each joint segment is 10 space-separated values: joint name, position (0–1, or `nan` when the joint has no position), pot raw (on a hall actuator this is the hall-B analog level, not a position), current raw, enable L/R, PWM L/R, the **signed hall count** (field 9), and composed connection state (`0` unknown, `1` connected, `2` disconnected). The parser still accepts legacy 9-field segments.
- **Field 9 (hall count):** on Rev 3 this is the signed quadrature count of the joint's hall slot (x2, both edges of A). On hips and knees **retract counts up** and extend counts down; the FRHY yaw counts the other way. It is the raw count since power-up, not re-zeroed at a home, so it resets at every boot. Position control uses the homed frame instead (see Feature 1). On Rev 1 the count has no direction; Rev 2 has no halls.
- **Example:** `FRONT; FLHY 0.723 740 694 0 0 0 0 0 1;FLHL 0.723 740 691 ...`

On the Arduino side, telemetry is built in **telemetry_manager.h** (struct `JointTelemetry`, `appendTo()`). The old standalone `joint_telemetry.h` was removed; all telemetry formatting and collection lives in `telemetry_manager.h` and `actuator_manager.h`.

### 2.3 Pin revisions (`KRABBY_PIN_REV`)

Wiring is selected at **compile time** in **`arduino/board_pins.h`** (`#define KRABBY_PIN_REV`, default **3**). Rev **3** matches **`MOTOR_HEADER_PINOUT.md`**.

| | **Rev 3** (default, Uno v0.2) | **Rev 2** (Uno v0.1) | **Rev 1** (original) |
|---|---|---|---|
| PWM | D2-D13 | D2-D13 | D2-D13 |
| FL EN (LHY / LHL / LKL) | D22 / D24 / D26 | D22 / D23 / D24 | D22 / D23 / D24 |
| FR EN (RHY / RHL / RKL) | D23 / D25 / D27 | D28 / D26 / D27 | D28 / D26 / D27 |
| HallA1-6 | D50, D51, D52, A12, A13, A14 (PCINT0+2) | none | D37, D36, D35, D32, D33, D34 (PCINT1) |

- **Arduino IDE:** open **`firmware/arduino/arduino.ino`**, set **Board → Arduino Mega 2560**, choose the correct **Port**, set **`KRABBY_PIN_REV`** in **`board_pins.h`** if needed, then **Upload**. The serial monitor at **250000** baud (`BAUD_RATE` in `arduino.ino`) should show **`PINS_REV3_UNO_V02`** (or the matching label) after reset.
- **Make + arduino-cli:** install [arduino-cli](https://arduino.github.io/arduino-cli/latest/installation/) and **GNU Make**. On Windows: `winget install GnuWin32.Make` then add **`C:\Program Files (x86)\GnuWin32\bin`** to your **`PATH`**. Put **arduino-cli** on your **`PATH`** (or set **`ARDUINO_CLI`**). Install **pyserial** for port auto-detect: `pip install -r firmware/requirements.txt`. From **`krabby-research`**:
  - `make -C firmware upload-firmware` — auto-detects serial device via **`firmware/mcu_port.default_target()`** (prints path from **`firmware/scripts/detect_mcu_port.py`**). Pass **`PORT=COM5`** (or `/dev/ttyACM0`) to override Makefile upload only.
  - Other revisions: `make -C firmware upload-firmware PIN_REV=1` (or `PIN_REV=2`).
  - Compile only: `make -C firmware compile-firmware`.
  - See **`firmware/Makefile`** for **`ARDUINO_CLI`**, **`FQBN`**, **`PIN_REV`**.

Flash each Mega with the image that matches **that** board’s wiring. All three boards use the same sketch; role is elected at runtime. **Flash all three boards together** from the same build: the hall-encoder calibration and the v2 EEPROM record need every board to run it, and an old follower would run the pot-only routine on hall joints. A Rev 3 leader refuses to start a calibration while a linked follower has sent no `SENS` line since the leader booted (`CAL ROBOT FAIL <role>: no SENS: ...`, see Feature 1).

Today, **LEFT** and **RIGHT** must each have their **own USB** cable to the host (or hub) when you flash them — same as FRONT. Runtime control still only needs Orin USB to FRONT (followers use UART). **Planned:** a future update path will flash LEFT/RIGHT through the leader so updating FRONT also updates the followers, without plugging USB into LEFT/RIGHT for that step.

#### Board vanishes during or after upload (port gone)

An interrupted upload (Ctrl+C), a flaky hub, or a bad cable can drop USB enumeration so the Mega looks bricked even when the sketch is fine.

**Symptoms**

- `cannot open port` / avrdude `error -110` / serial `Input/output error`
- `krabby firmware show` reports **No attached Mega**
- No `/dev/ttyACM*` or `/dev/ttyUSB*` (or the node exists but every open fails)

**Prevention**

- Do **not** interrupt `upload-firmware` / `krabby firmware update` mid-flash
- Prefer a **direct Orin USB port** over a powered hub when flashing
- Use a known-good **data** USB cable (charge-only cables drop the serial device)

**Recover**

1. Unplug the Mega USB cable, wait a few seconds, plug it back in. Check `ls /dev/ttyUSB* /dev/ttyACM*`.
2. If still missing: power-cycle the Mega (USB unplug is often enough; otherwise power-cycle the hub).
3. If the port stays gone: reboot the Orin, then replug and check again.
4. If an upload hangs with the port still present: press the Mega **RESET** button and retry once the port is back.
5. When `/dev/ttyUSB*` or `/dev/ttyACM*` is back, re-run the upload (`PORT=` if needed). Do not chase a “bad sketch” until the serial device has reappeared.

**Vanished port vs bad sketch**

| Observation | Likely cause |
| --- | --- |
| No tty / open fails with I/O error | USB-serial enum (cable, hub, interrupted upload) — recover as above |
| Port opens; boot prints nothing useful or wrong behavior | Sketch / pin rev / role — flash again once the port is stable |
| Second Mega on the same cable path works | First board’s USB-serial path, not the Orin |

#### Remote flashing over SSH (boards on another host)

When the USB hub is plugged into a **different machine** than the one you build on — e.g. a Jetson Orin you reach over SSH — use **`flash-remote`**. It compiles locally (where the arduino-cli toolchain lives), copies the `.hex` to the remote, and runs **`avrdude`** there against the board's serial port. No S3 publish and no Docker image needed; it flashes your exact working-tree build.

```bash
# from the build machine (REMOTE = any ssh target; PORT = the device ON the remote)
make -C firmware flash-remote REMOTE=user@orin PORT=/dev/ttyACM0
make -C firmware flash-remote REMOTE=orin PORT=/dev/ttyACM0 PIN_REV=1
```

One-time setup on the remote: `sudo apt install avrdude` and make sure your user can open the port (add to the `dialout` group). Flash the three boards one at a time, passing each board's `PORT` (find them with `krabby firmware show`, or `ls /dev/ttyACM*` / `ls /dev/ttyUSB*` on the remote). Overridable knobs: `AVRDUDE`, `SSH`, `SCP`, `REMOTE_HEX` (staging path on the remote) — see `firmware/Makefile`.

This is distinct from `krabby firmware update` (which downloads a **published** HEX from S3) — `flash-remote` flashes a **local, unpublished** build.

### 2.4 Python SDK

1. From **`krabby-research`**, install dependencies: `pip install -r firmware/requirements.txt`.
2. Ensure **`firmware/interfaces/`** is importable (e.g. run **`python -m firmware`** from **`krabby-research`** as in §3).

---

## 3. Usage Guide

### Keyboard menu

```bash
# From PyPI or a checkout; on Linux/Mac you may need sudo for keyboard access
python -m firmware
```

For troubleshooting (verbose telemetry):
```bash
python -m firmware --debug
```

### Joint GUI

Shipped in `krabby-firmware` (PyPI). Needs a display and Tk (`sudo apt install python3-tk` on Orin if missing):

```bash
pip install 'krabby-firmware>=0.2.18'
python -m firmware.gui
# or: krabby-firmware-gui
# optional: python -m firmware.gui --port /dev/ttyACM0 --debug
```


### EEPROM address layout

| Address | Size | Purpose |
|---------|------|---------|
| 0–29 | 30 bytes | `CalData` v2 — min/max (int16) for 6 actuators at 0–11 and 12–23, magic `0xCA2B` at 24, 2-bit sensor kind per joint at 26 (bits 2j..2j+1: 0 unknown, 1 pot, 2 hall; bits 12–15 zero), checksum at 28 (sum of bytes 0–27 XOR `0x5A5A`) |
| 30–31 | 2 bytes | Reserved (alignment gap) |
| 32 | 1 byte | Role magic sentinel (`0xAB`) — written once after first successful role election |
| 33 | 1 byte | `BoardRole` value: `1`=FRONT, `2`=LEFT, `3`=RIGHT |

Pot limits are ADC readings; hall limits are frame edges (hall hip `0..1069`, its derived stroke `CAL_HIP_STROKE_EDGES`; knee `-S..0`). On load each joint is validated: a hall hip only as `0..hi` with `hi` within 1049-1089 (the derived 1069, or a switch-verified value from a hollow-stand sweep), a hall knee as `-S..0` with `S` within 450-1500. A hall hip record saved by an earlier build (`[0, 700]`, or its measured stroke) is invalid and is reset with `Calibration joint <name> invalid: defaults`, so **run `C` again after flashing this build**. A loaded hall hip with `hi` = 1069 also prints `Calibration joint <name> extend derived (1069 edges): not switch-verified` at boot. A v1 record (magic `0xCA1B`, 26 bytes, pot only) is ignored on quadrature builds (run calibration again) and read as all-pot elsewhere. On a quadrature build a v2 record that stores any yaw as pot can only come from a build without halls (which saves every joint as pot), so it is ignored as a whole: every joint stays unknown until `C`. Only a completed SWEEP writes the record. The save clears the magic first, writes the rest, and writes the magic last, so a save cut short by a power loss leaves no record (defaults) rather than a mix of old and new values. The load result is printed as one of the lines in the Feature 1 operator-line table.

The role bytes survive power cycles. On each boot, the board prints `ROLE_HINT: LEFT/RIGHT/FRONT` immediately before the 3-second role-election window. `krabby-firmware show` reads this hint so follower boards can be labeled correctly even when probed individually (when they would otherwise appear as `ROLE_UNKNOWN` and show as "front").

Role bytes are only written when a valid role is elected (FRONT, LEFT, or RIGHT). A board that times out as ROLE_UNKNOWN does not update EEPROM, preserving the last valid role.

### Feature 1: Auto-Calibration

On this robot the hips and knees are **hall-encoder actuators**. Their counts are relative, so each hip and knee must be **homed** (curled to its end) after every power-up before position control works, and a full calibration (SEEK then SWEEP) measures each one's range and saves it: each leg in a **table-safe order**: the knee is parked 264 edges out of its curl stop, the hip is swept **by count to 840 edges** and back with the knee held there, and the knee is then swept home. The hip is never driven to its extended end on a solid stand (at 1069 edges no knee setting keeps the leg clear of the belly, the table and the floor together). Its saved range `[0, 1069]` is derived from the front-left leg's measured stroke and is marked not switch-verified. The firmware also supports pot actuators (Rev 2 boards, or a pot fitted on a Rev 3 slot); each joint's sensor kind is **detected automatically** during the run (see "How a sensor is recognised"). The run is slow, curls the whole robot first, and moves exactly one actuator at a time. Stand clear anyway, and keep a hand on **Hold All**: it stops the calibration immediately (see "Stopping it" below).

**Operator rules (read these first)**

 1. **Flash all three boards together**, from the same build. An old follower would run the old pot-only routine on hall joints and drive them to random stops. The robot-level run is only safe on a matched set (see also section 4.3). A Rev 3 leader checks this as far as it can: it refuses to start while a linked follower has sent no `SENS` line since the leader booted (`CAL ROBOT FAIL <role>: no SENS: flash all three boards with the same firmware, or wait a few seconds after connecting`), and nothing moves. Followers print `SENS` at boot and every 2 s, so right after connecting wait a few seconds before `C`. The check cannot see a follower reflashed with old firmware later in the same leader session.
 2. **No jogging is needed for the nudge.** After a power-up most hips and knees rest fully curled (the legs are stored that way). A hip or knee resting fully curled is nudged out a little and straight back so its sensor can be checked (at most 24 hall counts for a hip, 12 for a knee). A curled knee springs out 5-9 more counts after its nudge stops, so a knee nudge ends about 18-21 counts out; that is expected and accepted. A joint that still shows no movement stops the run, and the run names it in its `CAL ROBOT FAIL` line; nothing else moves after it.
 3. **Run the full calibration with the robot on its stand**, legs clear of the ground and anything else. After a power-up the first curl cannot tell an obstacle from the curl end, so a leg curled against the ground would be taken as its home. The SWEEP checks catch a wrong home later, but only the stand makes the run harmless.
 4. **Watch the first move of any newly fitted or rewired actuator with Hold All ready.** If the motor and the sensor are both reversed, every sensor check passes: "curl" runs to the uncurl end and is accepted. No firmware check can see that; a person watching can.
 5. After each power-up run **Curl only** (GUI) before **Neutral** or any `T` target. Until a joint is homed its `pos` is `nan` and a `T` to it is refused.
 6. **Before any `C` on the robot, jog every hip inside 0-560 edges and every knee to its curl end.** A hip past 560 edges is only safe with its knee parked. Watch every hip `HOME` line: it must follow a `by=switch` or `open=` END line (a limit switch really opened).
 7. **After a hip stops `stuck mid-stroke`, never rerun at 128.** Jog the hip to its curl end, or rerun at 150 (see Recovery below).

**Start it**

 - Keyboard menu (`python -m firmware`): press `9`. This sends a bare `C`: the **full** calibration (SEEK then SWEEP, the table-safe sweep of every leg: knee parked 264 edges out, hip swept to 840 edges and back, and saves) at the default speed, so press it only with the robot on its stand (rule 3). The menu has no Curl only key: to home the encoders after a power-up, use Curl only in the GUI, and stay in the GUI for Neutral (reopening the port can reset FRONT and lose its homes).
 - Joint GUI: the **Calibrate** button (Mega path). Its dialog has a **Curl only** box that runs just the curl pass (`C SEEK <pwm>`) on all three boards and saves nothing. Use it after every power-up to home the encoders. Keep the GUI open between a curl-only run and the full run, because reopening the port can reset the FRONT board on some hosts (see the DTR note below), and a reset leader forgets the run and every home.
 - Any serial terminal on the leader's USB port (250000 baud), one line per command:

| Command | What it does |
| :------ | :----------- |
| `C` | SEEK pass, then SWEEP pass, at the default speed (PWM 128) |
| `C <pwm>` | Same, at PWM magnitude `<pwm>` (integer, 110-150) |
| `C SEEK [pwm]` | SEEK pass only: curl every leg and home the encoders. Never saves anything |
| `C SWEEP [pwm]` | SWEEP pass only (it still curls each leg first) |
| `C PWR_SENSE ...` | Unchanged: power-monitor calibration, handled by the leader only, never forwarded |

A bad speed (non-numeric, or outside 110-150: PWM 80 stalled mid-stroke with no current signature, so nothing slower is accepted) or any other malformed `C` line is rejected with `CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150` and nothing moves. (The old `C ?` / `C VOLTAGE ...` usage help for the power calibration is gone: only `C PWR_SENSE ...` reaches it.) `C ACTUATOR [SEEK|SWEEP] [pwm]` is still accepted as an alias. A `C` sent while a calibration is already running is ignored and answered with `CAL ROBOT BUSY` (`CAL <ROLE> BUSY` from a follower).

**What moves, in what order**

The leader (FRONT, or a lone board with no role) orchestrates the whole robot. For each requested pass, in order SEEK then SWEEP, it runs the pass on FRONT, then LEFT, then RIGHT, strictly one board at a time, so **every board has curled before any SWEEP starts** and only one actuator on the whole robot is ever driven. It sends `C SEEK <pwm>` / `C SWEEP <pwm>` to a follower over its UART link and waits for that follower's `PASS_DONE`. A follower with no link is skipped with `CAL ROBOT SKIP <LEFT|RIGHT> no link`. The SEEK still runs on the linked boards, but a SWEEP never starts while one follower has a link and the other has none: the robot fails `CAL ROBOT FAIL <role>: no link, no SWEEP until every board is linked` before the first SWEEP move. A leader with no follower links at all (a single board on the bench) still runs its SWEEP.

On each board the six actuators are, in index order, leg A yaw, hip, knee, then leg B yaw, hip, knee. **The yaw joints are never driven** (yaw calibration will be re-added later); their stored limits are left as they were, and on Rev 3 they stay `?` (no position) and refuse `T`.

 - **Curl direction** (the same on all six legs): the **hip curls by retracting** (negative PWM) and the **knee curls by extending** (positive PWM). The firmware convention is +PWM = extend, -PWM = retract.
 - **SEEK pass**, leg A then leg B, per leg: **knee curl, hip curl, knee curl again.** A knee curled first while its hip rested extended may stop on the body, and that looks like its curl end. The second knee curl, with its hip curled, is proven: it moves the knee's home deeper if the first stopped short (`WARN <knee> home moved <n> edges`) and fails `stopped <n> edges short of its curl end` if it ends well short of the first.
 - **SWEEP pass**, leg A then leg B. A leg whose hip and knee are both hall encoders (the table-safe order), eight steps per leg:
   1. knee curl;
   2. hip curl (home 0);
   3. knee curl again, as in SEEK: with its hip curled the knee's home is proven before the park counts from it, also when SWEEP runs on its own (`C SWEEP`);
   4. knee uncurl to its limit switch (about 785 edges out on FL). The board records that switch (a raw count) as the knee's **switch frame** and prints `CAL <ROLE> KNEE <knee> home=<raw> sw=<raw> travel=<S> park=<p> pwm=<pwm> (FL 785)`, every run, so the first run records all six knees. `travel` S is the knee's travel from its home to its switch (the `hall=` of the uncurl's END line, negated; its `travel=` is from where the move started). FL's switch-ended travels are 778-789 stop to switch and 775-781 switch to stop at PWM 120-150; FR read 788-789. S is judged against FL's 785:
      - **769-801**: silent;
      - **745-768 or 802-825**: `WARN <knee> travel <S> edges, outside FL 769-801: T and Neutral off by <785 - S>`, and the run goes on. The park is still right (it counts from the switch), but that knee's home, and so its `T` and Neutral positions, sit `785 - S` off FL's;
      - **under 745 or over 825**: the pass fails `<knee> travel <S> edges, outside FL 745-825: check its curl end` here, before the park, with the hip at its curl switch and the knee at its switch; nothing is saved. A home that far off is an obstacle or a bind at the knee's curl end (or a different actuator);
   5. **knee park**: the knee curls back until it is **521 edges in from its switch** (`fromSwitch=521`; at a 785 travel that is the old 264 out of its curl stop, firmware position -264), settles, and must sit inside **505-541 from its switch** (at 785: `-280..-244`), else the pass fails. Because it counts from the switch, which repeats within a count, a home short of the stop no longer moves the park. The line `CAL <ROLE> PARK <knee> hall=<x> fromSwitch=<f> target=<t> (FL values)` reports it (`target` is the park in the knee's home frame, `<switch> + 521`);
   6. **hip sweep**: the hip uncurls **by count to 840 edges** with the knee held in its band, in its switch frame (checked on every loop). After it, `CAL <ROLE> KNEE <knee> sweep fromSwitch=<min>..<max>` reports where the knee sat over the whole sweep. From the knee's uncurl to the end of its hip's curl back no knee home may change: if one does (a guard cleared it, say), the pass fails `<knee> home changed during its park: run C again`, at the park's end or with the hip back at its curl switch, before the knee moves again;
   7. hip back to its curl switch, which must read within **10 edges of 0** (else counts were lost, `LostCounts`). The hip's range is then staged as `[0, 1069]`;
   8. knee curl (this completes the knee, as before). If it ends deeper than the knee's home, the home moves there with `WARN <knee> home moved <n> edges deeper after its park`: the knee's uncurl and park were counted from a home `<n>` edges short (a few edges is the stop's normal spread). The park itself is unaffected: it counts from the switch.
   A leg whose hip is a pot (builds without quadrature halls) keeps the old order: knee curl, hip curl, hip uncurl to its end, hip curl, knee uncurl, knee curl. A hall hip whose knee is not a hall fails `<hip> and its knee must both be hall encoders`.
 - **Table-safe hip sweep (hall hips).** The hip moves above 560 edges only with its knee parked. The sweep is driven by count, to `CAL_HIP_SWEEP_EDGES` = 840 (provisional): the model's worst case allowed 864, but a robot check at 866 put the toe about 5 in from the table side against about 8 in modelled, so 840 keeps margin. With the knee curled the tibia meets the body at about 941 edges. The hip's **extended end is never sought on a solid stand**: at 1069 no knee setting keeps the leg clear of the belly, the table and the floor together.
 - **Derived extend limit.** A hall hip's saved range is `[0, 1069]`: its home plus `CAL_HIP_STROKE_EDGES` = 1069, the switch-to-switch stroke FL measured 3 of 3 on 2026-11-24 (it matches the 203.2 mm label at 5.26 counts/mm). It is **derived, not switch-verified**: the JOINT line says so (`min=0 max=1069 edges (extend derived from FL, not switch-verified)`) and so does the boot line `Calibration joint <hip> extend derived (1069 edges): not switch-verified`. `T` stays 10 edges short of it (the hip's `T` range is 8..1059). Old `[0, 700]` hip records are invalid: run `C`.
 - **All six legs use FL's values** (knee stroke 785 and its travel window 745-825, knee park 521 from its switch, hip sweep 840, hip stroke 1069): only the front-left leg was modelled. The PARK and JOINT lines say so (`FL values`, `derived from FL`). Verify each leg before you trust `T` near the ends of its hip or knee.
 - **Obstacles.** Any end of the hip sweep before 840 edges (a stop, a stall, or a limit switch), or the knee leaving its park band, prints `WARN <hip> stopped at <x> edges: retracting`, retracts the hip to its curl switch, and **fails the pass with no automatic retry**. The reason names the cause (`SweepShort`, `LostCounts`, `KneeOutOfBand`, `ParkShort`, `NotRetracted`: see the failure table). A knee whose travel is outside FL's 745-825 fails (`KneeTravelOutOfRange`) before its park, so its hip never sweeps. When a failure or an abort stops a park, a sweep or a retract, the board prints `WARN <j> left out at <x> edges: jog it to its curl end`.
 - **Optional hollow-stand sweep** (off by default; a build with `-DKRABBY_CAL_HOLLOW_STAND=1` sets option bit `CAL_OPT_HOLLOW_STAND`, which neither `C` nor the GUI can change). The knee parks 192 edges out (band 157-232) and the hip sweeps on to its extend switch, which must be found within **1069 +- 20** edges: the range is then marked switch-verified (`edges (extend switch-verified)`). Elsewhere it is `LostCounts`. **Never on a solid table**: the toe meets the table's side at hip about 940-985.
 - **Kept home.** When a homed hall joint fails, its raw home count is kept (the home itself is cleared, as before). A later curl of that joint is judged against the kept home as if proven: ending more than 40 edges short of it fails `stopped <n> edges short of its curl end` instead of homing there; ending within the tolerance, or deeper, is accepted and re-homed. A lost-counts failure drops it. It is kept across a new `C`. This is what stops a rerun from homing a hip that stalled mid-stroke.
 - **Recovery after a hip `stuck mid-stroke`.** Never rerun at 128: jog the hip to its curl end, or rerun at 150. A rerun is checked against the home the joint proved earlier this boot (kept through the failure): ending more than 40 edges short of it fails `stopped <n> edges short of its curl end` instead of homing there. Reset the board (reopening the port resets FRONT) only if the joint was really moved by hand.
 - **Partner rule.** The hip sweep needs its knee parked. A hip curl, return or retract needs its knee curled or parked. A knee curl is free unless the knee is parked, and a knee park, a knee uncurl or a curl from the park needs its hip curled (a pot hip's uncurl needs its knee curled). A failed step stops the pass before the partner joint moves. The exceptions are the SEEK knee curls and the knee nudge.
 - **A knee never moves while its hall hip is known to be out past 560 edges.** The hip's position counts as known when it is homed this boot, kept after a failure (the kept home), or a home that a guard cleared (`ERR HOME`, the frame of its last home). A knee step then refuses with `<hip> out at <x> edges: jog it to its curl end first`: nothing moves and the hip keeps its home. Jog the hip to its curl end and run `C` again. A hip whose position is not known (after a power-up, before it is homed) is not checked, so operator rule 6 still applies.
 - The one exception is the **knee nudge** (`CAL_OPT_KNEE_NUDGE`), approved by the robot's owner on 2026-11-23 and the shipped default (`CAL_DEFAULT_OPTIONS = CAL_OPT_KNEE_NUDGE` in `calibration_constants.h`): a knee curl that shows no motion is followed by a nudge out of at most 12 edges (1 s cap, never faster than PWM 128, the default, whatever the `C` speed), then a full curl straight back, before its hip is proven. A stored knee needs 0.36-0.77 s of drive to break free of its curl end and reached 12 edges 0.67-1.0 s after the start (FLKL, 2026-11-23), so there is no earlier no-edge stop: a knee with no hall feedback (hall A dead, cable off) runs blind to the 1 s cap at no more than PWM 128, so it can end **up to about 95 edges out** if it breaks free at once (measured 2026-11-24: FLKL driven out of its curl end at 128 had moved 72 edges 1.0 s after its drive started, with a ramp slower than the firmware's; the FR knees run about 120-130 edges/s at 128 and coast 3-9 edges after a stop; at PWM 80 this was about 41), is curled back and fails `no feedback` before its hip moves. Its hip is not proven curled at that point (the nudge is the one move allowed before that), so the knee nudge's PWM cap trades this blind reach against breaking a stored knee free: FRKL could not break free at 80 or 100. The owner kept the cap at 128 (decision F3, 2026-11-24): a blind knee nudge unfolds away from the body, so the reach is accepted. The 12-edge and 18-21-edge figures below hold only for a knee whose hall works. After the drive stops a curled knee **springs out 5-9 more edges**, so a 12-edge nudge ends about 18-21 edges out; the owner accepted this, and those spring edges count toward the 12 edges of evidence (read after the 0.3 s settle). The curl back is accepted when it moved at least 2 edges toward curl and stalls no more than 40 edges short of where the knee rested before the nudge: a knee parked by a GUI jog (PWM 200) rests about 17 edges deeper than PWM 80 reaches (about 7 deeper than PWM 128 reaches) and at PWM 80 came back only about 5 edges. A knee whose curl back stalls inside the 40 edges but moved fewer than 2 fails `<j> curled back only <n> edges after its nudge`. At PWM 80 this happened once for a knee parked by the GUI's PWM 200 curl, at the low end of its spring-out (5-6 edges): the nudge already ended at its calibration-speed curl end (at the default PWM 128 the host simulation no longer shows it). It then rests there, and the next Curl only passes; just run Curl only again, without re-curling the knee at PWM 200. Every `C` run uses it. It is still a build-time default: neither the `C` command nor the GUI can change it. A build with `CAL_DEFAULT_OPTIONS = 0` is strict: there a knee with no motion fails `<j> no motion seen: jog it off its curl end, then retry` instead.
 - **Hip nudge.** A hip curl that shows no motion is not accepted on its own. The hip is driven out at most 24 edges (2 s cap), then a full curl back. The hip nudge is capped at PWM 128 whatever the `C` speed (a `C 150` hip nudge ramps only to 128), and on a quadrature build it stops after 1 s if it counted no edge: a blind hip would otherwise drive out about 140 edges in 2 s at 128 toward the body. If a hip or knee nudge itself shows no feedback, the curl back still runs before the run fails. If a run ends while a joint is nudged out (a wiring fault found during the nudge, or `H`), the leg is not driven back: the board prints `CAL <ROLE> WARN <j> nudged out, not driven back: jog it to its curl end` just before its `FAIL` or `ABORTED` line. The `MOVE <j> nudge` line announces each nudge, and **Hold All** stops a nudge at once like any other move.

**How a sensor is recognised**

Each hip and knee is classified from evidence, never from the pot reading alone. On a Rev 3 board the analog pins A0-A5 (the "pot" pins) are **hall channel B** on the hips and knees, so they swing between the rails whenever the joint moves.

 - **Hall**: 12 or more edges counted in the driven direction while driving. 12 or more edges **against** the drive fails the move at once (`hall counts the wrong way`), in any phase.
 - **Pot**: decided only from two snapshots taken with the motor stopped (each the mean of 4 direct reads): both must lie in 32..990, the move must have counted at most 1 edge, and the reading must have travelled at least 30 counts toward the drive. A pot whose end reads below 32 or above 990 cannot be told apart from hall B on Rev 3 and fails `no feedback` or `no motion` (visible and safe).
 - **Unknown**: no evidence yet. Such a joint is never accepted as curled on that move.
 - A kind decision or change is announced with `CAL <ROLE> SENSOR <j> hall|pot`. Without a quadrature board (Rev 2) every joint is a pot.

**How fast and how an end is found**

 - Default speed is PWM 128; **`C` accepts PWM 110-150** (the owner's choice of 2026-11-24, was 80). **Calibrate at the default PWM 128.** `C` accepts 110-150, but 110-127 is not validated on the robot (the stall evidence is at 80 and 128). PWM 80 is no longer accepted (owner decision F4): it stalled mid-stroke with no current signature (FLHL stopped 229 edges short of its curl switch, 2026-11-24), the FR knee could not break free of its curl end at 80 or 100, knee uncurls stalled short of their switch, and the FR knee's curl end showed almost no current rise; at 120 and 150 every front-leg end showed its signature. Below 128 a stored joint may not break free during its nudge and then fails `<j> no feedback at PWM <p>: check sensor cable, or calibrate at PWM 128`, and a healthy joint that stalls with no end signature fails `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load`. Each move ramps up from 0 in steps of 5 every 20 ms (0.52 s to 128) and never exceeds the requested PWM. A full stroke can still take several seconds.
 - **End detection (hall joints on Rev 3 boards).** Every hall (or not yet classified) hip and knee drive but the nudge out is ended by the end detector (`calibration_end_detector.h`), from the hall count and the motor current (`IS`, 10 x the smoothed reading, sampled every 50 ms), and the motor stops at once instead of pressing for seconds. Once a drive has run (at least 20 edges/s for 0.3 s, from 0.2 s after the ramp), the drive's end kind picks the rule.
   - **Into a limit switch** (every hip curl, every knee uncurl, and the hip's extended end): the end is the actuator's internal **limit switch**, and it counts as an end **only with the switch-open signature**: the count has stopped and the motor current is flat (within 1 count) at the channel's own offset, the level it also reads at the start of a drive. The drive measures that level itself in its first samples (its floor: FL 7-8 counts, FRHL 11). A motor stalled mid-stroke still conducts, so its current sits above the floor and wobbles (FL at PWM 80 9-14, FRHL at 128 17-31). The end is called about 0.3-0.5 s after the count stops (six samples flat at the floor; harmless, the switch has already cut the motor) once the current is at least 12% below its running median (a joint that arrives crawling must also be below every sample of that median's window and at least 10% below its running level). Arrival speed is taken over the last few edges, and a drop in progress holds the decision for up to 1 s, so a switch end is not lost one sample from stuck. **A stall without the signature is never taken as the end, however short the drive** (a nudge return, a start next to the stall point): it fails `stuck mid-stroke`. The one exception is a curl press that counted no edge at all: it is not accepted either, but goes on to the joint's verification nudge, whose return must show the signature. A stall on such a drive that has never run waits until the count has been still 0.45 s, then ends only if the signature is there. FRHL curling at 128 from the body stalled 878 edges short of its switch with its current wobbling above the floor, 2026-11-24: that is stuck, not a switch.
   - **A knee curl** (extending) ends at the leg's **mechanical stop**, which has no switch: the speed falls below 25% of running while the current rises at least 25% over its running median for 3 samples, or the knee simply stalls (any stall). An obstacle on a knee curl looks the same as its stop. The knee's second curl (SEEK) and the SWEEP re-curl catch it, and the SWEEP's knee travel check fails a home far from its stop.
   - **A knee curl that reaches its actuator's extend switch** before the leg's stop shows a current drop to the switch-open level, not a rise. No FL or FR trace shows one (FL's stop sits about 21 counts short of full extension; the other legs are unverified). The detector has no rule for it: a long curl that stalls so ends `stuck` and fails `<knee> stuck after <n> edges, switch open: extend switch before the stop?`; a short one (a press) ends as a plain stall and is accepted as before. Either way its END line ends ` open=1`. A knee curl's plain-stall and `by=stuck` END lines carry ` cur=<a>/<b> floor=<f>` (`/0`: no running reference). Treat ` open=1` on a knee as a sign that leg's actuator, not its stop, ends the curl: an open decision for the owner.
   - **A hip extend** (every hip sweep, on any stand) ends at its extend switch by the switch rule, or by the stop rule, which on a hip also needs the count still for 0.25 s (5 samples, unless the rise lasts 8 samples) so the extend switch can show its drop first. A stall with neither signature is `stuck`. On a hip sweep every end before its target, of any kind, is an obstacle.
   - The END line names the kind: `... by=switch cur=<a>/<b> floor=<f>`, `... by=stop cur=<a>/<b>` or `... by=stuck cur=<a>/<b> floor=<f>` (tenths of an IS count: the current at the decision, its reference, and on a switch-end drive the drive's floor). A stall on a switch-end drive that ends without a running phase prints ` open=<a>/<f>` (its current and the floor) instead of `by=`.
   - **A curl press that shows no motion at all** (no edge) and no switch-open current is not failed, and never accepted on its own: more than the tolerance short of a home proven earlier it fails `stopped <n> edges short of its curl end`; otherwise the joint gets its verification nudge, whose return needs the switch-open signature (and only there is it homed). So a hip resting at its curl switch whose current channel reads a count high is not failed `stuck`.
   - A stall without a running phase is only checked by its count. A drive that never ran (a press of a joint already at its end, a short curl back, a joint slower than 20 edges/s) is judged by stall alone: a knee curl after at least 1 s of drive (the curl back after a nudge at least as long as the nudge drove, and 1 s longer when the nudge saw no edges: joints curl back slower than they uncurl) with fewer than 3 edges in 0.5 s, or in 1 s once it has made 12 edges (a slow joint's pause is not its end), so a knee already at its end finishes in about 1 s; an uncurl gets the old 1.5 s + 1 s break-free allowance after the ramp. So below arming speed an end is found by stall alone, and a slow knee that pauses for over a second can still pass for being at its end. A curl that starts about 50-130 edges from its end arms while already slowing into it, so it has no running phase either: when it stalls with fewer than 12 normal-speed samples since arming and its last edge within 2 s of arming, it ends as a plain stall (no `by=`) instead of failing stuck: always for a knee curl, and for a drive into a limit switch only with the switch-open signature (else stuck).
   - Proven pot joints and builds without quadrature halls keep the window rule: after the ramp a 1.5 s grace period, then 1 s windows that must each move a pot 3 counts of its smoothed reading (a hall joint 3 edges). After any stop it waits 0.3 s and reads the end. While a joint is not proven Pot, the hall-B level on its pot pin is never used to stop, classify or fail a move.
 - Each move has a hard 120 s timeout. The longest move measured at PWM 80 took 36 s (FLHL curling back from about 1000 edges), so it has margin; moves at the default PWM 128 are faster. It also ends a joint that keeps creeping (an edge at least every 0.4 s is never stuck). The leader gives up on a silent follower after 135 s with no `CAL` line from it.
 - **A knee curl that moved and then stalled is accepted as the curl end** (12 or more edges, or 30 counts); a hip curl or knee uncurl is accepted only with the switch-open signature (see End detection). Stated plainly, this is a limit: after a power-up an obstacle looks the same as the end, which is why the full run belongs on the stand. A curl with **no** motion is accepted only at a curl end proven earlier this boot (within the tolerance below; a hall joint resting deeper moves its home there); a joint with a proven curl end that rests more than the tolerance short of it fails `stopped <n> edges short` with no nudge. Only a joint with no proven curl end is nudged (a strict build fails a knee instead).
 - Once a joint has a home from this boot, later curls are checked one-sided: ending more than 40 edges (30 counts for a pot) **short** of it fails (`stopped <n> edges short of its curl end`); ending deeper is accepted and moves the home (`WARN <j> home moved <n> edges`). A joint whose home was cleared by a failure is judged against its kept home (see Kept home).

**Homing, position and `T`**

 - Encoder joints use a frame where a **higher value means more extended**: `x = -(count - home)`. The hip curl end is 0 and a hall hip's limits are `[0, 1069]` (its derived stroke, not switch-verified); the knee curl end is 0 and its limits are `[-S, 0]`, with `S` the measured stroke in edges.
 - A joint is homed in RAM when its curl is proven (`CAL <ROLE> HOME <j> count=<raw>`). **The counts and every home reset at each power-up and board reset**, so Curl only is needed every time.
 - An encoder joint's `pos` is a number only when it has a home from this boot **and** a stroke from a saved SWEEP; otherwise `pos` is `nan` and a `T` to it is refused (one `ERR T refused <names> (no position)` line per board per second). Yaws and never-classified joints are refused the same way. The GUI Neutral button and the menu `0` key send only to hips and knees that have a position and name the ones they skip. On a solid stand keep every hip `T` at or below 560 edges (see Feature 3).
 - Encoder `T` is capped at PWM 150, with a 12-edge deadband and targets kept inside the ends calibration recorded: 8 edges inside the retracted end (a limit switch: the hip curl end, the knee uncurl end), 25 edges inside the knee's extended end (its compliant leg stop, which a harder push enters deeper) and 10 edges inside the hip's extended end (derived: a hip's `T` range is 8..1059). A `T` stall guard (both kinds) stops a joint that moves less than 3 in 1 s at PWM 60 or more (`ERR STALL <name>`; an encoder also loses its home). Driven at PWM 20-59, a joint that moves less than 3 in 3 s is stopped quietly (no `ERR` line, the home is kept). After an `ERR STALL` trip the joint stays stopped while any `T` in the stopped direction is sent (a held key such as menu `0`, a streaming host): it is not driven again and no `T refused` line is printed, until a `T` that reverses the drive, a jog (`J`/`B`) or Hold (`H`) re-arms it. After a quiet stop only the same `T`, or one whose target is within 12 edges of the stopped target on the same side, keeps the joint stopped; a different target drives again (still covered by the stall guard), and a reversing `T`, a jog or `H` also re-arm it. A range guard clears an encoder home that is clearly wrong (`ERR HOME <name> lost`). Both guards run only outside calibration. A hall hip's `T` range is `[0, 1069]` (derived), so with the insets its targets run 8..1059 edges. The range guard allows up to the stroke plus the 60-edge margin (a hip jogged past 1069 + 60 = 1129 edges loses its home: `ERR HOME <hip> lost`, then `T` is refused until the next Curl only or `C`). The curled tibia meets the body at about 941-1003 edges, and at 1069 no knee setting keeps the leg clear of the belly, the table and the floor together: do not drive or jog a hip that far on a stand.

**Results and saving**

 - For each hip and knee, `min` and `max` are the stops in the joint's own units. Pot: the ADC reading at the retracted and extended ends (the existing `CalData` meaning). Hall: frame edges, hip `0 .. 1069` (derived; its JOINT line ends `edges (extend derived from FL, not switch-verified)`, or `edges (extend switch-verified)` after a hollow-stand sweep), knee `-S .. 0`.
 - Hall knee strokes are checked: the stroke must lie between 450 and 1500 edges (provisional: one knee measured, FLKL 776 edges; otherwise `stroke <n> edges, outside 450-1500`), a stroke under 85% of the saved stroke fails (`stroke <n> edges, saved <m>: obstacle?`), and a stroke that differs from the saved one by more than max(30, 10%) prints `WARN <j> stroke <old> -> <new> edges`. These checks do not apply to hall hips, whose range is derived.
 - Results are staged during the pass. A board writes its limits to EEPROM only when its **SWEEP pass completes with no failure** (`PASS_DONE SWEEP saved`). SEEK-only runs never save. Any failure or abort leaves that board's saved and in-RAM limits as they were (boards that already finished their SWEEP keep what they saved). A failed hall joint loses its home (its raw home count is kept as the kept home, see What moves); an abort (`H`) keeps the home, because the counts keep tracking.
 - A reversed pot no longer warns and saves: it fails `pot moves the wrong way: check motor or pot end wires`, because accepting a wrong-way "curl" would let a partner move next to an unproven joint.
 - The record is 30 bytes (v2) and is reloaded at boot (see "EEPROM address layout"). However calibration ends, every actuator is stopped and has no position target, so nothing starts holding or moving afterwards.

**Stopping it**

`H` (Hold All), `J` and `B` (jog) and `T` (position targets) each **abort calibration immediately**: all motors off, nothing saved, `CAL ROBOT ABORTED <letter>` on the leader. The leader sends `H` to the followers first, then handles and forwards the command as usual, so a jog sent after an abort still works. In the GUI, **Hold All** sends `H`. `V` (version) and `C PWR_SENSE ...` keep working during calibration; while a calibration runs, `V` answers for the leader only (`-` for each follower), because asking the followers would stall the loop and could split a follower's `CAL` line.

If a follower reports `FAIL`, `ABORTED`, `BUSY` or `ERROR`, or goes silent past the watchdog, the leader sends `H` to every follower, stops its own motors and prints `CAL ROBOT FAIL <role>: <reason>` (the reason is cut at 78 characters).

A leader that resets (USB DTR auto-reset when a program opens its port, or a brown-out) forgets the run; at boot it sends `H` to both follower links, so a follower still running its pass stops. A follower that reports a `MOVE` outside its turn is treated the same way: during a run it fails the robot (`CAL ROBOT FAIL <role>: MOVE outside its turn`), and with no run active the leader sends that follower `H` and prints the same line.

**Known limit: a dropped follower link.** `H` cannot reach a follower whose UART link has dropped. That follower finishes or fails its own pass under its own bounds (each move at most 120 s, one joint at a time, curl first, hip nudge at most 2 s, knee nudge at most 1 s, a hall hip's sweep at most 840 edges, 1089 on a hollow stand). The power switch is the backstop.

**Status lines**

Each board prints one `CAL <ROLE> <event>` line per event on its main serial (USB on the leader, UART uplink on followers); the leader forwards follower lines to the host. `<ROLE>` is `FRONT`, `LEFT`, `RIGHT` (a lone board with no elected role prints `UNKWN`). They are never parsed as telemetry. `<j>` is a joint name such as `FLHL`; `<n>` and `<p>` are integers.

| Line | Meaning |
| :--- | :------ |
| `CAL <ROLE> PASS SEEK pwm=<p>` / `PASS SWEEP pwm=<p>` | Board pass starts |
| `CAL <ROLE> MOVE <j> curl\|uncurl\|nudge\|park\|sweep\|retract` | A move starts (`nudge` is the verification nudge; `park` = the knee's park, `sweep` = the hip's table-safe sweep, `retract` = the hip's retreat to its curl switch after an obstacle) |
| `CAL <ROLE> SENSOR <j> hall\|pot[ (was hall\|pot)]` | The joint's sensor kind was decided (or changed) |
| `CAL <ROLE> END <j> <dir> hall=<frame> travel=<frame delta>[ by=switch\|stop\|stuck cur=<a>/<b>][ floor=<f>][ open=<a>/<f>][ cur=<a>/<b> floor=<f>][ open=1]` | A move ended on a hall joint. `<dir>` is `curl`, `uncurl` or `nudge`; travel is signed, in frame edges. `by=` names the end the detector found, `floor=` is the drive's switch-open level on a drive into a limit switch, `open=` marks a stall at the switch-open level without a running phase (tenths of an IS count); a knee curl's plain stall prints ` cur=<a>/<b> floor=<f>`, and ` open=1` marks a knee curl stalled at the switch-open level |
| `CAL <ROLE> END <j> <dir> hall=<frame> travel=<frame delta> pot=<snap>` | Same, for a joint with no evidence yet (kind still unknown); `pot` is the at-rest snapshot |
| `CAL <ROLE> END <j> <dir> pot=<end> travel=<end-start>` | A move ended on a pot joint (signed travel in pot counts) |
| `CAL <ROLE> HOME <j> count=<raw>` | The encoder home was set at this raw count |
| `CAL <ROLE> WARN <j> home moved <n> edges` (or `counts`) | A curl ended deeper than the home from this boot; the home moved |
| `CAL <ROLE> WARN <knee> home moved <n> edges deeper[ after its park]` | The SWEEP's last knee curl ended deeper than the knee's home; the home moved. Its uncurl (and on a table-safe leg its park) was counted from the shallower home |
| `CAL <ROLE> KNEE <knee> home=<raw> sw=<raw> travel=<S> park=<p> pwm=<pwm> (FL 785)` | A table-safe knee at its uncurl switch: its home and switch as raw counts, its travel S from home to switch, and its park in its home frame (`<switch> + 521`; `+ 593` on a hollow stand) |
| `CAL <ROLE> WARN <knee> travel <S> edges, outside FL 769-801: T and Neutral off by <n>` | The knee's travel is 745-768 or 802-825: it parks right (from its switch), but its `T` and Neutral positions are `<n>` edges off FL's |
| `CAL <ROLE> PARK <knee> hall=<x> fromSwitch=<f> target=<t> (FL values)` | The knee is parked inside its band for the hip sweep, `<f>` edges in from its switch (505-541; 553-628 on a hollow stand, where it parks 593 in) |
| `CAL <ROLE> KNEE <knee> sweep fromSwitch=<min>..<max>` | Where the parked knee sat, from its switch, over its hip's sweep |
| `CAL <ROLE> WARN <hip> stopped at <x> edges: retracting` | The hip sweep ended before 840 edges, or the knee left its band: the hip retracts to its curl switch and the pass fails (no retry) |
| `CAL <ROLE> WARN <j> left out at <x> edges: jog it to its curl end` | A failure or an abort stopped a park, a sweep, a retract or the hip's curl back from its sweep: jog that joint to its curl end before anything else (the next `C` refuses to move a knee while its hip is known to be out past 560 edges) |
| `CAL <ROLE> WARN <j> stroke <old> -> <new> edges` | The new stroke differs from the saved one by more than max(30, 10%) |
| `CAL <ROLE> WARN <j> nudged out, not driven back: jog it to its curl end` | Printed just before a `FAIL` or `ABORTED` line when the run ended during a nudge. Jog that joint back to its curl end |
| `CAL <ROLE> JOINT <j> min=<a> max=<b>[ edges]` | A joint's result (SWEEP); the `edges` suffix marks a hall joint |
| `CAL <ROLE> JOINT <hip> min=0 max=1069 edges (extend derived from FL, not switch-verified)` | A hall hip's result: its extend end is FL's stroke, not found by this run (a hollow-stand sweep ends `edges (extend switch-verified)`) |
| `CAL <ROLE> PASS_DONE SEEK` / `PASS_DONE SWEEP saved` | Board finished the pass |
| `CAL <ROLE> FAIL <reason>` | A step failed; the motor is stopped and the partner joint is not moved. Reasons are in the table below |
| `CAL <ROLE> ABORTED <H\|J\|B\|T>` | Stopped by a command |
| `CAL <ROLE> BUSY` / `CAL <ROLE> ERROR usage: ...` | Follower rejected a `C` line |
| `CAL ROBOT START pwm=<p> passes=SEEK,SWEEP\|SEEK\|SWEEP` | Leader starts |
| `CAL ROBOT SKIP <LEFT\|RIGHT> no link` | Follower link absent (printed once per pass) |
| `CAL ROBOT DONE` | Everything finished |
| `CAL ROBOT FAIL <role>: <reason>` | Robot calibration stopped on a failure |
| `CAL ROBOT FAIL <role>: no link, no SWEEP until every board is linked` | One follower is linked and the other is not: no SWEEP runs. Fix the link and run `C` again |
| `CAL ROBOT FAIL <role>: no SENS: flash all three boards with the same firmware, or wait a few seconds after connecting` | Rev 3 leader only, at start: that linked follower has sent no `SENS` line since the leader booted (other firmware, or connected a moment ago). Nothing moved and there is no `START` line. Wait a few seconds and run `C` again. If it repeats: check that board's own output (its USB port, or its startup lines) for `ROLE: UNKNOWN` (no role elected, so it never prints `SENS` on the link); power-cycle or reset all three boards together and wait a few seconds; reflash all three boards with the same build only if `SENS` is still missing. This line is longer than the other `CAL ROBOT FAIL` lines (about 117 characters); the GUI and the menu show it whole |
| `CAL ROBOT ABORTED <cause>` / `BUSY` / `ERROR ...` | Aborted by a command / `C` while running / bad `C` line |

The old reversed-pot `WARN` is gone (see "Results and saving").

**Failure reasons** (`CAL <ROLE> FAIL <reason>`; the reason is at most 78 characters). The unit is `edges` for a hall or unclassified joint and `counts` for a pot.

| Reason | Meaning and what to do |
| :----- | :--------------------- |
| `<j> pot not connected` | A proven pot's pin reads open. Nothing moved. Check the pot cable |
| `<j> <curl\|uncurl>: no end of travel in 120 s` | The move timed out; motor stopped, nothing saved. Check for a stuck sensor or an obstacle. If the stroke is simply long for the speed, the timeout (`CAL_MOVE_TIMEOUT_MS`, see the measurement below) must be raised in the firmware; do not raise the calibration speed to fit it |
| `<j> barely moved (<n> edges) at PWM <p>: check hall wiring or raise speed` | An uncurl travelled too little, or a nudge counted a few edges but too few to prove the sensor (a slow joint rather than a dead cable). A pot says `counts` and `check pot wiring` |
| `<j> did not reach its curl end (hall=<n>)` / `(pot=<n>)` | A curl stalled short of its curl end |
| `<j> hall counts the wrong way: check motor or hall A/B wiring` | The count ran against the drive. Reversed motor polarity and reversed hall A/B look the same |
| `<j> pot moves the wrong way: check motor or pot end wires` | The pot travelled against the drive |
| `<j> no feedback: no hall edges, pot not moving (check sensor cable)` | Neither sensor showed anything (a hip or knee, after its nudge and curl back) |
| `<j> no motor current at PWM <p>: check motor wiring` | No motor current was seen while driven |
| `<j> no motion seen: jog it off its curl end, then retry` | Strict builds only (`CAL_DEFAULT_OPTIONS = 0`): a knee showed no motion (probably resting fully curled). Jog it off its curl end and retry. The shipped build nudges such a knee instead |
| `<j> no feedback at PWM <p>: check sensor cable, or calibrate at PWM 128` | As `no feedback`, but the run's speed `<p>` was below the default PWM 128, where a stored joint may not break free during its nudge (FRKL could not at 80 or 100). Run `C` (or Curl only) again at the default PWM 128; if it repeats there, check the sensor cable |
| `<j> stopped <n> edges short of its curl end` (or `counts`) | A curl ended short of a curl end proven this boot (with or without motion): an obstacle? After a nudge, the curl back stopped more than 40 edges (30 counts) short of where the joint rested before the nudge. Clear the obstacle, jog it to its curl end and retry |
| `<j> curled back only <n> edges after its nudge` (or `counts`) | After a nudge, the curl back stalled within 40 edges of where the joint rested but moved too little toward curl (a knee fewer than 2 edges, a hip fewer than 12): a joint that cannot curl may be stuck mid-stroke, so its hip is not moved. For a knee parked by the GUI's PWM 200 curl this can happen once, at the low end of its spring-out: the knee now rests at its calibration-speed curl end, so **just run Curl only again** (do not re-curl it at PWM 200). If it repeats, look for something holding the joint |
| `<j> stopped <n> edges short of its 840-edge sweep: obstacle? check its home` | A hall hip's sweep ended `<n>` edges short of its 840 edges (SweepShort). Something is in the way (the leg, the table, the body), or its home is wrong. The hip has retracted to its curl switch (`WARN <hip> stopped at <x> edges: retracting`); clear the obstacle, check that it is home, and run `C` again. No automatic retry |
| `<j> lost counts: switch at <x> edges, expected 1069` (or `expected 0`) | The hip's sweep met a limit switch at `<x>` edges instead of at 1069 (LostCounts), or the return found its curl switch more than 10 edges from 0 (`expected 0`): counts were lost. Home the hip again (Curl only) before another run |
| `<j> at <f> from its switch, outside its park band 505..541` | The knee left (or parked outside) its park band, counted from its switch (KneeOutOfBand; `553..628` on a hollow stand). Check what holds or moves the knee; the hip has retracted if it was sweeping |
| `<j> stopped <n> edges short of its park at -264: obstacle?` | The knee's park drive ended `<n>` edges short of -264 (ParkShort): an obstacle? Clear it, jog the knee to its curl end and run `C` again |
| `<j> travel <S> edges, outside FL 745-825: check its curl end` | A table-safe knee's travel from its home to its switch is more than 40 edges off FL's 785 (KneeTravelOutOfRange): its home is far from its curl stop. Nothing has parked and the hip has not moved; the knee rests at its switch. Look for an obstacle or a bind at the knee's curl end (or a different actuator), clear it, jog the knee to its curl end and run `C` again |
| `<j> home changed during its park: run C again` | A knee's home changed between its uncurl and its hip's curl back (KneeFrameMoved), so its switch frame no longer matches its home. Nothing more moves; run `C` again |
| `<j> stuck after <n> edges, switch open: extend switch before the stop?` | A knee curl stalled with its current at the switch-open level (StuckMidStroke): its actuator may have reached its extend switch before the leg's stop. FL's stop sits about 21 counts short of full extension; the other legs are unverified. Check that leg's stop and actuator before rerunning |
| `<j> not back at its curl switch (hall=<x>): jog it home` | The hip's retract did not reach its curl switch (NotRetracted). Jog it to its curl end before anything else |
| `<j> and its knee must both be hall encoders` | The table-safe sweep needs a hall hip and a hall knee (TableSafeNeedsHall) |
| `<j> out at <x> edges: jog it to its curl end first` | A knee was about to move while its hall hip is known (homed this boot, or by its kept home) to be out past 560 edges, where the knee must stay parked (HipOut): nothing moved. Jog the hip to its curl end, then run `C` again. A hip whose position is not known (after a power-up) is not checked: rule 6 |
| `<j> no edges and no switch signature at PWM <p>: check hall cable or load` | A drive into a limit switch counted no edge at all and its current never sat at the switch-open level (StuckMidStroke with no motion): a joint that cannot move from where it rests (stalled mid-stroke, jammed), or one whose hall cable is off while its motor drives it blind. Check the cable first; if it is fine, treat it as stuck mid-stroke (jog the hip home, or rerun at 150) |
| `<j> stuck mid-stroke after <n> edges at PWM <p>: raise PWM or check load` | A hall drive into a limit switch stalled without the switch-open signature (current flat at its own offset), or a knee drive stalled and stopped with no end signature at all: an obstacle, a joint too weak at this speed, a hall dropout, or a bypassed limit switch. It is not accepted as an end, however short the drive (a curl press that counted no edge at all goes on to the verification nudge instead, and its return must show the signature): the joint's home is cleared (its raw home count is kept), the motor is stopped and the partner joint is not moved. **Never rerun a hip at 128**: jog it to its curl end, or rerun at 150; a rerun that ends more than 40 edges short of the kept home fails `stopped <n> edges short of its curl end`. If it repeats, look for what holds the joint and check its hall cable |
| `<j> stroke <n> edges, outside 450-1500` | A hall knee's measured stroke is outside the bounds (printed from the constants) |
| `<j> stroke <n> edges, saved <m>: obstacle?` | A hall knee's stroke is under 85% of the saved one |
| `<j> partner not proven curled` | Guard only: the partner gate refused a move. It should not occur through the step table |

**Board operator lines (outside calibration)**

These go through the same output as the `CAL` lines (USB on the leader; a follower's go to the leader, which forwards them). None starts with `CAL `.

| Line | Meaning |
| :--- | :------ |
| `SENS <ROLE> <NAME>:<c> <NAME>:<c> ...` | Rev 3 (quadrature) builds only. Printed at boot, on any change and every 2 s, one token per actuator in index order. `<c>` is `?` unknown, `P` pot, `e` encoder not homed, `s` homed but no saved stroke, `x` home lost, `E` encoder ready (homed with a stroke). The host GUI reads this, so it survives a reconnect |
| `ERR STALL <name>` | The `T` stall guard stopped the joint at PWM 60 or more (an encoder also loses its home). The quiet stop at PWM 20-59 prints nothing |
| `ERR HOME <name> lost (x=<frame>)` | The range guard found the encoder frame clearly outside the stroke and cleared the home |
| `ERR T refused <name> ... (no position)` | A `T` target was refused for the named joints. At most once a second per board |
| `Calibration v2 loaded` | The v2 EEPROM record was valid and applied |
| `Calibration v1 loaded` | A v1 record was migrated (builds without quadrature halls: all pots) |
| `Calibration v1 ignored: run C` | A v1 record was ignored (quadrature builds) |
| `Calibration from a build without halls ignored: run C` | A v2 record with a pot yaw, written by a build without halls, was ignored (quadrature builds) |
| `Calibration record corrupt: defaults` | A v2 magic with a bad check or reserved bits; defaults are used |
| `Calibration joint <name> invalid: defaults` | That joint's stored values failed validation and were reset. Every hall hip record saved before this build (`[0, 700]`, or a measured stroke) is rejected this way: run `C` again. A hall hip with `hi` = 1069 also prints `Calibration joint <name> extend derived (1069 edges): not switch-verified` |
| `No calibration record: defaults` | Nothing usable in EEPROM |
| `Limits saved` | A SWEEP wrote the record (the `CAL <ROLE> PASS_DONE SWEEP saved` line confirms the same thing for every board) |

The host SDK logs `ERR ...` lines as errors and keeps the `SENS` states. It logs the EEPROM messages above (`Calibration ...`, `No calibration record: defaults`, `Limits saved`) at info level. A board prints its load line only at its own boot, so a follower's load line is seen only when that follower resets while the leader is connected; the `SENS` states (`?` after an ignored or missing record) show the result at any time.

**GUI**

The joint grid has a **Sensor** column (`POT`, `ENC`, `ENC unhomed`, `ENC no stroke`, `ENC home lost`, or `?`), and a banner `Encoder joints not ready: <names>. Run Calibrate > Curl only` shows while any hip or knee is `e`, `s` or `x`. Hips and knees at `?` (after a fresh flash, or with an ignored, corrupt or invalid record) are listed first as `Sensor not yet detected: <names>`. The banner says `full calibration` instead of `Curl only` when any listed joint is `s` or `?`, because Curl only never saves a stroke or a sensor kind. Neutral sends only to joints that have a position, and when it reaches none it names the same run as the banner. The **Hall** column is telemetry field 9, the **signed hall count** (see section 2.2).

**First-time measurement and acceptance (with you at the robot)**

The stroke bounds, the table-safe sweep values and the encoder gains are provisional: they rest on one leg (FLHL and FLKL, measured 2026-11-23, below). Do the measurement and the acceptance run yourself, with the GUI, and keep agents and scripts off the COM port.

 - **Measurement** (done 2026-11-23 on the front-left leg, robot on its stand, firmware before the fixed hip sweep, scripted jogs at PWM 80 ramped 0 to 80 in 0.32 s; repeat on other legs with the GUI jog, which runs at PWM 200 only, so there record the full stroke in edges and the edges per second at PWM 200; for timeouts at PWM 40 assume 10 edges/s for a hip). Results:
   - Knee FLKL, hip curled: full stroke **776 edges**; about 34 edges/s uncurling (retract) and 60 edges/s curling (extend) at PWM 80. No collision at full knee extension with the hip curled. The stroke bounds 450-1500 are about 0.6 and 2 times this one knee (provisional until more knees are measured).
   - Hip FLHL, knee curled: a full hip uncurl ends with the **curled tibia pressing up against the bottom of the robot** (one run was stopped by hand for it). The hip stopped at 941 edges with the knee 24 edges off its curl end and at 1003 edges with the knee curled tight: that is body contact, not the actuator's end. The first build swept 700 edges to clear the body; the table-safe sweep (840 edges with the knee parked 264 out, measured 2026-11-24 below) replaced it. Hip about 45 edges/s extending and 21-30 edges/s retracting; it returns exactly to its curl end (limit switch).
   - Knee curl end at PWM 80 varies from 3 to 24 edges with the approach (a full-stroke approach tapers and stops shallower). A PWM 200 curl (the GUI jog) parks the knee about 17 edges deeper than PWM 80 reaches.
   - Knee nudge from a curled knee: the first count came 0.36-0.77 s after the drive started (it has to break free), 12 edges were reached 0.67-1.0 s after the start, and after the drive stopped the knee sprang out 5-9 more edges (a 12-edge nudge ended 18-21 edges out). Return at PWM 80: 14 edges back from a knee parked at the PWM 80 end, only 5 from one parked by a PWM 200 curl (it stalls at the PWM 80 curl end).
   - Hip nudge from its curled end: 24 edges reached at 1.31 s; it returned exactly to its curl end.
   - Motor current (IS) at an end at PWM 80 is 8-14, the same as while moving (9-12): at PWM 80 the ends showed only in the counts.
   - Longest move at PWM 80: 36 s (the hip curling back from about 1000 edges), so the 120 s move timeout has margin. The slow rates logged earlier (only 10-20 edges/s for a hip at PWM 40-50) are history: PWM 40 and 80 are no longer accepted.
   - Hip extend end (2026-11-24, FL): the limit switch showed its signature 3 of 3, the current dropping from 11-12 to 7-8 at 1070 edges from the curl switch (stroke 1069, switch to switch). The PWM 80 mid-stroke stall at hip 841 held at 11, with no drop, and FRHL's 878 stall read 17-31: a stall never shows the switch-open level. At 866 edges a robot check put the toe about 5 in from the table side (about 8 in modelled), hence the 840-edge sweep. FL's knee parks 264 out (band 244-280); the knee's own switch is about 785 out.
   - Still unmeasured: the other legs (they use FL's values), the stall current at 150. Finish with each jogged joint back at its curl end.
 - **End-current measurement** (2026-11-24, front legs on the stand, firmware hallcal-1123, scripted jogs at PWM 80, 120 and 150; the `current` field is the smoothed IS reading, about 20 frames/s). The hip curl end and the knee uncurl end are the actuator's **limit switch**: the count stops dead and the current drops from its running level (FL hip 10-13, FR hip 15-17) to the joint's driven-but-switch-open level (FL hip 7-8, FRHL 11; knees 7.0 within 0.1 s), with almost no slowdown before. The knee curl end (rod extended into the leg's own stop, no switch) is a **compliant mechanical stop**: the speed collapses over the last 0.3-0.5 s while the current rises (FL +30%; FR +11% at 80, +60% at 120, +80% at 150), and harder pushes end deeper (FL 15/3/-5 edges at 80/120/150). Frame-to-frame noise is about +-3-5 counts, and running is only 2-8 counts above switch-open. At PWM 80 FLKL uncurls stalled about 100 edges out and FRKL stalled 18 short of its switch with flat current (a stall with no signature), and FRKL could not break free of its curl end at 80 or 100. Replayed through the end detector (40 moves): every switch end at 120/150 is found about 0.3-0.5 s after the count stopped (the detector now waits for six flat samples at the floor; the switch has already cut the motor) and recorded exactly there, every knee curl end 0-5 edges shallower than its final creep, no running move ever fires, and the FR knee curl at 80 (current +11%) is called stuck mid-stroke. Follower boards have not been traced yet: read the `by=` / `cur=` tokens of the END lines on the first full run.
 - **Acceptance** (after flashing all three boards together with this build; any hall hip record from an earlier build is rejected at boot with `Calibration joint <hip> invalid: defaults`, so the full calibration in step 4 is required):
   1. Boot: `SENS` shows `?` or `e` for the hips and knees, and with nothing moving the Hall counts stay at 0 (no stray +-1 on idle joints). On this first boot after the flash, while the hips and knees show `?`, the **Pot** column (hall B on Rev 3, with its pull-up now kept on) must read below 32 or above 990 on all 12 hips and knees: the pot classifier relies on hall B resting outside 32..990. Stop and report any joint that reads inside that band. Once a joint shows `e` its Pot cell is blank; its hall-B level is then only in field 3 of its telemetry segment (the `serial rx` lines of `python -m firmware.gui --debug`).
   2. Wait a few seconds after connecting (until `SENS` lines from LEFT and RIGHT show in the grid), then Curl only with the knees parked fully curled, and press Hold All during the first nudge (`MOVE FLKL nudge`): it stops at once (`CAL ROBOT ABORTED H`) and nothing moves after. A `CAL ROBOT FAIL <role>: no SENS: ...` line instead means that board has not been heard (not given time, no role elected, or other firmware): follow the troubleshooting in the `no SENS` row of the calibration line table (check that board for `ROLE: UNKNOWN`, reset all three together, reflash all three only if `SENS` is still missing).
   3. Curl only again, no jogging, knees still parked: each parked knee shows `MOVE <j> nudge`, ends about 20 edges out (the spring), curls back and is homed (`HOME <j>`), each parked hip the same (24 edges, exactly back), and every hip and knee reads `s` or `E`. No knee with a working hall goes out further than about 21 edges before its hip is homed (a knee with hall A dead runs blind to its 1 s cap, up to about 95 edges out at PWM 128 before its hip is proven curled, and fails `no feedback`). A knee that fails `curled back only <n> edges after its nudge` (parked at PWM 200, low spring-out) is accepted behaviour: run Curl only once more, without re-curling it, and it must pass.
   4. Full calibration on the stand: before it, jog every hip inside 0-560 and every knee to its curl end. Each leg runs the table-safe order: each knee prints its `KNEE ... travel=<S>` line (record all six: FL's is 769-801 when quiet) and parks 521 edges in from its switch (`PARK ... fromSwitch=521`, a few more after its coast), each hip goes out 840 edges and back with its knee parked and the toe never touches the table or the body (watch FLHL first, with Hold All ready), and each hip HOME follows a `by=switch` or `open=` END line. The hip JOINT lines read `min=0 max=1069 edges (extend derived from FL, not switch-verified)`, each knee stroke is plausible (about 785 edges; its travel window is 745-825), and the count sign holds on all 12 hips and knees (retract counts up).
   5. `T` to one joint, near its curl end. The GUI has no single-joint `T` control (its Neutral button sends 0.5 to every joint), so use a serial terminal on FRONT's USB port (250000 baud). **Opening a port resets the board and clears every home: run Curl only again after any port reopen** (in the terminal, `C SEEK`, and wait for `CAL ROBOT DONE`). Then send `T FLHL 0.1` (the hip about 107 edges out of its curl end), and after it settles `T FLHL 0.03` (about 32 edges out: targets stay 8 edges inside the hip's curl end); then `T FLKL 0.9` (the knee about 78 edges out of its curl end on a 776-edge stroke; a knee's curl end is `T 1`, which stops 25 edges short of the leg stop) and `T FLKL 1`. These targets start each drive with an error of about 60-80 edges, so it starts near the PWM 60-80 at which a stored joint breaks free (encoder `T` commands about 1 PWM per edge of error). Each must stop within about 20 edges of its target (the drive stops below PWM 20, which is 20 edges of error) with no `ERR STALL`, or be stopped quietly after 3 s with no `ERR` line (a small error commands PWM 20-59, which may not break a stored joint free). A quiet stop holds only that target: re-sending it keeps the joint stopped, while `H`, a jog or a different `T` re-arms it. Encoder `T` drives at up to PWM 150 (the calibration runs at PWM 128), so keep these first targets close to the curl end and Hold All (`H`) ready. Close the terminal and Curl only again from the GUI before step 6.
   6. Then Neutral for the whole robot. Neutral (`T 0.5`) sends every hall hip to about 534 edges (it was 350) and every knee to mid-stroke (about 392 edges out); a hip's `T` range reaches 8..1059. **On a solid stand keep every hip `T` at or below 560 edges** until the owner decides on a stand envelope (an open decision): nothing beyond about 864 is clear at any knee, and a curled tibia meets the body at 941-1003. Neutral's own 534 is inside that limit.

**Where the constants live**

All calibration constants, including the curl signs (`CAL_HIP_CURL_PWM_SIGN = -1`, `CAL_KNEE_CURL_PWM_SIGN = +1`), the speed limits, the hall thresholds, the nudge bounds, the stroke bounds and the option bits, are in `firmware/arduino/src/actuator/calibration_constants.h` (the encoder gains and `T` guards are in `actuator_constants.h`). If a leg ever curls the wrong way, change the sign there. The sensor classifiers are in `src/actuator/joint_sensor.h`. The per-board sequencer is `src/actuator/leg_calibration.h`, the leader's robot-level sequencer is `src/actuator/robot_calibration.h`, the sketch's command glue (the `C` dispatch, the H/J/B/T abort, the `V` guard; shared with the host sim) is `src/actuator/calibration_command_dispatch.h`, and the `C` parser is `src/cli/actuator_calibration_command.h`.

**GUI jog buttons**

The Joint GUI's **Retract** and **Extend** buttons send -PWM and +PWM respectively, matching the firmware. To curl a leg by hand, press Retract on the hip and Extend on the knee. A parked knee needs no jog before calibration: the run nudges it.

### Feature 2: Manual Jog Mode
 - Select Option 3 (Jog Mode).
 - Type the joint name (e.g., LHY or LKL).
 - Hold 'W' to Extend, Hold 'S' to Retract.
 - Release keys to stop immediately.

### Feature 3: Neutral Pose
 - Select Option 1 (menu key `0`), or the GUI **Neutral (0.5)** button.
 - Robot moves the hips and knees that have a position to center (0.5). Useful to verify calibration accuracy. Neutral sends every hall hip to about 534 edges and every knee to mid-stroke (about 392 edges out). **On a solid stand keep hip `T` at or below 560 edges** until the owner decides on a stand envelope (open decision): a hip `T` reaches 8..1059, nothing beyond about 864 is clear at any knee, and a curled tibia meets the body at 941-1003.
 - Yaws are never sent, and hips and knees without a position (encoders not homed after a power-up, or no saved stroke) are skipped and named. Run Curl only first (Feature 1), or the full calibration on the stand for joints at `s` or `?`; the message after Neutral names the run.

---

## 4. Firmware Store (`krabby-firmware-public`)

Built firmware lives in a public S3 bucket. CI publishes a new build on every push to `main` (dev channel) or `release/*` (stable), plus a daily scheduled build of the newest `release/*` branch.

### 4.1 Bucket layout

```
s3://krabby-firmware-public/
  index.json                               ← all branches, latest build per branch
  <branch>/latest.json                     ← pointer to the most recent build on <branch>
  <branch>/builds.json                     ← full build history for <branch> (powers `show <branch>`)
  <branch>/<YYYYMMDD-HHMMSS-<sha7>>/
    firmware.hex                           ← compiled Arduino HEX
    manifest.json                          ← branch, commit, timestamp, board FQBN, VER string
```

`<branch>` mirrors the Git branch name (`main`, `release/0.2.0`, etc.).

**`manifest.json` fields:** `schema_version`, `branch`, `commit`, `commit_date`, `build_timestamp`, `board_fqbn`, `ver_string`, `hex_filename`.

### 4.2 V protocol

Send `V\n` on the main serial (250000 baud). The leader board collects replies from all three boards and responds with a single line:

```
VER <versions> <branches> <commits>
```

Each field is `front|left|right` pipe-delimited. Example:

```
VER 0.2.0|0.2.0|0.2.0 release/0.2.0|release/0.2.0|release/0.2.0 abc1234|def5678|ghi9012
```

If a follower board is missing, its slot contains `-`.

### 4.3 Three-board update procedure

> **⚠ M16 firmware is a matched set — flash ALL THREE boards *and* update the
> host image in the same session.** M16 moved `BAUD_RATE` from 115200 to
> 250000 (see the serial-budget section under the M16 sensor cluster below),
> so a mixed fleet talks at mixed bauds: the symptom is garbage characters or
> no telemetry at all on the host. A partial reflash also breaks role
> election — boards on different bauds can't hear each other, so every board
> times out and boots `ROLE_UNKNOWN` (front actuator map), including the ones
> wired as LEFT/RIGHT. Do not stop halfway through step 3.

```bash
# 1. One-time host setup (udev rules, dialout group, flash tools)
sudo krabby-firmware install

# 2. Check attached boards and the latest build per branch
krabby-firmware show

# 2b. List one branch's full build history, newest-first (paged via $PAGER)
krabby-firmware show release/0.2.0

# 3. Flash all three boards in turn (replug USB between boards —
#    LEFT/RIGHT need USB for flash today; see note above about future leader-driven flash)
krabby-firmware update                        # latest release/* build, auto-detects port
krabby-firmware update release/0.2.0          # specific branch
krabby-firmware update /dev/ttyACM1           # specific port, latest release
krabby-firmware update release/0.2.0 /dev/ttyACM2  # specific branch + port
```

Flash all three boards together before running calibration: a mixed set runs the old pot-only routine on the hall joints of an old follower. A Rev 3 leader refuses the run while a linked follower has sent no `SENS` line (`CAL ROBOT FAIL <role>: no SENS: ...`). After flashing, run `C` on the stand again: hall hip records from earlier builds are rejected.

Do not interrupt mid-update. If the serial device disappears afterward, follow **Board vanishes during or after upload** above before assuming the board is bricked.

Downloaded HEX files are cached under `~/.cache/krabby-firmware/<branch>/<sha7>/firmware.hex` and reused on subsequent calls.

### `krabby-firmware` vs `krabby firmware`

Two ways to reach the same flash CLI:

- **`krabby-firmware <args>`** — runs the flash tool **directly on the host**. Requires the
  `krabby-firmware` package and host flash tools (`krabby-firmware install` sets up
  `avrdude`/`arduino-cli`, udev, and `dialout`). Use this on a laptop or bench machine.
- **`krabby firmware <args>`** — runs that same CLI **inside the locomotion image** (the
  flash tools are bundled there), so a kit owner who only `pip install krabby-launcher`
  can flash with no host setup. It forwards every argument verbatim, mounts the
  `~/.cache/krabby-firmware` download cache, and passes the serial devices through.

So `krabby firmware show release/0.2.0` and `krabby-firmware show release/0.2.0` behave
identically — they differ only in *where* the tool runs.

---

## I2C Sensor Cluster (Milestone 16) — LSM6DSO IMU

The **leader board only** (role `FRONT` after election, or the solo-board `UNKWN`
bench case) carries a shared I2C bus on the Mega's hardware I2C pins. Followers
never initialize the bus. Device-specific bus constants and the concrete adapter
live in `arduino/src/imu/lsm6dso_adapter.h`.

Naming note — three spellings, one state: `UNKWN` is the telemetry **wire
prefix** the firmware actually emits for an un-elected role, `ROLE: UNKNOWN
(front actuators)` is the same state in the **boot log**, and the `UNKNOWN` in
§2.2's role-prefix list is the long-form name of that prefix slot. All three
mean "no role elected; front actuator map assumed".

### Wiring (SparkFun 6DoF LSM6DSO Qwiic breakout, via Qwiic→Dupont adapter)

| Qwiic wire | Mega pin | Note |
| :--- | :--- | :--- |
| VCC (red) | **3.3V** | LSM6DSO is a 3.3 V part (1.71–3.6 V) — never 5 V |
| GND (black) | GND | |
| SDA (blue) | **D20** | Mega hardware I2C SDA |
| SCL (yellow) | **D21** | Mega hardware I2C SCL |

- I2C address **0x6B** (0x6A with the ADR/SA0 jumper cut). Bus runs at **100 kHz**
  for noise margin; the payload per 50 ms telemetry tick is tiny.
- Later sensors (Qwiic OLED, INA228 ×2) daisy-chain on the same bus.

### Telemetry segment

The leader appends one segment to its own telemetry line (append-only; old
parsers drop it — see `firmware/interfaces/joint_telemetry.py`):

```text
;IMU <accel_x> <accel_y> <accel_z> <gyro_x> <gyro_y> <gyro_z> <temp_c> <valid>
```

Units: accel **m/s²**, gyro **rad/s** (gyro is boot-bias-subtracted), temp **°C**.
`valid` is `0` when the sensor did not respond that tick (init failure ships
zeros with `valid=0` and never stalls the gait loop). The LSM6DSO driver reads
acceleration, angular rate, and temperature as one sample; it does not expose
an independent temperature-read failure. The Python parser nevertheless
preserves finite motion data if a future driver reports a non-finite
temperature, displaying that temperature as `nan`.

### Axis convention / sensor→body transform

Reported acceleration and angular-rate vectors use the robot body frame:

- **+X points forward** toward the front of the robot.
- **+Y points left** when viewed from above.
- **+Z points up**, opposite gravity while the robot is upright.

`body[i] = IMU_AXIS_SIGN[i] * sensor[IMU_AXIS_SRC[i]]`
(`src/imu/imu_constants.h`).
**Currently identity** — the breakout's mounting orientation is not final.
Update the constants and this section together when the mount is fixed.

### Boot calibration (EEPROM)

Gyro zero-rate bias is captured at first boot while the robot is stationary
(200 samples, ~1 s), persisted to EEPROM, and reloaded on every subsequent
boot. If motion is detected during capture (gyro spread >
`IMU_CAL_MAX_SPREAD_DPS`), nothing is saved and the capture retries on the
next boot.

The boot log is the operator's gate that calibration succeeded: `IMU CAL:
gyro bias captured and saved to EEPROM` (or, on later boots, `loaded from
EEPROM`) means the bias is in effect. An aborted capture (`motion detected`)
is **not** visible on the wire — the board ships bias-uncorrected gyro with
`valid=1` until a stationary reboot completes a capture. A wire-visible
cal-status field is a Task-3 wire-budget discussion item.

To force a re-capture, invalidate the magic byte with a throwaway sketch:

```cpp
#include <EEPROM.h>
void setup() { EEPROM.write(40, 0x00); }  // 40 = EEPROM_IMU_CAL_ADDR
void loop() {}
```

There is deliberately **no serial command** for this: a command that wipes
EEPROM could be triggered by line noise on the serial link, and a
noise-triggerable calibration wipe is a worse hazard than a bench chore (see
hazard issue #2).

Two terms, defined once: the **magic byte** is a sentinel value (`0xC7` here)
whose only job is to prove this EEPROM region was ever written by this
firmware — a factory-fresh AVR reads `0xFF` at every EEPROM address, so
anything other than the expected magic means "no calibration stored; capture
one". The **schema byte** is a layout version number: if a future firmware
changes the field layout of `ImuCalibrationRecord`, it bumps the schema, and old data
is rejected as stale instead of being silently misread field-by-field.

Full EEPROM map after M16 Task 1. Every address below is a byte offset into
the 4 KB EEPROM, ranges inclusive. Bytes 0–33 are the pre-existing layout
(same as the "EEPROM address layout" table earlier in this file); M16 adds
only bytes 40–65. Constants live in `src/imu/imu_constants.h`; the
`ImuCalibrationRecord` struct lives in `src/imu/imu_calibrator.h`.

| Bytes | Size | Owner | Contents |
| :--- | ---: | :--- | :--- |
| 0–29 | 30 | Joint calibration (`CalData` v2) | per-actuator min/max (int16) for 6 actuators, magic `0xCA2B` at 24, sensor kinds at 26, checksum at 28 |
| 30–31 | 2 | — | unused (alignment gap) |
| 32 | 1 | Role election (pre-existing, M14) | role magic sentinel `0xAB` |
| 33 | 1 | Role election (pre-existing, M14) | `BoardRole` value (1=FRONT, 2=LEFT, 3=RIGHT) |
| 34–39 | 6 | — | unused gap left before the M16 block |
| 40 | 1 | `ImuCalibrationRecord.magic` | `0xC7` (`EEPROM_IMU_CAL_MAGIC`) |
| 41 | 1 | `ImuCalibrationRecord.schema` | layout version, currently `1` (`EEPROM_IMU_CAL_SCHEMA`) |
| 42–53 | 12 | `ImuCalibrationRecord.gyroBiasDegreesPerSecond[3]` | 3 × 4-byte float; gyro zero-rate bias, deg/s, raw sensor frame |
| 54–65 | 12 | `ImuCalibrationRecord.accelBiasG[3]` | 3 × 4-byte float; reserved accelerometer offset, g, raw sensor frame; zero until accelerometer calibration is implemented |
| 66– | — | free | `EEPROM_SENSOR_CAL_NEXT_ADDR` = 66; Task 3 (INA228 cal) and later blocks allocate from here, each with its own magic + schema |

So "`ImuCalibrationRecord` is 26 bytes" means exactly bytes 40–65:
1 (magic) + 1 (schema) + 12 (gyro bias) + 12 (accel bias) = 26. The
test-only EEPROM layout contract pins this stored shape to
`EEPROM_IMU_CAL_SIZE` and verifies that it cannot overlap the joint, role, or
next-sensor regions.

### Loop timing (AC 1c) and serial budget

**Motivation:** the IMU adds two per-tick costs — blocking I2C time inside
the 50 ms telemetry tick, and extra bytes on a serial link that was already
near saturation at 115200 baud — so this section derives both costs from the
code, then checks the arithmetic against bench measurements.

Per-tick I2C cost added by the IMU comes from one coherent 14-byte LSM6DSO
output-register burst. `Lsm6dsoAdapter::read()` writes the starting register
with a repeated start, then reads temperature, angular rate, and acceleration
in one transaction. One wire byte is 8 data bits plus 1 ACK:

| Transaction | Wire bytes | Time |
| :--- | ---: | ---: |
| Set output-register pointer (addr+W, reg) | 2 | 180 µs |
| Read 14-byte output sample (addr+R, data) | 15 | 1350 µs |
| **Wire-time floor per tick** | **17** | **1530 µs** |

The figures use the configured 100 kHz bus and exclude START/STOP and AVR Wire
ISR overhead. `I2C_BUS_TIMEOUT_MICROSECONDS` bounds a stalled transfer at 10 ms.
Current-device timing remains a bench measurement rather than a claim derived
from this wire-time floor.

The `;IMU` segment adds **49–63 characters** to the leader's line: exactly 49
on the all-zeros `valid=0` path (which is the measured 229 − 180 = 49 below),
up to 63 with every field at its widest (accel `-78.453` at the ±8 g default
range, gyro `-69.8132` worst case, temp `-41.0`). The field-by-field
derivation of the 49 and 63 figures (float-formatting rules and the value
bound behind each width) lives in `docs/M16-DESIGN-DECISIONS.md` §1.1.1 (PR #3, branch m16-docs); the
I2C wire math is the table above.

At the old 115200 baud the upstream link budget was 11520 B/s × 50 ms =
576 B per tick. Three lines per tick at the bench-measured 180 B = 540 B
(94%) before M16; adding the 49 B IMU segment makes 589 B (102%) — over
budget, with the blocking `flush()` stretching the tick and the hall
counters growing every line. `BAUD_RATE` is now **250000** (exact 0%-error
divider on the 16 MHz Mega), giving 1250 B/tick — 589 B nominal = 47%
utilization (per-field line derivation in the byte-accounting comment next
to `TELEMETRY_LINE_MAX` in `arduino.ino`). Host-side defaults in
`krabby_mcu.py`, `gui/app.py`, `gui/__main__.py`, the `cli.py` V-probe, and
the Jetson HAL (`hal/server/jetson/krabby_mcusdk.py`, `hal_server.py`)
match; the avrdude *bootloader* baud (Makefile / `cli.py` flash path) is
separate and stays 115200. Because of this baud change, M16 firmware must be
deployed to all three boards and the host together — see the boxed warning in
§4.3.

Bench evidence — captured 2026-07-06 on a solo Mega 2560 R3 (ROLE_UNKNOWN
bench leader, 400 lines per row, host-side inter-line arrival timestamps):

| Build | line len (B) | mean tick (ms) | p95 (ms) | max (ms) |
| :--- | :--- | :--- | :--- | :--- |
| upstream/main @ 115200 | 180 | 50.72 | 53.29 | 57.09 |
| M16 Task 1 @ 250000, IMU absent (valid=0 path) | 229 | 50.77 | 53.38 | 58.84 |
| M16 Task 1 @ 250000, IMU attached | 234 | 51.02 | 53.08 | 59.38 |

The IMU-attached row was captured 2026-07-14 using the superseded BMI270
prototype. It is retained only as historical timing evidence and does not
validate the current LSM6DSO hardware. Current-device timing remains a bench
validation item.

Delta with the IMU segment added: **+0.05 ms mean** — inside run-to-run
noise, satisfying "no measurable change to loop timing" for the serial path.
The IMU-attached row (adds the live ~4 ms I2C read inside the tick) and a
full three-board `tests/integration/test_timing.py` run are captured at
robot integration.

### Bench bring-up runbook (M16, solo board)

> Formal ATP-style test procedures (with run logs and an AC traceability matrix) live in `firmware/bench_tests/INDEX.md` (PR #3, branch m16-docs); this runbook is the narrative version.

Replicated 2026-07-06 at a café table. Everything below assumes the repo venv
(`testenv`) has `pyserial`, and `PORT` = the board's device (macOS:
`ls /dev/cu.usbmodem*`; if nothing appears but the board is powered, check the
"Allow accessory to connect" gate in System Settings → Privacy & Security —
the Mega enumerates but gets no serial driver until allowed).

1. **Voltage check (before first sensor connect).** Meter probes don't fit
   female headers: plant two M-M Dupont jumpers in `3V3` and `GND` and probe
   their free ends (don't let them touch). Expect 3.30 ± 0.1 V. The LSM6DSO is
   **not 5 V tolerant** — this check is the one that saves the sensor.
2. **Wire (USB unplugged).** Qwiic→Dupont: black→GND, red→3V3, blue→D20 (SDA),
   yellow→D21 (SCL). Either Qwiic jack on the breakout works.
3. **Flash + watch boot.** `make -C firmware upload-firmware PORT=$PORT`, then
   `python firmware/scripts/imu_bench.py $PORT watch`. Expected boot on a solo
   board: `ROLE: UNKNOWN (front actuators)` (the bench-leader case), then
   `IMU CAL: LSM6DSO online at 0x6B` (or `0x6A` if the ADR/SA0 jumper is cut);
   firmware probes 0x6B then 0x6A. First
   boot: `gyro bias captured and saved to EEPROM` (board must sit still ~1 s;
   `motion detected` means it retries next boot). Later boots:
   `loaded from EEPROM`.
4. **Verify.** At rest `|accel| ≈ 9.81 m/s²` and gyro ≈ 0 (bias-subtracted).
   Then `imu_bench.py $PORT flip` — flip the **breakout board itself** (not
   the Mega; the sensor is the thing at the end of the cable) upside down and
   hold ~10 s: PASS requires inverted samples. The mode exists because a
   remote-guided test needs a *confirmed* physical action — assume nothing.
5. **Timing evidence (AC 1c).** `imu_bench.py $PORT timing` with the board
   still; numbers land in the table above.
6. **Bus debugging ladder** (when init fails): flash
   `bench_sketches/i2c_scanner` — idle SDA/SCL must both read 1; a found
   address ≠ expected means jumper strap; found-but-init-fails means driver
   timing/data (see patches 2 and 4 in the fetched-libraries section below).
   Reflash real firmware afterwards.

Note: opening the serial port resets the board (macOS pulses DTR on open
regardless of pyserial settings) — every capture in `imu_bench.py` waits
through the ~4 s boot for this reason. Leader boot blocks ~1.2 s for IMU init
(2.9 s worst case when a bias capture runs), which the SDK's 5 s post-connect
sleep already covers. `krabby_mcu.connect()` avoids the
reset with its pre-open `dtr = False` on Linux/Jetson, but macOS resets anyway.

### Fetched libraries

The current M16 build fetches the pinned, upstream-clean SparkFun LSM6DSO
library declared in `scripts/fetch_arduino_libs.py`. `make` and CI pass the
materialized library directory to `arduino-cli`; Arduino IDE users must expose
that same directory through their sketchbook rather than installing an
uncontrolled Library Manager version.

#### Historical: superseded BMI270 AVR integration

The remainder of this subsection records the retired BMI270 prototype and is
not a setup procedure for current M16 hardware. Do not install or wire a BMI270
for M16; the authoritative current procedure is the LSM6DSO runbook above.

Third-party Arduino libraries are **not committed**. `make compile-firmware`
(and CI) first runs `scripts/fetch_arduino_libs.py`, which downloads the
pinned upstream release (SparkFun BMI270 `v1.0.3` =
`21ea234de321da07c552f7a43cb36f7df4f73a27`, MIT), verifies the archive's
SHA-256, unpacks it into the gitignored `arduino/libraries/`, and applies the
committed delta `arduino/patches/SparkFun_BMI270_Arduino_Library.patch`. The
first fetch needs network once (~2.7 MB); every later build is offline (a
stamp file guards re-fetching, and a pin or patch change invalidates it).
Design rationale and alternatives: `docs/M16-DESIGN-DECISIONS.md` §2.1 (PR #3, branch m16-docs).

The patch carries four AVR fixes, all tagged `Krabby patch` in-source:

1. **PROGMEM config blob** — Bosch's ~8 KB config blob is declared `PROGMEM`
   in `bmi270.c` (**the only Bosch-API-file change**; `bmi2.c` is pristine).
   The API never dereferences the config bytes itself — they flow only through
   the user write callback — so the flash-aware read lives in the wrapper's
   own `writeRegistersI2C/SPI` (`readDataByte`, keyed on
   `BMI2_INIT_DATA_ADDR`, the one register the API sources from the config
   file). Unpatched, the blob lands in SRAM (8 KB total on the Mega) and the
   build is rejected by the toolchain's size check ("data section exceeds
   available space in board" — linking itself succeeds). Restructured
   2026-07-13 from a `memcpy_P` staging block inside `bmi2.c:upload_file` to
   this form, to keep the Bosch-file modification to a single declaration
   (matches the upstream PR; the sensor sees identical bus bytes).
2. **Config chunk size ≤ 30** — `sensor.read_write_len` lowered 32 → 30 in
   `BMI270::begin()` (`SparkFun_BMI270_Arduino_Library.cpp`). Each config
   chunk is one I2C write of 1 register byte + N data bytes against AVR
   Wire's 32-byte TX buffer, so N=32 silently drops the last byte of every
   chunk and config load fails. N must be even and ≤ 30 (32-byte buffer − the
   register byte = 31; Bosch's half-word config indexing tightens it to 30).
   An exact divisor of 8192 is **not** required — `write_config_file` handles a
   ragged final chunk in 2-byte writes — so 30 (the largest legal chunk, matching
   Arduino's own `Arduino_BMI270_BMM150`) is used over 16 to roughly halve the
   one-time upload transaction count (512 → 274 chunks) at `begin()`.
3. **Short-read detection** — `readRegistersI2C` fails on a short
   `requestFrom()` instead of returning OK with a stale buffer
   (`SparkFun_BMI270_Arduino_Library.cpp`).
4. **usDelay overflow** — AVR's `delayMicroseconds()` is only valid to
   16383 µs; `bmi270_init`'s config-load validation waits 20 ms in a single
   `delay_us(20000)` call, which overflows, so the bare upstream call makes
   `bmi270_init` fail with `BMI2_E_CONFIG_LOAD` (-9) on every boot. (The 51 ms
   delay some driver paths use is in `bmi2_perform_accel_self_test`, which
   `begin()` never calls — the init killer is the 20 ms wait.) The patch splits the
   wait into `delay(ms)` + `delayMicroseconds(remainder)`. Found on the
   bench 2026-07-06: the sensor ACKed and returned its chip ID, config bytes
   round-tripped perfectly, and init still failed — the last suspect standing
   was time itself.

`make`/CI builds always use the fetched, patched copy — arduino-cli's
`--libraries` outranks sketchbook libraries — so a globally installed
upstream copy cannot shadow it there. The **Arduino IDE** is the exception:
it does not scan sketch-local `libraries/`, so an IDE build fails to find the
header. To build from the IDE, materialize the library once, then symlink it
into your sketchbook:

```sh
python3 scripts/fetch_arduino_libs.py   # one-time; make compile-firmware also runs this
ln -s "$(pwd)/arduino/libraries/SparkFun_BMI270_Arduino_Library" ~/Documents/Arduino/libraries/
```

On Windows, use a directory junction instead of the symlink:
`mklink /J "%USERPROFILE%\Documents\Arduino\libraries\SparkFun_BMI270_Arduino_Library" "arduino\libraries\SparkFun_BMI270_Arduino_Library"`
(run from `firmware/` in `cmd`; junctions need no admin rights).

**Never install this library via the Arduino Library Manager.** That installs
the unpatched upstream version, whose ~8 KB config blob overflows the Mega's
SRAM (build fails its size check) — and if both copies are present the Library Manager copy
can shadow the symlinked, patched one in IDE builds.

Otherwise prefer `make compile-firmware` / `make upload-firmware`.

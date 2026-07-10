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
  - Libraries: `pip install pyserial keyboard`
  - Arduino IDE

---

## 1. Hardware Wiring (Rev 3 — Krabby Uno v0.2)

**Polarity Note:**
* **RPWM / R_EN** = Right (Extend/Forward).
* **LPWM / L_EN** = Left (Retract/Reverse).

| Board     | Joint         | PWM (R, L)       | EN    | Potentiometer | Current Sense | HallA  |
| :-------- | :------------ | :--------------- | :---- | :------------ | :------------ | :----- |
| **FL**    | Yaw (LHY)     | D2, D3           | D22   | A0            | A6            | D50    |
|           | Hip (LHL)     | D4, D5           | D24   | A1            | A7            | D51    |
|           | Knee (LKL)    | D6, D7           | D26   | A2            | A8            | D52    |
| **FR**    | Yaw (RHY)     | D8, D9           | D23   | A3            | A9            | A12    |
|           | Hip (RHL)     | D10, D11         | D25   | A4            | A10           | A13    |
|           | Knee (RKL)    | D12, D13         | D27   | A5            | A11           | A14    |

**Note:** Ensure all Enable (EN) pins are connected and driven HIGH when driving, otherwise calibration will get 'lost' as it will not know where joint positions are.

---

## 2. Installation

### 2.1 Serial RX buffer (leader board, 3-board setup)

When using the **leader** board that forwards telemetry from left/right followers, the default 64-byte serial RX buffer can overflow and drop bytes (corrupt or missing actuators in telemetry). Use a **256-byte** RX buffer for Serial2/Serial3 on the leader.

The Makefile passes this define on every build, so you usually don't have to do anything. `make compile-firmware` / `make upload-firmware` bake `-DSERIAL_RX_BUFFER_SIZE=256` into the `arduino-cli compile` invocation unconditionally (see `firmware/Makefile` `BUILD_PROPS`), exactly as CI (`.github/workflows/publish-firmware.yml`) does. `firmware/install.py`'s `platform.local.txt` write is a **belt-and-suspenders backup for IDE builds, not a requirement** — a `make`-built or CI-built binary already has the 256-byte buffer regardless of whether `install.py` ran or which AVR core version is installed. (Some core versions, e.g. 1.8.7, already default the Mega's RX buffer to 256; passing the define guarantees it on every core version and board variant.)

> **This define was the *hypothesized* prime suspect for the primary↔follower comms failure — not the actual bench bug.** On the deployed core (`arduino:avr` 1.8.7) it is a no-op (the Mega already defaults to a 256-byte RX buffer), so it is kept as defensive hygiene and CI parity across other cores/board variants, not as the fix. The failures actually hit on the bench were a firmware **floating-RX starvation** bug and **wiring** faults. See **[`COMMS_DEBUG.md`](COMMS_DEBUG.md)** for the staged root-cause analysis, captured logs, and the repro.

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
- **Segment format:** Each joint segment is 9 space-separated values: joint name, position (0–1), pot raw, current raw, enable L/R, PWM L/R, safety.
- **Example:** `FRONT; FLHY 0.723 740 694 0 0 0 0 0;FLHL 0.723 740 691 ...`

On the Arduino side, telemetry is built in **telemetry_manager.h** (struct `JointTelemetry`, `appendTo()`). The old standalone `joint_telemetry.h` was removed; all telemetry formatting and collection lives in `telemetry_manager.h` and `actuator_manager.h`.

### 2.3 Command protocol (host → firmware)

Commands are **newline-terminated lines** sent to the main serial (115200 baud). The first byte selects the command; the dispatch lives in `arduino/arduino.ino` `loop()`. The leader forwards every command down `leftSerial`/`rightSerial` to the follower boards. A leading `S` or `G` is read as a whole line and dispatched as a multi-letter config command (`SET…`/`GET…`, see §3 "Board roles"); any other unknown leading byte is discarded as line noise, one byte at a time (see the floating-RX guard comment in `loop()`).

| Cmd | Format | What it does | Python SDK sender (`krabby_mcu.py`) |
|-----|--------|--------------|-------------------------------------|
| `T` | `T <name> <pos> [<name> <pos> ...]` | Closed-loop position targets (0–1 per joint); parsed by `parseCommands` (`command.h`), applied by each board's actuator manager | `send_command_joints` |
| `B` | `B <name> <pwm> [<name> <pwm> ...]` | Batch jog — multiple joints at raw PWM (−255 to 255) in one line | `send_commands_jog` |
| `J` | `J<name> <pwm>` (no space after `J`) | Single-joint jog at raw PWM (−255 to 255) | `send_command_jog` |
| `C` | `C` | Start the auto-calibration sequence | `send_command_calibrate` |
| `H` | `H` | Hold all joints at their current position | `send_command_joints_hold` |
| `V` | `V` | Version query — leader collects follower versions and replies with a single `VER` line (see §4.2) | `read_version` |
| `SET` | `SET <key> <val> [<key> <val> ...]` | Write config (role, serial) to EEPROM on the receiving board; fire-and-forget, no reply | `send_set` |
| `GET` | `GET <key> [<key> ...]` | Read config; replies `GET <key> <val> …` (keys: role, serial, version) | `send_get` |
| `SET_LEFT` / `SET_RIGHT` | same payload as `SET` | Front-only: strips the suffix and relays the bare `SET …` to the LEFT/RIGHT follower over Serial1/Serial2 | `send_set(board="left"/"right")` |
| `GET_LEFT` / `GET_RIGHT` | same payload as `GET` | Front-only: relays `GET …` to the follower, reads its reply, re-tags it `GET_LEFT …`/`GET_RIGHT …` on USB | `send_get(board="left"/"right")` |

### 2.4 Pin revisions (`KRABBY_PIN_REV`)

Wiring is selected at **compile time** in **`arduino/board_pins.h`** (`#define KRABBY_PIN_REV`, default **3**). Rev **3** matches **`MOTOR_HEADER_PINOUT.md`**.

| | **Rev 3** (default, Uno v0.2) | **Rev 2** (Uno v0.1) | **Rev 1** (original) |
|---|---|---|---|
| PWM | D2-D13 | D2-D13 | D2-D13 |
| FL EN (LHY / LHL / LKL) | D22 / D24 / D26 | D22 / D23 / D24 | D22 / D23 / D24 |
| FR EN (RHY / RHL / RKL) | D23 / D25 / D27 | D28 / D26 / D27 | D28 / D26 / D27 |
| HallA1-6 | D50, D51, D52, A12, A13, A14 (PCINT0+2) | none | D37, D36, D35, D32, D33, D34 (PCINT1) |

- **Arduino IDE:** open **`firmware/arduino/arduino.ino`**, set **Board → Arduino Mega 2560**, choose the correct **Port**, set **`KRABBY_PIN_REV`** in **`board_pins.h`** if needed, then **Upload**. The serial monitor at **115200** baud should show **`PINS_REV3_UNO_V02`** (or the matching label) after reset.
- **Make + arduino-cli:** install [arduino-cli](https://arduino.github.io/arduino-cli/latest/installation/) and **GNU Make**. On Windows: `winget install GnuWin32.Make` then add **`C:\Program Files (x86)\GnuWin32\bin`** to your **`PATH`**. Put **arduino-cli** on your **`PATH`** (or set **`ARDUINO_CLI`**). Install **pyserial** for port auto-detect: `pip install -r firmware/requirements.txt`. From **`krabby-research`**:
  - `make -C firmware upload-firmware` — auto-detects serial port via **`firmware/mcu_port.default_port()`**. Pass **`PORT=COM5`** (or `/dev/ttyACM0`) to override.
  - Other revisions: `make -C firmware upload-firmware PIN_REV=1` (or `PIN_REV=2`).
  - Compile only: `make -C firmware compile-firmware`.
  - See **`firmware/Makefile`** for **`ARDUINO_CLI`**, **`FQBN`**, **`PIN_REV`**.

Flash each Mega with the image that matches **that** board’s wiring. All three boards use the same sketch; the board's role comes from EEPROM (`krabby-firmware set role=…`, see §3 "Board roles").

### 2.5 Python SDK

1. From **`krabby-research`**, install dependencies: `pip install -r firmware/requirements.txt`.
2. Ensure **`firmware/interfaces/`** is importable (e.g. run **`python -m firmware`** from **`krabby-research`** as in §3).

---

## 3. Usage Guide

Run the interactive MCU menu from the **krabby-research** directory:

```bash
# On Linux/Mac, you may need sudo for keyboard access
python -m firmware
```

For troubleshooting (verbose telemetry):
```bash
python -m firmware --debug
```


### Board roles (`set` / `get`)

The three boards run the same firmware; a board's **role** selects which 6 of the 18 joints it drives — `FRONT`, `LEFT`, or `RIGHT`. Each board reads its role from EEPROM at boot and keeps it across power cycles. A board with no role set (e.g. freshly flashed) comes up `UNKNOWN`: it drives no actuators but still answers `set`/`get`, so you can assign it.

`set` writes one or more `key=value` pairs (and reads them back to confirm); `get` reads one or more keys. Allowed keys: **`role`** (`FRONT` / `LEFT` / `RIGHT` / `UNKNOWN`) and **`serial`** (a short per-board identifier); `get` additionally accepts read-only **`version`**. There are two ways to say *which* board:

**Bench — `--port` (each board directly).** With all three Megas on a USB hub, address each one by its serial port:

```bash
krabby-firmware set --port /dev/ttyUSB0 role=FRONT
krabby-firmware set --port /dev/ttyUSB1 role=LEFT  serial=LEF-0007
krabby-firmware set --port /dev/ttyUSB2 role=RIGHT
krabby-firmware get --port /dev/ttyUSB1 role serial    # -> role=LEFT  serial=LEF-0007
```

(`--port` defaults to auto-detect, or `$KRABBY_MCU_PORT`.)

**Deployed robot — `--board` (through the FRONT board).** On the assembled robot only the FRONT board is on USB; the LEFT and RIGHT followers connect to it over the inter-board serial links (FRONT `Serial 1` → LEFT, `Serial 2` → RIGHT). Configure and read the followers *through* FRONT with `--board`:

```bash
krabby-firmware set role=FRONT                 # the board on USB
krabby-firmware set --board left  role=LEFT
krabby-firmware set --board right role=RIGHT
krabby-firmware get --board left  role serial  # -> role=LEFT  serial=…
```

`set --board left` forwards a bare `SET …` out FRONT's `Serial 1` to the LEFT follower; `get --board left` forwards a `GET …` and relays the follower's reply back, re-tagged so the host knows the source. Because roles persist in EEPROM, you can equally assign all three on the bench by `--port` and they'll come up correctly once deployed — `--board` is for configuring or reading the followers in place. To check a role stuck, power-cycle and `get` again.

Each board prints `ROLE_HINT: <role>` at boot, which `krabby-firmware show` uses to label each port — so a board probed on its own port is identified by its role.

To validate the whole role/config/motion stack on the rig in one pass, run the bench suite: `python3 -m firmware.tools.bench_suite` (see its docstring for phases and flags; `--skip-motion` for config-only).

### EEPROM layout

Board configuration lives in a single `EepromLayout` struct at EEPROM address 0 (defined in [`firmware/arduino/eeprom_layout.h`](arduino/eeprom_layout.h)). It is validated on load by a magic word, a schema version, and a CRC32, so a blank or corrupt EEPROM reads back as `UNKNOWN` rather than a garbage role.

| Field | Type | Purpose |
|-------|------|---------|
| `magic` | `uint16` | `0x4B17` when valid |
| `schema_version` | `uint8` | identifies the struct layout |
| `role` | `uint8` | `0`=UNKNOWN, `1`=FRONT, `2`=LEFT, `3`=RIGHT |
| `serial` | `char[16]` | zero-padded ASCII; empty if unset |
| `crc32` | `uint32` | checksum over all preceding fields |

Per-joint calibration is planned to live in this same struct as an added field; until then, auto-calibration results are not persisted (see Feature 1).

### Feature 1: Auto-Calibration
The robot finds its joint limits automatically.
 - Select Option 2 (Auto-Calibrate) in the menu.
 - Stand Back: The robot will perform the safety sequence:
    - Yaw Left -> Yaw Right -> Hip Up -> Knee Out -> Knee In -> Hip Down.
 - Result: the joint limits are found and used for the current session. They are **not persisted**, so re-run auto-calibration after each power cycle.

### Feature 2: Manual Jog Mode
 - Select Option 3 (Jog Mode).
 - Type the joint name (e.g., LHY or LKL).
 - Hold 'W' to Extend, Hold 'S' to Retract.
 - Release keys to stop immediately.

### Feature 3: Neutral Pose
 - Select Option 1.
 - Robot moves all joints to center (0.5). Useful to verify calibration accuracy.

---

## 4. Firmware Store (`krabby-firmware-public`)

Built firmware lives in a public S3 bucket. CI publishes a new build on every push to `mainline` or `release/*`.

### 4.1 Bucket layout

```
s3://krabby-firmware-public/
  index.json                               ← all branches, latest build per branch
  <branch>/latest.json                     ← pointer to the most recent build on <branch>
  <branch>/<YYYYMMDD-HHMMSS-<sha7>>/
    firmware.hex                           ← compiled Arduino HEX
    manifest.json                          ← branch, commit, timestamp, board FQBN, VER string
```

`<branch>` mirrors the Git branch name (`mainline`, `release/0.2.0`, etc.).

**`manifest.json` fields:** `schema_version`, `branch`, `commit`, `commit_date`, `build_timestamp`, `board_fqbn`, `ver_string`, `hex_filename`.

### 4.2 V protocol

Send `V\n` on the main serial (115200 baud). The leader board collects replies from all three boards and responds with a single line:

```
VER <versions> <branches> <commits>
```

Each field is `primary|left|right` pipe-delimited. Example:

```
VER 0.2.0|0.2.0|0.2.0 release/0.2.0|release/0.2.0|release/0.2.0 abc1234|def5678|ghi9012
```

If a follower board is missing, its slot contains `-`.

### 4.3 Three-board update procedure

```bash
# 1. One-time host setup (udev rules, dialout group, flash tools)
sudo krabby-firmware install

# 2. Check attached boards and available S3 builds
krabby-firmware show

# 3. Flash all three boards in turn (replug USB between boards)
krabby-firmware update                        # latest release/* build, auto-detects port
krabby-firmware update release/0.2.0          # specific branch
krabby-firmware update /dev/ttyACM1           # specific port, latest release
krabby-firmware update release/0.2.0 /dev/ttyACM2  # specific branch + port
```

Downloaded HEX files are cached under `~/.cache/krabby-firmware/<branch>/<sha7>/firmware.hex` and reused on subsequent calls.
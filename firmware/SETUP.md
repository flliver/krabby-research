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

### 2.3 Pin revisions (`KRABBY_PIN_REV`)

Wiring is selected at **compile time** in **`arduino/board_pins.h`** (`#define KRABBY_PIN_REV`, default **2**). Rev **3** matches **`MOTOR_HEADER_PINOUT.md`**.

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

Flash each Mega with the image that matches **that** board’s wiring. All three boards use the same sketch; role is elected at runtime.

### 2.4 Python SDK

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


### Feature 1: Auto-Calibration (Run Once)
The robot now calibrates itself automatically and saves limits to EEPROM.
 - Select Option 2 (Auto-Calibrate) in the menu.
 - Stand Back: The robot will perform the safety sequence:
    - Yaw Left -> Yaw Right -> Hip Up -> Knee Out -> Knee In -> Hip Down.
 - Result: Limits are saved. You do not need to repeat this after rebooting.

### Feature 2: Manual Jog Mode
 - Select Option 3 (Jog Mode).
 - Type the joint name (e.g., LHY or LKL).
 - Hold 'W' to Extend, Hold 'S' to Retract.
 - Release keys to stop immediately.

### Feature 3: Neutral Pose
 - Select Option 1.
 - Robot moves all joints to center (0.5). Useful to verify calibration accuracy.
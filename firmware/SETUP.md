# Krabby-Uno MCU Setup Guide (Task 1)

## Overview

This guide covers the setup, wiring, and execution of the Single-Motor Yaw Controller for the Krabby-Uno robot.

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560 R3
  - BTS7960 43A H-Bridge Driver
  - 12V DC Gear Motor (131:1) with Encoder
  - 12V Power Supply
  - Jetson Orin (or Ubuntu Laptop, or Windows)
- **Software:**
  - Ubuntu 22.04/Windows 10+
  - Python 3.x (`pip install -r requirements.txt`)
  - Arduino IDE or CLI

## 1. Wiring

Follow this pinout carefully. Failure to ground common pins may result in signal noise.

| Component   | Pin Label | Arduino Mega Pin | Notes         |
| :---------- | :-------- | :--------------- | :------------ |
| **Encoder** | Phase A   | **D2**           | Interrupt Pin |
| **Encoder** | Phase B   | **D3**           | Interrupt Pin |
| **Encoder** | VCC       | 5V               | Logic Power   |
| **Encoder** | GND       | GND              | Common Ground |
| **Driver**  | R_EN      | **D4**           | Enable Right  |
| **Driver**  | L_EN      | **D5**           | Enable Left   |
| **Driver**  | R_PWM     | **D6**           | PWM Forward   |
| **Driver**  | L_PWM     | **D7**           | PWM Reverse   |
| **Driver**  | VCC       | 5V               | Logic Power   |
| **Driver**  | GND       | GND              | Common Ground |

_Power Connection:_ Connect 12V Battery to Driver `B+` and `B-`. Connect Motor to Driver `M+` and `M-`.

## 2. Firmware Installation

1.  Open `Task1/Task1_YawControl/Task1_YawControl.ino` in Arduino IDE.
2.  Select Board: **Arduino Mega or Mega 2560**.
3.  Select Port: usually `/dev/ttyACM0` on Linux, or COM1-4 on Windows (confirm in Arduino IDE from board dropdown)
4.  Click **Upload**.

## 3. Running the SDK Test

1.  Navigate to the SDK directory:
    ```bash
    cd krabby-research/firmware
    ```
2.  Setup venv and install requirements:
    ```bash
    python -m venv .venv
    .venv/bin/activate (or .venv/Scripts/Activate.ps1 on windows)
    pip install -r requirements.txtl
    ```
3.  Run the test script:
    ```bash
    export KRABBY_MCU_PORT=/dev/ttyACM0 (or set KRABBY_MCU_PORT=COM4 on Windows w/ whatever correct COM port is)
    python3 krabby_mcu.py
    ```

## 4. Verification

- The script will attempt to move the motor to **-30 degrees** (Left), **+30 degrees** (Right), and **0 degrees** (Center).
- Watch the terminal output. You should see `Pos:` values updating to match the target (e.g., nearing -1.0 or 1.0).
- If the motor spins continuously without stopping, check your Encoder wiring (specifically Phase A/B order).

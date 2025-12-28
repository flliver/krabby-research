# Krabby-Uno MCU Setup Guide (Task 2: Dual Yaw)

## Overview

This guide covers the setup, wiring, and execution of the **Dual Hip Yaw Controller** (Left & Right). It expands on Task 1 by adding a second motor driver, integrating current sensing for obstacle detection, and establishing the pinout for the full 6-motor architecture.

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560 R3
  - **2x** BTS7960 43A H-Bridge Drivers
  - **2x** 12V DC Gear Motors (131:1) with Encoders
  - **4x** Resistors (4.7kΩ to 10kΩ) for Current Sensing protection
  - 12V Power Supply (Battery or Benchtop)
  - Jetson Orin (or Ubuntu Laptop)
- **Software:**
  - Ubuntu 22.04
  - Python 3.x (`pip install pyserial`)
  - Arduino IDE or CLI

## 1. Wiring Map

**Safety Warning:** Do **not** connect the BTS7960 `R_IS` or `L_IS` pins directly to the Arduino. You **must** verify a resistor (4.7kΩ - 10kΩ) is in series to protect the MCU from over-voltage.

### A. Yaw Motors (for Task 2)

| Joint         | Driver Pin    | Arduino Pin | Function  | Notes                        |
| :------------ | :------------ | :---------- | :-------- | :--------------------------- |
| **LEFT YAW**  | **Encoder A** | **D18**     | Interrupt | Master Tick                  |
| (Motor 1)     | Encoder B     | D19         | Input     | Direction                    |
|               | R_EN          | D22         | Output    | Enable Forward               |
|               | L_EN          | D23         | Output    | Enable Reverse               |
|               | R_PWM         | **D46**     | PWM       | Forward Drive                |
|               | L_PWM         | **D45**     | PWM       | Reverse Drive                |
|               | **R_IS**      | **A4**      | Analog    | **Series Resistor Required** |
|               | **L_IS**      | **A5**      | Analog    | **Series Resistor Required** |
|               |               |             |           |                              |
| **RIGHT YAW** | **Encoder A** | **D20**     | Interrupt | Master Tick                  |
| (Motor 2)     | Encoder B     | D21         | Input     | Direction                    |
|               | R_EN          | D24         | Output    | Enable Forward               |
|               | L_EN          | D25         | Output    | Enable Reverse               |
|               | R_PWM         | **D2**      | PWM       | Forward Drive                |
|               | L_PWM         | **D3**      | PWM       | Reverse Drive                |
|               | **R_IS**      | **A6**      | Analog    | **Series Resistor Required** |
|               | **L_IS**      | **A7**      | Analog    | **Series Resistor Required** |

_Common Connections:_

- **Driver VCC:** Connect to Arduino 5V.
- **Driver GND:** Connect to Arduino GND (Common Ground is critical).
- **Motor Power:** Connect 12V Battery to Driver `B+` and `B-`.

### B. Linear Actuators (Future Provisioning)

_These pins are reserved in the code but not physically active yet._

| Joint      | Pot Pin | PWM Pins | IS Pins (Safety) |
| :--------- | :------ | :------- | :--------------- |
| Hip Left   | A0      | D4, D5   | A8, A9           |
| Knee Left  | A1      | D6, D7   | A10, A11         |
| Hip Right  | A2      | D8, D9   | A12, A13         |
| Knee Right | A3      | D10, D11 | A14, A15         |

## 2. Firmware Installation

1.  Open `TASK2_PHASE1/Dual_Yaw/Dual_Yaw.ino` in Arduino IDE.
2.  **Safety Check:** Ensure `CURRENT_LIMIT` is set to `600` (approx 2.9V) in the config section.
3.  Select Board: **Arduino Mega or Mega 2560**.
4.  Select Port: usually `/dev/ttyACM0`.
5.  Click **Upload**.

## 3. Running the Test SDK

1.  Navigate to the SDK directory:
    ```bash
    cd Task2_PHASE1
    ```
2.  Run the Dual Yaw Controller:
    ```bash
    python3 kraby_mcu_dual_yaw.py
    ```

## 4. Verification & Troubleshooting

- **Normal Operation:**

  - Script logs: `Command -> L: -0.50, R: -0.50`.
  - Motors should move simultaneously.
  - Telemetry should track targets: `Pos -> L: -0.498 R: -0.501`.

- **Safety Triggered (Stall):**

  - If you hold the motor shaft, the script should log: `[WARN] Obstacle Detected`.
  - The motor will stop. You must restart the script or send `0.0` to reset.

- **Runaway Detected:**
  - If the log screams `CRITICAL: RUNAWAY DETECTED`, your **Encoder wiring is reversed** relative to the motor wires.
  - **Fix:** Swap the `M+` and `M-` wires on the H-Bridge for that motor.

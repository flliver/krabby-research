# Krabby-Uno Task 2: Six-Axis Leg Controller

## Overview

This firmware drives a full leg pair (Left & Right) consisting of **6 Motors**:

- **2x Hip Yaw Motors:** DC Motor + Encoder
- **4x Linear Actuators:** DC Motor + Potentiometer (Limited travel: Hip/Knee)

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560
  - **6x** BTS7960 43A H-Bridge Drivers
  - **12x** Resistors (4.7kΩ to 10kΩ) for Current Sense protection
  - 12V Power Supply
- **Software:**
  - Python 3 (`pip install pyserial`)
  - Arduino IDE

## 1. Hardware Wiring

**CRITICAL SAFETY WARNING:** You MUST place a resistor (4.7k-10kΩ) between every BTS7960 `IS` pin and the Arduino Analog pin. Connecting 12V drivers directly to Arduino Analog pins can destroy the MCU.

### A. Left Leg

| Joint         | Driver Pin        | Arduino Pin | Function          |
| :------------ | :---------------- | :---------- | :---------------- |
| **Yaw Left**  | **Encoder A**     | **D18**     | Interrupt         |
|               | Encoder B         | D19         | Direction         |
|               | PWM (Fwd/Rev)     | D46, D45    | Drive             |
|               | EN (Fwd/Rev)      | D22, D23    | Enable            |
|               | IS (R/L)          | A4, A5      | Current Sense     |
| **Hip Left**  | **Potentiometer** | **A0**      | Position Feedback |
|               | PWM (Up/Dn)       | D4, D5      | Drive             |
|               | EN (Up/Dn)        | D26, D27    | Enable            |
|               | IS (R/L)          | A8, A9      | Current Sense     |
| **Knee Left** | **Potentiometer** | **A1**      | Position Feedback |
|               | PWM (Out/In)      | D6, D7      | Drive             |
|               | EN (Out/In)       | D28, D29    | Enable            |
|               | IS (R/L)          | A10, A11    | Current Sense     |

### B. Right Leg

| Joint          | Driver Pin        | Arduino Pin | Function          |
| :------------- | :---------------- | :---------- | :---------------- |
| **Yaw Right**  | **Encoder A**     | **D20**     | Interrupt         |
|                | Encoder B         | D21         | Direction         |
|                | PWM (Fwd/Rev)     | D2, D3      | Drive             |
|                | EN (Fwd/Rev)      | D24, D25    | Enable            |
|                | IS (R/L)          | A6, A7      | Current Sense     |
| **Hip Right**  | **Potentiometer** | **A2**      | Position Feedback |
|                | PWM (Up/Dn)       | D8, D9      | Drive             |
|                | EN (Up/Dn)        | D30, D31    | Enable            |
|                | IS (R/L)          | A12, A13    | Current Sense     |
| **Knee Right** | **Potentiometer** | **A3**      | Position Feedback |
|                | PWM (Out/In)      | D10, D11    | Drive             |
|                | EN (Out/In)       | D32, D33    | Enable            |
|                | IS (R/L)          | A14, A15    | Current Sense     |

## 2. Installation

1.  **Configure Firmware:** Open `Six_Axis_Controller.ino`.
2.  **Calibrate:** _Before_ running the robot, follow instructions in `CALIBRATION.md` to set your potentiometer limits.
3.  **Upload:** Flash the code to the Arduino Mega.
4.  **Run SDK:**
    ```bash
    python3 krabby_mcu_six_axis.py --debug
    ```

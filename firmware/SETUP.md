# Krabby-Uno MCU Setup Guide (Task 1)

## Overview

This guide covers the setup, wiring, and execution of the Single-Motor Yaw Controller.
**Note:** The wiring below uses the Arduino Mega's extended headers to ensure compatibility with the MultiMoto shield (Task 2).

## Prerequisites

- **Hardware:**
  - Arduino Mega 2560 R3
  - BTS7960 43A H-Bridge Driver
  - 12V DC Gear Motor (131:1) with Encoder
  - 12V Power Supply
  - Jetson Orin (or Ubuntu Laptop)
- **Software:**
  - Ubuntu 22.04
  - Python 3.x (`pip install pyserial`)
  - Arduino IDE or CLI

## 1. Wiring (Shield-Safe Pinout)

**Crucial:** Do not use pins 2-13, as these will be covered by the MultiMoto shield in later tasks.
**NOTE** On dfrobot 12V motor Encoder phases seem reversed. Yellow wire, despite being marked A needs to go to 19, and white wire to 18

| Component   | Pin Label | Arduino Mega Pin | Notes                         |
| :---------- | :-------- | :--------------- | :---------------------------- |
| **Encoder** | Phase A   | **D18**          | Interrupt Pin (Tx1)           |
| **Encoder** | Phase B   | **D19**          | Interrupt Pin (Rx1)           |
| **Encoder** | VCC       | 5V               | Logic Power                   |
| **Encoder** | GND       | GND              | Common Ground                 |
| **Driver**  | R_EN      | **D22**          | Enable Right (Digital)        |
| **Driver**  | L_EN      | **D23**          | Enable Left (Digital)         |
| **Driver**  | R_PWM     | **D46**          | PWM Forward (Extended Header) |
| **Driver**  | L_PWM     | **D45**          | PWM Reverse (Extended Header) |
| **Driver**  | VCC       | 5V               | Logic Power                   |
| **Driver**  | GND       | GND              | Common Ground                 |

_Power Connection:_ Connect 12V Battery to Driver `B+` and `B-`. Connect Motor to Driver `M+` and `M-`.

## 2. Firmware Installation

1.  Open `Task1_YawControl/Task1_YawControl.ino` in Arduino IDE.
2.  Select Board: **Arduino Mega or Mega 2560**.
3.  Select Port: usually `/dev/ttyACM0` on Linux.
4.  Click **Upload**.

## 3. Running the SDK Test

1.  Navigate to the SDK directory:
    ```bash
    cd Task1
    ```
2.  Install requirements:
    ```bash
    pip install pyserial
    ```
3.  Run the test script:
    ```bash
    python3 krabby_mcu.py
    ```

## 4. Verification

- The script will attempt to sweep the motor Left, Right, and Center.
- **Success:** You will see `Pos:` updates in the terminal matching the target.
- **Error Handling:**
  - If the script says `[SDK] No feedback detected`, check your **Encoder Wiring**.
  - If the script says `Command sent but no movement`, check your **12V Power Supply** and **H-Bridge Wiring**.

## 5. Configuration Notes

- TODO: If you reverse pin 18/19, the motor will just sweep forever instead of recognizing/stopping.
- TODO: Kp/Kd is currently set to avoid overshooting during movements on DFRobot 12V motor. This may change per motor, and MCU currently is not smart enough to self-adjust Kp/Kd
- TODO: Debugging is somewhat verbose on krabby_mcu.py to help w/ motor position debugging, would be good to add a warn vs debug log at some point, and warn on non-essential misbehavior (such as overshooting on movements)
- TODO: Motor start jerks pretty hard, would be very aggressive on large leg. Would be good to have smoother acceleration/deceleration curve on MCU side when commands are sent.
- TODO: I added a 10k pull-down resistor to L_EN/R_EN to try to stop motor chatter while centering. However, after changing code to stop accidentally sending all commands at once, chatter doesn't happen anymore and I've removed pull-down resistors and it's all fine. So not sure if they ever need to come back at some point, but should keep in mind.
- TODO: Sending yaw commands w/o waiting was causing chatter. Needs better lock management around control states to avoid I'm guessing
- TODO: motor doesn't appear to have an actual home position, it just starts wherever it was, so it can slowly drift over time. Next Task should utilize IS sensor to detect forward/aft end stops. 


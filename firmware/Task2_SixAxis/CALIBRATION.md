# Linear Actuator Calibration Guide

Since every mechanical assembly is slightly different, you must find the specific "Retracted" and "Extended" potentiometer values for your 4 linear actuators.

**Safety Note:** Do not force the actuators by hand. Use the provided Wizard tool to gently pulse the motors.

### Step 1: Run the Wizard

1.  Connect your computer via USB.
2.  Run the interactive calibration tool:
    ```bash
    python3 calibration_wizard.py
    ```

### Step 2: Calibrate Each Joint

The wizard will ask you to select a joint (e.g., "1: Hip Left").

1.  **Find Minimum (Retracted):**

    - Press `-` (or `s`) to gently retract the actuator.
    - Stop when it reaches the physical minimum limit.
    - Write down the `POT VALUE` shown on screen. This is your `minPot`.
    - _Example:_ HipL Min = 102

2.  **Find Maximum (Extended):**

    - Press `+` (or `w`) to gently extend the actuator.
    - Stop when it reaches the physical maximum limit.
    - Write down the `POT VALUE`. This is your `maxPot`.
    - _Example:_ HipL Max = 890

3.  **Repeat** for all 4 linear joints (Hip Left, Knee Left, Hip Right, Knee Right).

### Step 3: Update Firmware

1.  Open `Six_Axis_Controller.ino` in the Arduino IDE.
2.  Scroll down to the **Instantiation** section (bottom of file).
3.  Find the lines where `LinearActuator` objects are created.
4.  **Action:** In `setup()`, add these lines with YOUR measured values:

    ```cpp
    void setup() {
      // ... existing init code ...

      // SET YOUR CALIBRATION VALUES HERE:
      // (Replace these numbers with the ones you wrote down)
      hipL.minPot = 102; hipL.maxPot = 890;
      kneeL.minPot = 110; kneeL.maxPot = 905;
      hipR.minPot = 98;  hipR.maxPot = 880;
      kneeR.minPot = 105; kneeR.maxPot = 910;

      // ... rest of setup ...
    }
    ```

5.  **Re-upload** the code to the Arduino. Your robot is now calibrated!

/*
 * Krabby-Uno Task 2: Six-Axis Leg Controller
 * Supports: 2x Yaw Motors (Encoder) + 4x Linear Actuators (Potentiometer)
 */

#include <Arduino.h>

// --- CONFIGURATION ---
const int TELEMETRY_INTERVAL_MS = 50; // 20Hz update

// [TUNING PARAMETER 1] Stall Safety
// 600 ~= 3.0V from Current Sense. Adjust if false triggers occur.
const int CURRENT_TRIP_THRESHOLD = 600;

// --- MOTION PROFILE ---
// [TUNING PARAMETER 2] Acceleration Step
const int PWM_RAMP_STEP = 5;
// [TUNING PARAMETER 3] Acceleration Speed
const int RAMP_INTERVAL_MS = 10;

unsigned long lastTelemetry = 0;

// ==========================================
// CLASS 1: YAW MOTOR (Encoder Based)
// Formerly "JointMotor" in Dual_Yaw_v2
// ==========================================
class YawMotor
{
public:
    // Pins
    int pinPwmR, pinPwmL, pinEnR, pinEnL, pinISR, pinISL;

    // State
    volatile long encoderPosition = 0;
    long targetCounts = 0;
    int currentPwm = 0;

    // Safety & Averaging State
    float averageCurrent = 0.0;
    const float ALPHA = 0.1; // Smoothing factor

    bool safetyTriggered = false;
    bool runawayTriggered = false;

    unsigned long lastMoveTime = 0;
    unsigned long lastRampTime = 0;
    long lastEncoderPos = 0;

    // Constants
    const float MAX_COUNTS = 2096.0 * (30.0 / 360.0); // ~174 counts
    float Kp = 3.5;

    YawMotor(int pR, int pL, int eR, int eL, int isR, int isL)
    {
        pinPwmR = pR;
        pinPwmL = pL;
        pinEnR = eR;
        pinEnL = eL;
        pinISR = isR;
        pinISL = isL;
    }

    void init()
    {
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEnR, OUTPUT);
        pinMode(pinEnL, OUTPUT);
        pinMode(pinISR, INPUT);
        pinMode(pinISL, INPUT);
        stopMotor();
    }

    // --- SAFETY: FILTERED OBSTACLE DETECTION ---
    bool checkStall()
    {
        int maxRaw = max(analogRead(pinISR), analogRead(pinISL));
        averageCurrent = (ALPHA * maxRaw) + ((1.0 - ALPHA) * averageCurrent);

        if (averageCurrent > CURRENT_TRIP_THRESHOLD)
        {
            safetyTriggered = true;
            stopMotor();
            return true;
        }
        return false;
    }

    // --- SAFETY: RUNAWAY PROTECTION ---
    void checkRunaway(int targetPwm)
    {
        if (runawayTriggered)
            return;

        if (abs(targetPwm) > 60)
        {
            unsigned long now = millis();
            if (now - lastMoveTime > 200)
            { // 200ms grace period
                long delta = encoderPosition - lastEncoderPos;
                bool fault = false;

                // Case A: PWM Positive (Forward) but Encoder Negative
                if (targetPwm > 0 && delta < -5)
                    fault = true;
                // Case B: PWM Negative (Backward) but Encoder Positive
                if (targetPwm < 0 && delta > 5)
                    fault = true;
                // Case C: Power applied but No Movement
                if (abs(delta) < 2)
                    fault = true;

                if (fault)
                {
                    runawayTriggered = true;
                    stopMotor();
                }
                lastMoveTime = now;
                lastEncoderPos = encoderPosition;
            }
        }
        else
        {
            lastMoveTime = millis();
            lastEncoderPos = encoderPosition;
        }
    }

    void setTarget(float val)
    {
        if (safetyTriggered || runawayTriggered)
            return;
        // Clamp [-1.0, 1.0]
        if (val > 1.0)
            val = 1.0;
        if (val < -1.0)
            val = -1.0;
        targetCounts = (long)(val * MAX_COUNTS);
    }

    void update()
    {
        if (safetyTriggered || runawayTriggered)
            return;
        if (checkStall())
            return;

        long error = targetCounts - encoderPosition;
        int desiredPwm = (int)(error * Kp);

        checkRunaway(desiredPwm);

        // Smooth Ramping
        if (millis() - lastRampTime >= RAMP_INTERVAL_MS)
        {
            lastRampTime = millis();
            if (currentPwm < desiredPwm)
            {
                currentPwm += PWM_RAMP_STEP;
                if (currentPwm > desiredPwm)
                    currentPwm = desiredPwm;
            }
            else if (currentPwm > desiredPwm)
            {
                currentPwm -= PWM_RAMP_STEP;
                if (currentPwm < desiredPwm)
                    currentPwm = desiredPwm;
            }
        }
        driveHardware(currentPwm);
    }

    void driveHardware(int pwm)
    {
        if (pwm > 255)
            pwm = 255;
        if (pwm < -255)
            pwm = -255;
        if (abs(pwm) < 20)
            pwm = 0; // Deadband

        if (pwm > 0)
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, pwm);
            analogWrite(pinPwmL, 0);
        }
        else if (pwm < 0)
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, abs(pwm));
        }
        else
        {
            stopMotor();
        }
    }

    void stopMotor()
    {
        digitalWrite(pinEnR, LOW);
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        currentPwm = 0;
    }

    float getPos() { return (float)encoderPosition / MAX_COUNTS; }
};

// ==========================================
// CLASS 2: LINEAR ACTUATOR (Potentiometer Based)
// New class for Task 2 expansion
// ==========================================
class LinearActuator
{
public:
    int pinPwmR, pinPwmL, pinEnR, pinEnL, pinISR, pinISL, pinPot;
    int currentPwm = 0;

    // CALIBRATION: RAW POT VALUES
    // These must be calibrated per motor!
    int minPot = 100; // Retracted (0.0)
    int maxPot = 900; // Extended (1.0)

    int targetRaw = 0;
    bool safetyTriggered = false;
    float averageCurrent = 0.0;
    const float ALPHA = 0.1;
    unsigned long lastRampTime = 0;
    float Kp = 2.0; // Lower Kp for linear actuators

    LinearActuator(int pR, int pL, int eR, int eL, int isR, int isL, int pot)
    {
        pinPwmR = pR;
        pinPwmL = pL;
        pinEnR = eR;
        pinEnL = eL;
        pinISR = isR;
        pinISL = isL;
        pinPot = pot;
    }

    void init()
    {
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEnR, OUTPUT);
        pinMode(pinEnL, OUTPUT);
        pinMode(pinISR, INPUT);
        pinMode(pinISL, INPUT);
        pinMode(pinPot, INPUT);
        stopMotor();
        // Default target to current position to prevent startup jump
        targetRaw = analogRead(pinPot);
    }

    bool checkStall()
    {
        int maxRaw = max(analogRead(pinISR), analogRead(pinISL));
        averageCurrent = (ALPHA * maxRaw) + ((1.0 - ALPHA) * averageCurrent);
        if (averageCurrent > CURRENT_TRIP_THRESHOLD)
        {
            safetyTriggered = true;
            stopMotor();
            return true;
        }
        return false;
    }

    // Input: 0.0 (Retracted) to 1.0 (Extended)
    void setTarget(float val)
    {
        if (safetyTriggered)
            return;
        if (val > 1.0)
            val = 1.0;
        if (val < 0.0)
            val = 0.0;
        // Map 0.0-1.0 to MinPot-MaxPot
        targetRaw = minPot + (int)(val * (maxPot - minPot));
    }

    void update()
    {
        if (safetyTriggered || checkStall())
            return;

        int currentRaw = analogRead(pinPot);
        int error = targetRaw - currentRaw;

        // Deadband (higher for pots to prevent jitter)
        if (abs(error) < 10)
            error = 0;

        int desiredPwm = (int)(error * Kp);

        // Ramp Logic
        if (millis() - lastRampTime >= RAMP_INTERVAL_MS)
        {
            lastRampTime = millis();
            if (currentPwm < desiredPwm)
            {
                currentPwm += PWM_RAMP_STEP;
                if (currentPwm > desiredPwm)
                    currentPwm = desiredPwm;
            }
            else if (currentPwm > desiredPwm)
            {
                currentPwm -= PWM_RAMP_STEP;
                if (currentPwm < desiredPwm)
                    currentPwm = desiredPwm;
            }
        }
        driveHardware(currentPwm);
    }

    void driveHardware(int pwm)
    {
        if (pwm > 255)
            pwm = 255;
        if (pwm < -255)
            pwm = -255;
        if (abs(pwm) < 30)
            pwm = 0;

        if (pwm > 0)
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, pwm);
            analogWrite(pinPwmL, 0);
        }
        else if (pwm < 0)
        {
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, abs(pwm));
        }
        else
        {
            stopMotor();
        }
    }

    void stopMotor()
    {
        digitalWrite(pinEnR, LOW);
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        currentPwm = 0;
    }

    // Return Normalized Position (0.0 - 1.0)
    float getPos()
    {
        int val = analogRead(pinPot);
        // Avoid division by zero if not calibrated
        if (maxPot == minPot)
            return 0.0;
        return (float)(val - minPot) / (float)(maxPot - minPot);
    }
};

// ==========================================
// INSTANTIATION (6 MOTORS)
// ==========================================

// 1. YAW MOTORS (Encoder) - Pins from Task 2
YawMotor yawL(46, 45, 22, 23, A4, A5);
YawMotor yawR(2, 3, 24, 25, A6, A7);

// 2. LINEAR ACTUATORS (Potentiometer)
// Using new Enable Pins 26-33 for the 4 linear drivers
// Mapping: PWM_R, PWM_L, EN_R, EN_L, IS_R, IS_L, POT
LinearActuator hipL(4, 5, 26, 27, A8, A9, A0);
LinearActuator kneeL(6, 7, 28, 29, A10, A11, A1);
LinearActuator hipR(8, 9, 30, 31, A12, A13, A2);
LinearActuator kneeR(10, 11, 32, 33, A14, A15, A3);

// --- ISRs ---
void isrL()
{
    if (digitalRead(19))
        yawL.encoderPosition++;
    else
        yawL.encoderPosition--;
}
void isrR()
{
    if (digitalRead(21))
        yawR.encoderPosition++;
    else
        yawR.encoderPosition--;
}

void setup()
{
    Serial.begin(115200);

    // Init All Motors
    yawL.init();
    yawR.init();
    hipL.init();
    kneeL.init();
    hipR.init();
    kneeR.init();

    // Encoder Interrupts
    pinMode(18, INPUT_PULLUP);
    pinMode(19, INPUT_PULLUP);
    pinMode(20, INPUT_PULLUP);
    pinMode(21, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(18), isrL, RISING);
    attachInterrupt(digitalPinToInterrupt(20), isrR, RISING);
}

// --- MAIN LOOP ---
void loop()
{
    // 1. INPUT PARSING: "T <yL> <yR> <hL> <kL> <hR> <kR>"
    if (Serial.available())
    {
        if (Serial.read() == 'T')
        {
            float v[6];
            // Expect 6 float values
            for (int i = 0; i < 6; i++)
                v[i] = Serial.parseFloat();

            // Safety Reset: If all zeros, reset flags
            bool allZeros = true;
            for (int i = 0; i < 6; i++)
                if (v[i] != 0)
                    allZeros = false;

            if (allZeros)
            {
                yawL.runawayTriggered = false;
                yawL.safetyTriggered = false;
                yawR.runawayTriggered = false;
                yawR.safetyTriggered = false;
                hipL.safetyTriggered = false;
                kneeL.safetyTriggered = false;
                hipR.safetyTriggered = false;
                kneeR.safetyTriggered = false;
            }

            // Yaw Targets [-1.0, 1.0]
            yawL.setTarget(v[0]);
            yawR.setTarget(v[1]);
            // Linear Targets [0.0, 1.0]
            hipL.setTarget(v[2]);
            kneeL.setTarget(v[3]);
            hipR.setTarget(v[4]);
            kneeR.setTarget(v[5]);
        }
    }

    // 2. UPDATE CONTROL LOOPS
    yawL.update();
    yawR.update();
    hipL.update();
    kneeL.update();
    hipR.update();
    kneeR.update();

    // 3. TELEMETRY STREAMING
    if (millis() - lastTelemetry > TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = millis();

        // Format: FB:yL,yR,hL,kL,hR,kR
        Serial.print("FB:");
        Serial.print(yawL.getPos(), 3);
        Serial.print(",");
        Serial.print(yawR.getPos(), 3);
        Serial.print(",");
        Serial.print(hipL.getPos(), 3);
        Serial.print(",");
        Serial.print(kneeL.getPos(), 3);
        Serial.print(",");
        Serial.print(hipR.getPos(), 3);
        Serial.print(",");
        Serial.print(kneeR.getPos(), 3);

        // Debug: Print Raw Pot Values to help Calibration
        // The client needs this to set minPot/maxPot
        Serial.print(",POT:");
        Serial.print(analogRead(A0));
        Serial.print(",");
        Serial.print(analogRead(A1));
        Serial.print(",");
        Serial.print(analogRead(A2));
        Serial.print(",");
        Serial.print(analogRead(A3));

        // Debug: Print Average Current (Optional, good for tuning)
        Serial.print(",AVG:");
        Serial.print((int)yawL.averageCurrent);

        Serial.println();
    }
}
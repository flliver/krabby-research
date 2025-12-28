/*
 * Krabby-Uno Task 2: Dual Yaw Control
 * Features: Soft-Start, Anti-Runaway, Obstacle Safety, Auto-Homing Capable
 * Platform: Arduino Mega 2560
 * Drivers: BTS7960
 */

#include <Arduino.h>

// --- CONFIGURATION ---
const int TELEMETRY_INTERVAL_MS = 20; // 50Hz update rate
const int CURRENT_LIMIT = 600;        // Stall detection threshold (0-1023)
const int PWM_RAMP_STEP = 10;         // Max PWM change per loop (Smoothing)
const int HOMING_PWM = 60;            // Slow speed for calibration

class JointMotor
{
public:
    // Pins
    int pinPwmR, pinPwmL, pinEnR, pinEnL, pinISR, pinISL;

    // State
    volatile long encoderPosition = 0;
    long targetCounts = 0;
    int currentPwm = 0; // For ramping/smoothing

    // Safety State
    bool safetyTriggered = false;
    bool runawayTriggered = false; // Wiring protection
    unsigned long lastMoveTime = 0;
    long lastEncoderPos = 0;

    // Constants
    // 16CPR motor * 131 gear ratio = ~2096 counts per output rev
    // We assume +/- 30 degrees range.
    const float MAX_COUNTS = 2096.0 * (30.0 / 360.0); // ~174 counts
    float Kp = 3.5;

    // Constructor
    JointMotor(int pR, int pL, int eR, int eL, int isR, int isL)
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

    // --- SAFETY: OBSTACLE DETECTION ---
    bool checkStall()
    {
        // Check current sensors on both sides of H-Bridge
        if (analogRead(pinISR) > CURRENT_LIMIT || analogRead(pinISL) > CURRENT_LIMIT)
        {
            safetyTriggered = true;
            stopMotor();
            return true;
        }
        return false;
    }

    // --- SAFETY: REVERSE WIRING / RUNAWAY PROTECTION ---
    void checkRunaway(int targetPwm)
    {
        if (runawayTriggered)
            return;

        // Only check if we are trying to move reasonably fast
        if (abs(targetPwm) > 50)
        {
            unsigned long now = millis();
            // Check every 100ms
            if (now - lastMoveTime > 100)
            {
                long delta = encoderPosition - lastEncoderPos;

                // Case A: PWM Positive (Forward), but Encoder Negative (Backward)
                if (targetPwm > 0 && delta < -2)
                    runawayTriggered = true;

                // Case B: PWM Negative (Backward), but Encoder Positive (Forward)
                if (targetPwm < 0 && delta > 2)
                    runawayTriggered = true;

                // Case C: PWM High, but Encoder not moving (Stalled or Disconnected)
                if (abs(delta) < 1)
                    runawayTriggered = true;

                if (runawayTriggered)
                {
                    stopMotor(); // HARD STOP
                }

                lastMoveTime = now;
                lastEncoderPos = encoderPosition;
            }
        }
    }

    // --- CALIBRATION: HOMING ROUTINE ---
    // Moves motor positive until stall, sets that as "Max Limit"
    void home()
    {
        Serial.println("Homing...");
        // 1. Move slowly positive
        while (!checkStall())
        {
            driveHardware(HOMING_PWM);
            delay(10);
            // Note: In production, We will add a timeout here to prevent infinite loop
        }
        // 2. Hit hard stop. Stop immediately.
        stopMotor();
        // 3. Set current position as the physical maximum
        encoderPosition = MAX_COUNTS;
        targetCounts = MAX_COUNTS; // Stay here
        safetyTriggered = false;   // Clear the safety flag triggered by the stall
        Serial.println("Homed!");
    }

    // --- CONTROL: SET TARGET ---
    void setTargetNormalized(float val)
    {
        // Do not accept new targets if safety triggered (must reset 0 first)
        if (safetyTriggered || runawayTriggered)
            return;

        // Clamp input -1.0 to 1.0
        if (val > 1.0)
            val = 1.0;
        if (val < -1.0)
            val = -1.0;

        targetCounts = (long)(val * MAX_COUNTS);
    }

    // --- TELEMETRY: GET POSITION ---
    float getNormalizedPosition()
    {
        return (float)encoderPosition / (float)MAX_COUNTS;
    }

    // --- MAIN UPDATE LOOP ---
    void update()
    {
        // 1. Check Safety
        if (safetyTriggered || runawayTriggered)
            return;
        if (checkStall())
            return;

        // 2. Calculate Error
        long error = targetCounts - encoderPosition;
        int targetPwm = (int)(error * Kp);

        // 3. Check Health (Runaway/Stall)
        checkRunaway(targetPwm);

        // 4. Soft-Start / Ramping Logic
        // Slowly adjust currentPwm towards targetPwm to prevent jerking
        if (currentPwm < targetPwm)
        {
            currentPwm += PWM_RAMP_STEP;
            if (currentPwm > targetPwm)
                currentPwm = targetPwm;
        }
        else if (currentPwm > targetPwm)
        {
            currentPwm -= PWM_RAMP_STEP;
            if (currentPwm < targetPwm)
                currentPwm = targetPwm;
        }

        // 5. Drive Motor
        driveHardware(currentPwm);
    }

    // --- HARDWARE INTERFACE ---
    void driveHardware(int pwm)
    {
        // Clamp PWM Range
        if (pwm > 255)
            pwm = 255;
        if (pwm < -255)
            pwm = -255;

        // Deadband (Prevent buzzing at low power)
        if (abs(pwm) < 20)
            pwm = 0;

        if (pwm > 0)
        {
            // Forward
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, pwm);
            analogWrite(pinPwmL, 0);
        }
        else if (pwm < 0)
        {
            // Reverse
            digitalWrite(pinEnR, HIGH);
            digitalWrite(pinEnL, HIGH);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, abs(pwm));
        }
        else
        {
            // Stop
            stopMotor();
        }
    }

    void stopMotor()
    {
        digitalWrite(pinEnR, LOW);
        digitalWrite(pinEnL, LOW);
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        currentPwm = 0; // Reset ramp
    }
};

// --- INSTANTIATION ---
// 1. LEFT YAW (Encoder: 18/19, PWM: 46/45, IS: A4/A5)
JointMotor yawLeft(46, 45, 22, 23, A4, A5);

// 2. RIGHT YAW (Encoder: 20/21, PWM: 2/3, IS: A6/A7)
JointMotor yawRight(2, 3, 24, 25, A6, A7);

// --- INTERRUPT SERVICE ROUTINES (Global) ---
void isrLeft()
{
    // Read Channel B (Pin 19) to determine direction
    if (digitalRead(19) > 0)
        yawLeft.encoderPosition++;
    else
        yawLeft.encoderPosition--;
}

void isrRight()
{
    // Read Channel B (Pin 21) to determine direction
    if (digitalRead(21) > 0)
        yawRight.encoderPosition++;
    else
        yawRight.encoderPosition--;
}

// --- SETUP ---
void setup()
{
    Serial.begin(115200);

    yawLeft.init();
    yawRight.init();

    // Setup Interrupt Pins
    pinMode(18, INPUT_PULLUP);
    pinMode(19, INPUT_PULLUP);
    pinMode(20, INPUT_PULLUP);
    pinMode(21, INPUT_PULLUP);

    // Attach Interrupts to Channel A
    attachInterrupt(digitalPinToInterrupt(18), isrLeft, RISING);
    attachInterrupt(digitalPinToInterrupt(20), isrRight, RISING);

    // OPTIONAL: Homing on startup
    // Uncomment these lines if we want the robot to self-calibrate on boot
    // yawLeft.home();
    // yawRight.home();
}

unsigned long lastTelemetry = 0;

// --- LOOP ---
void loop()
{
    // 1. SERIAL INPUT PARSING
    // Expected Format: "T <float_left> <float_right>"
    if (Serial.available())
    {
        char c = Serial.read();
        if (c == 'T')
        {
            float t1 = Serial.parseFloat();
            float t2 = Serial.parseFloat();

            // MANUAL RESET: Sending "T 0 0" clears safety locks
            if (t1 == 0 && t2 == 0)
            {
                yawLeft.runawayTriggered = false;
                yawLeft.safetyTriggered = false;
                yawRight.runawayTriggered = false;
                yawRight.safetyTriggered = false;
            }

            yawLeft.setTargetNormalized(t1);
            yawRight.setTargetNormalized(t2);
        }
    }

    // 2. UPDATE CONTROL LOOPS
    yawLeft.update();
    yawRight.update();

    // 3. TELEMETRY STREAMING
    if (millis() - lastTelemetry > TELEMETRY_INTERVAL_MS)
    {
        lastTelemetry = millis();

        // Format: "FB:PosL,PosR,S:SafeL,SafeR,RunL,RunR"
        Serial.print("FB:");
        Serial.print(yawLeft.getNormalizedPosition(), 4);
        Serial.print(",");
        Serial.print(yawRight.getNormalizedPosition(), 4);

        Serial.print(",S:");
        Serial.print(yawLeft.safetyTriggered);
        Serial.print(",");
        Serial.print(yawRight.safetyTriggered);
        Serial.print(",");
        Serial.print(yawLeft.runawayTriggered);
        Serial.print(",");
        Serial.println(yawRight.runawayTriggered);
    }
}
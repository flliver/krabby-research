/*
 * Krabby-Uno Task 1: Single Hip Yaw Motor Control
 * Platform: Arduino Mega 2560
 * Driver: BTS7960
 * Motor: 131:1 Metal DC Geared Motor with Encoder
 * * UPDATE: Pins remapped to avoid conflict with MultiMoto Shield.
 */

#include <Arduino.h>

// --- CONFIGURATION ---
// Pins (Mapped to Extended Header & Comms to avoid Shield overlap)
const int PIN_ENC_A = 18; // Interrupt Pin (Tx1)
const int PIN_ENC_B = 19; // Interrupt Pin (Rx1)
const int PIN_R_EN = 22;  // BTS7960 Right Enable (Digital 22)
const int PIN_L_EN = 23;  // BTS7960 Left Enable (Digital 23)
const int PIN_R_PWM = 46; // BTS7960 Right PWM (PWM 46)
const int PIN_L_PWM = 45; // BTS7960 Left PWM (PWM 45)

// Mechanical Constants
// 16CPR motor * 131 gear ratio = ~2096 counts per output rev
const float COUNTS_PER_REV = 2096.0;
const float MAX_ANGLE_DEG = 30.0; // Physical limit (Left/Right 30 deg)
const float GEARBOX_MULT = 1.0;

// Calculated Limits
const long MAX_COUNTS = (long)((MAX_ANGLE_DEG / 360.0) * COUNTS_PER_REV * GEARBOX_MULT);
// Note: We use a larger max for safety clamping, but target logic handles the range.

// PID Control Constants
float Kp = 3.5; // Proportional gain
float Ki = 0.0; // Integral gain (unused for simple pos control)
float Kd = 0.1; // Derivative gain

// Globals
volatile long encoderPosition = 0;
float targetPositionNormalized = 0.0; // -1.0 to 1.0
long targetCounts = 0;
unsigned long lastTelemetryTime = 0;
const int TELEMETRY_INTERVAL_MS = 20; // ~50Hz

// --- INTERRUPT ROUTINES ---
void readEncoder()
{
  // Read Channel B to determine direction
  // Note: On Pin 18/19, digitalRead is fast enough.
  int b = digitalRead(PIN_ENC_B);
  if (b > 0)
  {
    encoderPosition++;
  }
  else
  {
    encoderPosition--;
  }
}

// --- MOTOR DRIVER HELPER ---
void setMotorPWM(int pwm)
{
  // pwm range: -255 (full reverse) to 255 (full forward)

  // Clamp PWM
  if (pwm > 255)
    pwm = 255;
  if (pwm < -255)
    pwm = -255;

  // Deadband (prevent buzzing at low power)
  if (abs(pwm) < 25)
    pwm = 0;

  if (pwm > 0)
  {
    // Forward
    digitalWrite(PIN_R_EN, HIGH);
    digitalWrite(PIN_L_EN, HIGH);
    analogWrite(PIN_R_PWM, pwm);
    analogWrite(PIN_L_PWM, 0);
  }
  else if (pwm < 0)
  {
    // Reverse
    digitalWrite(PIN_R_EN, HIGH);
    digitalWrite(PIN_L_EN, HIGH);
    analogWrite(PIN_R_PWM, 0);
    analogWrite(PIN_L_PWM, abs(pwm));
  }
  else
  {
    // Stop/Brake
    digitalWrite(PIN_R_EN, LOW);
    digitalWrite(PIN_L_EN, LOW);
    analogWrite(PIN_R_PWM, 0);
    analogWrite(PIN_L_PWM, 0);
  }
}

void setup()
{
  Serial.begin(115200);

  pinMode(PIN_ENC_A, INPUT_PULLUP);
  pinMode(PIN_ENC_B, INPUT_PULLUP);
  pinMode(PIN_R_EN, OUTPUT);
  pinMode(PIN_L_EN, OUTPUT);
  pinMode(PIN_R_PWM, OUTPUT);
  pinMode(PIN_L_PWM, OUTPUT);

  // Attach interrupt to Encoder A
  // DigitalPinToInterrupt works for 18, 19, 2, 3, etc.
  attachInterrupt(digitalPinToInterrupt(PIN_ENC_A), readEncoder, RISING);

  // Initialize Motor to stopped
  setMotorPWM(0);
}

void loop()
{
  // 1. SERIAL INPUT PARSING
  // Format: "T <float>" (e.g., "T -0.5")
  if (Serial.available() > 0)
  {
    char cmd = Serial.read();
    if (cmd == 'T')
    {
      float val = Serial.parseFloat();
      // Clamp input -1.0 to 1.0
      if (val > 1.0)
        val = 1.0;
      if (val < -1.0)
        val = -1.0;

      targetPositionNormalized = val;
      // Map normalized input to encoder counts
      targetCounts = (long)(targetPositionNormalized * MAX_COUNTS);
    }
  }

  // 2. CONTROL LOOP (PID)
  long error = targetCounts - encoderPosition;

  // PD Controller
  int controlSignal = (int)(error * Kp);

  setMotorPWM(controlSignal);

  // 3. TELEMETRY STREAMING (~50Hz)
  if (millis() - lastTelemetryTime >= TELEMETRY_INTERVAL_MS)
  {
    lastTelemetryTime = millis();

    // Normalize current position back to -1.0 to 1.0
    float currentNormalized = (float)encoderPosition / (float)MAX_COUNTS;

    // Format: "FB:<position>,PWM:<signal>"
    Serial.print("FB:");
    Serial.print(currentNormalized, 4);
    Serial.print(",PWM:");
    Serial.println(controlSignal);
  }
}
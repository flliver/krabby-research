#pragma once
#include <Arduino.h>
#include <EEPROM.h>
#include "command.h"
#include "hall_hw.h"

// Linear actuator controller (w/ potentiometer feedback)
class LinearActuator
{
public:
    struct ControlConfig
    {
        int pwmRampStep = 5;     // how much to change PWM per ramp step, higher will cause motor to accelerate faster
        int rampIntervalMs = 10; // time in millis between ramp steps
        int pwmDeadband = 20;    // PWM values below this are treated as zero, to avoid motor "creep"
        int pwmErrDeadband = 10; // Position error below which no PWM is applied, to avoid motor oscillation near target
        float Kp = 2.0;          // proportional gain applied to position error to derive desired PWM, PWM is set to max((targetPos - currentPos) * Kp, 255)
        float alphaPot = 0.15;    // Smoothing factor used to calculate potentiometer average (0.1 - 1.0)
        float alphaIS = 0.10;     // Smoothing factor used to calculate current sense average (0.1 - 1.0)

        ControlConfig() = default;
        ControlConfig(int rampStep, int intervalMs, int deadband, int errDeadband, float kp)
            : pwmRampStep(rampStep), rampIntervalMs(intervalMs), pwmDeadband(deadband), pwmErrDeadband(errDeadband), Kp(kp) {}
    };

    const char *name;  // Short joint name/id (e.g. "LHY" = "Left Hip Yaw")
    // PIN ASSIGNMENTS
    const int pinPwmR; // Sending on PWM_R defines desired motor voltage in the right/extend direction (note: Only send one of PWM_R or PWM_L at a time to avoid motor chatter/damage)
    const int pinPwmL; // Sending on PWM_L defines desired motor voltage in the left/retract direction (note: Only send one of PWM_R or PWM_L at a time to avoid motor chatter/damage)
    const int pinEn;   // Single enable (HIGH while driving)
    const int pinIS;   // Analog current sense pin, reads motor current from H-Bridge as a value between 0 (0 Amp) and 1023 (Max motor Amps, e.g. ~8A @ 12V for common linear actuators)
    const int pinPot;  // Analog potentiometer pin, reads actuator position as a value between 0 (fully retracted) and 1023 (fully extended)
    const int8_t hallSlot; // 0–5 = HallA index for this joint (-1 = no Hall telemetry)

    // Motion limits (raw ADC 0-1023), setting minStop higher than 0 will limit retraction, setting maxStop lower than 1023 will limit extension
    // Typically set to slightly less than max allowed by mechanical endstops to avoid motor damage
    int minStop = 0;
    int maxStop = 1023;

    // Control states
    int currentPwm = 0;              // Current PWM being applied to motor (-255 to 255)
    int currentTarget = 0;           // Target position (raw ADC); only used when hasTarget is true
    bool hasTarget = false;          // True only after a T (target) command; if false, motor stays idle
    unsigned long lastRampTime = 0;  // Last time PWM ramp was updated, in millis, used along with rampIntervalMs to control ramp timing
    float avgPot = 0.0;              // Global state variable to track smoothed potentiometer value
    float avgIS = 0.0;               // Global state variable to track smoothed current sense value


    // TODO: This should accept a name and a 'SlotConfig' struct for pin assignment, so we can reuse the pin config w/ different actuator names (on different leader/follower boards)
    LinearActuator(const char *n, int pR, int pL, int en, int isPin, int pot, int8_t hallIdx = -1)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEn(en), pinIS(isPin), pinPot(pot), hallSlot(hallIdx) {}
    void setControlConfig(const ControlConfig &cfg) { controlConfig = cfg; }

    void init()
    {
        // Configure pin modes
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEn, OUTPUT);
        pinMode(pinIS, INPUT);
        pinMode(pinPot, INPUT);

        // Safe startup: explicitly drive PWM low and EN low.
        // EN will only be driven HIGH later when we actually command motion
        // via driveActuator() (in update() / manualDrive()).
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        digitalWrite(pinEn, LOW);

        // Initialize averaging
        avgPot = analogRead(pinPot);
        avgIS = analogRead(pinIS);
        hasTarget = false; // No target until host sends T command
    }

    // Called during update to calculate new smoothed sensor readings, called internally on a fixed interval to exponentially average pot/IS readings
    void updateSensors()
    {
        int rawPot = analogRead(pinPot);
        int rawIS = analogRead(pinIS);

        // Exponential Moving Average
        avgPot = (avgPot * (1.0 - controlConfig.alphaPot)) + (rawPot * controlConfig.alphaPot);
        avgIS = (avgIS * (1.0 - controlConfig.alphaIS)) + (rawIS * controlConfig.alphaIS);
    }

    // Returns normalized position [0.0,1.0], where 0.0 = minStop, 1.0 = maxStop
    float getPos()
    {
        float range = maxStop - minStop;
        if (range == 0)
            return 0.5;
        return ((int)avgPot - minStop) / range;
    }

    int getRawPos() { return (int)avgPot; } // Returns smoothed RAW value

    // Set position target (T command only). Only this sets hasTarget = true.
    void setTarget(float val)
    {
        val = constrain(val, 0.0, 1.0);
        currentTarget = minStop + (int)(val * (maxStop - minStop));
        hasTarget = true;
    }

    // Hold: just stop the motor. No target is set or updated.
    void stopMotor()
    {
        currentPwm = 0;
        driveActuator(0, controlConfig.pwmDeadband);
        hasTarget = false;
    }

    // Jog: direct PWM. Does not set or clear target; when pwm is 0 we just stop.
    void manualDrive(int pwm)
    {
        pwm = constrain(pwm, -255, 255);
        if (pwm == 0)
        {
            currentPwm = 0;
            driveActuator(0, controlConfig.pwmDeadband);
        }
        else
        {
            driveActuator(pwm, 0);
            currentPwm = pwm;
        }
    }

    // Drives actuator to desired position using controlConfig; call frequently in main loop
    void update()
    {
        updateSensors(); // Always update sensors to recalculate avgPot/avgIS

        // No target: motor chills electrically. manualDrive() directly sets PWM/EN
        // when jogging, and in the no-target state we do not override that here.
        if (!hasTarget)
            return;

        int error = currentTarget - getRawPos();
        if (abs(error) < controlConfig.pwmErrDeadband)
            error = 0;

        int desiredPwm = (int)(error * controlConfig.Kp);
        desiredPwm = constrain(desiredPwm, -255, 255);

        // Ramping Logic
        if (millis() - lastRampTime >= (unsigned long)controlConfig.rampIntervalMs)
        {
            lastRampTime = millis();
            if (currentPwm < desiredPwm)
            {
                currentPwm += controlConfig.pwmRampStep;
                if (currentPwm > desiredPwm)
                    currentPwm = desiredPwm;
            }
            else if (currentPwm > desiredPwm)
            {
                currentPwm -= controlConfig.pwmRampStep;
                if (currentPwm < desiredPwm)
                    currentPwm = desiredPwm;
            }
        }
        driveActuator(currentPwm, controlConfig.pwmDeadband);
    }

    // Helper to detect stall (used in calibration)
    // Returns true if motor is powered but position hasn't changed for 'timeout' ms
    bool isStalled(unsigned long timeout)
    {
        static int lastPos = -1;
        static unsigned long lastMoveTime = 0;

        if (abs(currentPwm) < 50)
        { // Not trying to move
            lastMoveTime = millis();
            return false;
        }

        if (abs(getRawPos() - lastPos) > 2)
        { // Moved
            lastPos = getRawPos();
            lastMoveTime = millis();
            return false;
        }

        if (millis() - lastMoveTime > timeout)
            return true;
        return false;
    }

    // JT wire format: "<role>; <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <hallEdges>;"
    // e.g. 'FRONT; FLHY 0.123 0 12 1 1 0 120 0; FRHY 0.234 0 13 1 1 0 130 0; ...'
    // Keep in sync with firmware/interfaces/joint_telemetry.py
    // Keeping it super simple to avoid any string parsing and external library overhead
    void printTelemetry(Print& out) const
    {
        out.print(name);
        out.print(' ');
        out.print(getPos(), 3);
        out.print(' ');
        out.print((int)avgPot);
        out.print(' ');
        out.print((int)avgIS);
        out.print(' ');
        int en = digitalRead(pinEn);
        out.print(en);
        out.print(' ');
        out.print(en);
        out.print(' ');
        out.print(currentPwm < 0 ? abs(currentPwm) : 0);
        out.print(' ');
        out.print(currentPwm > 0 ? currentPwm : 0);
        out.print(' ');
        if (hallSlot >= 0 && hallSlot < 6)
            out.print(hallHwGetEdgeCount((uint8_t)hallSlot));
        else
            out.print(0);
    }

private:
    // Helper to drive actuator with given PWM, deadband is normally from controlConfig, but is optionally prvoided to bypass deadband during manual drive
    // 0 PWM = stop, Positive PWM = extend, Negative PWM = retract
    void driveActuator(int pwm, int pwmDeadband)
    {
        if (abs(pwm) < pwmDeadband)
        {
            digitalWrite(pinEn, LOW);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, 0);
        }
        else if (pwm < 0)
        {
            digitalWrite(pinEn, HIGH);
            analogWrite(pinPwmR, 0);
            analogWrite(pinPwmL, abs(pwm));
        }
        else
        {
            digitalWrite(pinEn, HIGH);
            analogWrite(pinPwmR, pwm);
            analogWrite(pinPwmL, 0);
        }
    }

    ControlConfig controlConfig;
};

class ActuatorManager
{
public:
    ActuatorManager(LinearActuator **actsArray, size_t actsCount)
        : actuators(actsArray), count(actsCount) {}

    void initAll()
    {
        for (size_t i = 0; i < count; i++)
            actuators[i]->init();
    }

    void handleJog(String name, int pwm)
    {
        // TODO: Improve brute force O(N) lookup
        for (size_t i = 0; i < count; i++)
        {
            if (String(actuators[i]->name) == name)
            {
                actuators[i]->manualDrive(pwm);
                return;
            }
        }
    }

    void updateAll()
    {
        if (calState != CAL_IDLE) // Run calibration logic instead of normal PID
        {
            updateCalibration(); 
        }
        else
        {
            for (size_t i = 0; i < count; i++)
                actuators[i]->update();
        }
    }

    void applyCommands(const Command *cmds, size_t cmdCount)
    {
        // TODO: This is O(N^2), but N is small so probably ok for now. Would need to add a map for larger actuator sets.
        for (size_t i = 0; i < cmdCount; i++)
        {
            const auto &cmd = cmds[i];
            for (size_t j = 0; j < count; j++)
            {
                if (cmd.name == actuators[j]->name)
                {
                    actuators[j]->setTarget(cmd.val);
                    break;
                }
            }
        }
    }

    void holdAll()
    {
        // For now, "hold" means fully de‑energize all joints:
        // EN low and PWM 0 on every actuator. This avoids any
        // PID activity that could move other joints when the
        // user expects everything to stay still.
        for (size_t i = 0; i < count; i++)
        {
            actuators[i]->stopMotor();
        }
    }

    void printTelemetry(Print& out) const
    {
        for (size_t i = 0; i < count; i++)
        {
            if (i) out.print(';'); // Only print semicolons between joints, not at the end
            actuators[i]->printTelemetry(out);
        }
        out.println();
    }

    // ==================================================
    // AUTO-CALIBRATION & PERSISTENCE
    // ==================================================
    enum CalState
    {
        CAL_IDLE,
        CAL_START,
        CAL_YAW_L_MIN,
        CAL_YAW_L_MAX,
        CAL_YAW_L_CENTER,
        CAL_YAW_R_MIN,
        CAL_YAW_R_MAX,
        CAL_YAW_R_CENTER,
        // Left Leg Sequence
        CAL_LHL_MIN,
        CAL_LKL_MAX,
        CAL_LKL_MIN,
        CAL_LHL_MAX,
        // Right Leg Sequence
        CAL_RHL_MIN,
        CAL_RKL_MAX,
        CAL_RKL_MIN,
        CAL_RHL_MAX,
        CAL_FINISH
    };

    CalState calState = CAL_IDLE;
    unsigned long stateTimer = 0;

    // Struct to save to EEPROM
    struct CalData
    {
        // TODO: Should be stored with Joint information, so that when joints change this changes, not hardcoded here
        int minVals[6];
        int maxVals[6];
        int magic; // 0xDEADBEEF to check validity
    };

    void startAutoCalibration()
    {
        calState = CAL_START;
        stateTimer = millis();
        Serial.println("Starting Auto-Calibration Sequence...");
    }

    void updateCalibration()
    {
        // TODO: Fix hardcoded actuator order, store actuator naming information in EEPROM struct
        // Helper lambda to get actuator by index (Hardcoded order: LHY, LHL, LKL, RHY, RHL, RKL)
        // 0=LHY, 1=LHL, 2=LKL, 3=RHY, 4=RHL, 5=RKL
        auto drive = [&](int idx, int pwm)
        { actuators[idx]->manualDrive(pwm); };
        auto isStalled = [&](int idx)
        { return actuators[idx]->isStalled(250); }; // 250ms stall check
        auto saveMin = [&](int idx)
        { actuators[idx]->minStop = actuators[idx]->getRawPos(); };
        auto saveMax = [&](int idx)
        { actuators[idx]->maxStop = actuators[idx]->getRawPos(); };

        // Simple State Machine
        switch (calState)
        {
        case CAL_START:
            calState = CAL_YAW_L_MIN;
            break;

        // --- YAWS FIRST ---
        case CAL_YAW_L_MIN:
            drive(0, -150); // Retract LHY
            if (isStalled(0))
            {
                saveMin(0);
                calState = CAL_YAW_L_MAX;
            }
            break;
        case CAL_YAW_L_MAX:
            drive(0, 150); // Extend LHY
            if (isStalled(0))
            {
                saveMax(0);
                calState = CAL_YAW_L_CENTER;
            }
            break;
        case CAL_YAW_L_CENTER:
            drive(0, 0); // Stop
            calState = CAL_YAW_R_MIN;
            break;

        case CAL_YAW_R_MIN:
            drive(3, -150); // Retract RHY
            if (isStalled(3))
            {
                saveMin(3);
                calState = CAL_YAW_R_MAX;
            }
            break;
        case CAL_YAW_R_MAX:
            drive(3, 150); // Extend RHY
            if (isStalled(3))
            {
                saveMax(3);
                calState = CAL_YAW_R_CENTER;
            }
            break;
        case CAL_YAW_R_CENTER:
            drive(3, 0);
            calState = CAL_LHL_MIN;
            break;

        // --- LEFT LEG SEQUENCE (Hip Up -> Knee Out -> Knee In -> Hip Down) ---
        case CAL_LHL_MIN: // Hip Retract (Up)
            drive(1, -200);
            if (isStalled(1))
            {
                saveMin(1);
                calState = CAL_LKL_MAX;
            }
            break;
        case CAL_LKL_MAX: // Knee Extend (Out)
            drive(2, 200);
            if (isStalled(2))
            {
                saveMax(2);
                calState = CAL_LKL_MIN;
            }
            break;
        case CAL_LKL_MIN: // Knee Retract (In)
            drive(2, -200);
            if (isStalled(2))
            {
                saveMin(2);
                calState = CAL_LHL_MAX;
            }
            break;
        case CAL_LHL_MAX: // Hip Extend (Tuck)
            drive(1, 200);
            if (isStalled(1))
            {
                saveMax(1);
                calState = CAL_RHL_MIN;
            }
            break;

        // --- RIGHT LEG SEQUENCE ---
        case CAL_RHL_MIN:
            drive(4, -200);
            if (isStalled(4))
            {
                saveMin(4);
                calState = CAL_RKL_MAX;
            }
            break;
        case CAL_RKL_MAX:
            drive(5, 200);
            if (isStalled(5))
            {
                saveMax(5);
                calState = CAL_RKL_MIN;
            }
            break;
        case CAL_RKL_MIN:
            drive(5, -200);
            if (isStalled(5))
            {
                saveMin(5);
                calState = CAL_RHL_MAX;
            }
            break;
        case CAL_RHL_MAX:
            drive(4, 200);
            if (isStalled(4))
            {
                saveMax(4);
                calState = CAL_FINISH;
            }
            break;

        case CAL_FINISH:
            // Stop all
            for (int i = 0; i < 6; i++)
                actuators[i]->manualDrive(0);
            saveCalibration(); // Write to EEPROM
            calState = CAL_IDLE;
            Serial.println("CALIBRATION COMPLETE & SAVED.");
            break;

        default:
            calState = CAL_IDLE;
            break;
        }
    }

    void saveCalibration()
    {
        CalData data;
        data.magic = 0xDEADBEEF;
        for (int i = 0; i < count; i++)
        {
            data.minVals[i] = actuators[i]->minStop;
            data.maxVals[i] = actuators[i]->maxStop;
        }
        EEPROM.put(0, data);
        Serial.println("Limits saved to EEPROM.");
    }

    void loadCalibration()
    {
        CalData data;
        EEPROM.get(0, data);
        if (data.magic == 0xDEADBEEF)
        {
            for (int i = 0; i < count && i < 6; i++)
            {
                actuators[i]->minStop = data.minVals[i];
                actuators[i]->maxStop = data.maxVals[i];
            }
            Serial.println("Calibration loaded from EEPROM.");
        }
        else
        {
            Serial.println("No EEPROM calibration found. Using defaults.");
        }
    }

private:
    LinearActuator **actuators;
    size_t count;
};
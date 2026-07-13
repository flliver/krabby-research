#pragma once
#include <Arduino.h>
#include "command.h"
#include "eeprom_layout.h"
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
    // Returns true if motor is powered but position hasn't changed for 'timeout' ms.
    // State is per-actuator (member, not static): shared state leaks stale
    // positions between actuators or across sweep reversals.
    int stallLastPos = -1;
    unsigned long stallLastMoveTime = 0;

    bool isStalled(unsigned long timeout)
    {
        if (abs(currentPwm) < 50)
        { // Not trying to move
            stallLastMoveTime = millis();
            return false;
        }

        if (abs(getRawPos() - stallLastPos) > 2)
        { // Moved
            stallLastPos = getRawPos();
            stallLastMoveTime = millis();
            return false;
        }

        if (millis() - stallLastMoveTime > timeout)
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
        for (size_t i = 0; i < count; i++)
            actuators[i]->update();
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
    // PER-JOINT CALIBRATION (M17 Task 2)
    // ==================================================
    // Load persisted limits into the actuators. Called once after initAll();
    // slots without both stops recorded (or with a degenerate range) keep the
    // actuator's full-range defaults.
    void loadCalibration()
    {
        jointCalLoad(cal);
        for (size_t i = 0; i < count && i < JOINTCAL_COUNT; i++)
        {
            if (calComplete(i))
            {
                actuators[i]->minStop = cal.minStop[i];
                actuators[i]->maxStop = cal.maxStop[i];
            }
        }
    }

    // Calibrate ONE joint. dir "" sweeps both stops: retract until the pot
    // stops changing, record minStop; extend the same way, record maxStop.
    // dir "retract"/"extend" records ONLY that stop — for joints whose full
    // sweep isn't possible in the robot's current stance (e.g. extending a hip
    // presses the foot into the ground and lifts the chassis instead of
    // reaching the stop). Limits are persisted to EEPROM immediately, but only
    // applied to the live actuator once BOTH stops have been recorded.
    // BLOCKING (up to ~2x CAL_TIMEOUT_MS) — bench-time only; telemetry and
    // commands pause while it runs. Replies "CAL <name> <min> <max> saved",
    // "CAL <name> <dir> <val> saved", or "CAL <name> FAIL <why>" on `out`.
    // Returns false if this board doesn't own the joint (not an error: the
    // leader broadcasts C to all boards and only the owner acts). An unknown
    // dir is dropped silently — the SDK validates client-side.
    bool calibrateJoint(const String &name, const String &dir, Print &out)
    {
        for (size_t i = 0; i < count; i++)
        {
            if (String(actuators[i]->name) != name)
                continue;
            if (dir.length())
                return calibrateOneStop(i, dir, out);
            int mn = sweepToStop(actuators[i], -CAL_PWM);
            // Second sweep starts at the stop the first just found — require
            // real motion before its stall can be believed (see sweepToStop).
            int mx = (mn >= 0) ? sweepToStop(actuators[i], CAL_PWM, true) : -1;
            out.print("CAL ");
            out.print(name);
            if (mn < 0 || mx < 0)
                out.println(" FAIL no_stop");        // never stalled: no pot signal, or free-spinning (yaw)
            else if (abs(mx - mn) < CAL_MIN_RANGE)
                out.println(" FAIL range_too_small"); // stops indistinguishable: jammed or pot not tracking
            else
            {
                actuators[i]->minStop = mn;
                actuators[i]->maxStop = mx;
                cal.minStop[i] = mn;
                cal.maxStop[i] = mx;
                cal.flags[i] = JOINTCAL_FLAG_BOTH;
                jointCalSave(cal);
                out.print(' '); out.print(mn);
                out.print(' '); out.print(mx);
                out.println(" saved");
            }
            return true;
        }
        return false;
    }

private:
    // Gentle by design: low PWM, and the sweep gives up after CAL_TIMEOUT_MS
    // instead of pushing harder (the knee can be damaged by a hard shove).
    // CAL_PWM must be >= 50 or isStalled() treats the joint as idle.
    static const int CAL_PWM = 120;
    static const unsigned long CAL_TIMEOUT_MS = 30000;  // bench-measured ~15 s full-travel at CAL_PWM; 2x headroom
    static const unsigned long CAL_STALL_MS = 400;      // pot quiet this long = at the stop
    static const int CAL_MIN_RANGE = 100;               // sane travel spans far more raw ADC than this
    static const int CAL_MIN_MOVE = 30;                 // pot counts a sweep must cover before a stall is believed

    // Both stops recorded and distinguishable — safe to apply to the live actuator.
    bool calComplete(size_t i) const
    {
        return (cal.flags[i] & JOINTCAL_FLAG_BOTH) == JOINTCAL_FLAG_BOTH
            && abs((int)cal.maxStop[i] - (int)cal.minStop[i]) >= CAL_MIN_RANGE;
    }

    // Directional calibration: sweep toward one stop and record just that one.
    // Replies "CAL <name> <dir> <val> saved" / "CAL <name> FAIL no_stop".
    bool calibrateOneStop(size_t i, const String &dir, Print &out)
    {
        bool retract = dir == "retract";
        if (!retract && dir != "extend")
            return true;  // owned, but unknown dir token: SDK validates, drop silently
        int v = sweepToStop(actuators[i], retract ? -CAL_PWM : CAL_PWM);
        out.print("CAL ");
        out.print(actuators[i]->name);
        if (v < 0)
            out.println(" FAIL no_stop");
        else
        {
            if (retract) { cal.minStop[i] = v; cal.flags[i] |= JOINTCAL_FLAG_MIN; }
            else         { cal.maxStop[i] = v; cal.flags[i] |= JOINTCAL_FLAG_MAX; }
            if (calComplete(i))
            {
                actuators[i]->minStop = cal.minStop[i];
                actuators[i]->maxStop = cal.maxStop[i];
            }
            jointCalSave(cal);
            out.print(' '); out.print(dir);
            out.print(' '); out.print(v);
            out.println(" saved");
        }
        return true;
    }

    // Drive one actuator until its pot stops changing; return the resting raw
    // pot value, or -1 on timeout. Blocking. Pumps updateSensors() itself —
    // loop()'s updateAll() isn't running while we block, and without fresh
    // sensor reads avgPot freezes and isStalled() fires instantly on stale data.
    //
    // requireMotion: don't believe a stall until the pot has moved CAL_MIN_MOVE
    // from where the sweep began. Set on the SECOND sweep of a full calibration,
    // which by construction starts at the opposite stop — reversing off a hard
    // stop (gear lash, chassis load) can hold the pot quiet past the grace
    // period and fake a stall at the same value the first sweep recorded.
    // First/directional sweeps leave it false: legitimately starting at the
    // target stop should record immediately.
    int sweepToStop(LinearActuator *act, int pwm, bool requireMotion = false)
    {
        act->stopMotor();   // clears any position target so update paths can't fight the sweep
        act->isStalled(1);  // |pwm|<50 path resets this actuator's stall clock before we start
        int startPos = act->getRawPos();
        unsigned long start = millis();
        while (millis() - start < CAL_TIMEOUT_MS)
        {
            act->updateSensors();
            act->manualDrive(pwm);
            bool armed = !requireMotion
                         || abs(act->getRawPos() - startPos) > CAL_MIN_MOVE;
            // Grace period so the motor can start moving before stall counts.
            if (armed && millis() - start > 600 && act->isStalled(CAL_STALL_MS))
            {
                act->manualDrive(0);
                return act->getRawPos();
            }
            delay(10);
        }
        act->manualDrive(0);
        return -1;
    }

    JointCalBlock cal;
    LinearActuator **actuators;
    size_t count;
};
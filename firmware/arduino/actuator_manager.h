#pragma once
#include <Arduino.h>
#include <EEPROM.h>
#include "command.h"
#include "hall_hw.h"
#include "eeprom_layout.h"  // SensorType, JointCal, JointCalBlock + load (M17 Task 2)

// Runtime calibration state (NOT persisted — re-derived every boot, M17 Task 2 §6.5).
// Pot is absolute → FULL on boot. Hall is incremental → PARTIAL on boot (span known,
// absolute position unknown) until the joint self-heals against an end-stop.
enum CalibrationState : uint8_t {
    CAL_STATE_UNCALIBRATED = 0,
    CAL_STATE_PARTIAL      = 1,
    CAL_STATE_FULL         = 2,
};

inline const char* calStateName(uint8_t s)
{
    switch (s) {
        case CAL_STATE_FULL:    return "FULL";
        case CAL_STATE_PARTIAL: return "PARTIAL";
        default:                return "UNCAL";
    }
}

// Linear actuator controller (pot or Hall feedback)
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
        unsigned long jogWatchdogMs = 300; // Jog auto-stop window: coast a jogged motor if no new jog command arrives within this many ms (host-crash / cable-pull safety). 0 disables.

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
    unsigned long lastJogMs = 0;     // millis() of the last manualDrive() (jog) command; drives the jog watchdog in update()
    float avgPot = 0.0;              // Global state variable to track smoothed potentiometer value
    float avgIS = 0.0;               // Global state variable to track smoothed current sense value

    // --- Per-joint calibration (M17 Task 2) ---
    // Defaults give the legacy minStop/maxStop pot path with no flip; applyJointCal()
    // overrides these once a recorded JointCal is loaded from EEPROM (calValid=true).
    uint8_t  sensorType     = SENSOR_POT;  // SensorType auto-detected during cal
    uint8_t  sensorReversed = 0;           // 1 = sensor reads inverted vs drive direction
    bool     calValid       = false;       // true once applyJointCal() loads min/max
    uint16_t calPotMin      = 0;           // raw ADC at retract-stop (pre-flip)
    uint16_t calPotMax      = 1023;        // raw ADC at extend-stop  (pre-flip)
    int32_t  calHallMin     = 0;           // signed count at retract-stop
    int32_t  calHallMax     = 0;           // signed count at extend-stop

    // --- Boot state + Hall self-heal (M17 Task 2 §6.5 / 2h) ---
    uint8_t  calibrationState = CAL_STATE_UNCALIBRATED;  // runtime; not persisted
    int32_t  hallOffset       = 0;         // added to the boot-relative Hall count to anchor it
    bool     notCalErrSent    = false;     // throttle: one not_calibrated ERR per uncal period
    static constexpr unsigned long SELF_HEAL_STALL_MS = 400;  // end-stop dwell before anchoring


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
        // On a Hall joint the pot pin (A0-A5) carries HallB, which the quadrature ISR
        // samples digitally via PINF. analogRead()-ing it here disturbs that pin (ADC
        // sample-and-hold) and corrupts the direction decode, so skip it — avgPot is
        // meaningless for a Hall joint anyway (getPos() uses the signed Hall count).
        if (sensorType != SENSOR_HALL)
        {
            int rawPot = analogRead(pinPot);
            avgPot = (avgPot * (1.0 - controlConfig.alphaPot)) + (rawPot * controlConfig.alphaPot);
        }
        int rawIS = analogRead(pinIS);
        avgIS = (avgIS * (1.0 - controlConfig.alphaIS)) + (rawIS * controlConfig.alphaIS);
    }

    // --- Sensor abstraction (M17 Task 2 §2e/§3) ---
    // Full-scale span used by the direction-flip math: 1023 for a pot, the recorded
    // count range for a Hall.
    int32_t sensorFullScale() const
    {
        return (sensorType == SENSOR_HALL) ? (calHallMax - calHallMin) : 1023;
    }

    // Hall signed position: A/B quadrature count from hall_hw (Task 2 §5). Tracks
    // direction (climbs extending, falls retracting), unlike the legacy edge count.
    int32_t hallSignedCount() const
    {
        if (hallSlot < 0) return 0;
        return hallHwGetSignedCount((uint8_t)hallSlot);
    }

    // Apply the calibrated direction-flip to a raw sensor value (Task 2 §2b).
    int32_t applyFlip(int32_t raw) const
    {
        return sensorReversed ? (sensorFullScale() - raw) : raw;
    }

    // Flip-corrected raw position from whichever sensor is wired. avgPot / the Hall
    // count themselves are never rewritten — the flip is applied only here, at the
    // read site, so the raw smoothed values stay inspectable.
    int32_t getRawPos()
    {
        // Hall is boot-relative; hallOffset anchors it to the cal frame once self-healed
        // (0 until then, and 0 immediately after a cal since that IS the live frame).
        int32_t raw = (sensorType == SENSOR_HALL) ? (hallSignedCount() + hallOffset)
                                                  : (int32_t)avgPot;
        return applyFlip(raw);
    }

    // Normalized position [0.0, 1.0]: 0.0 at retract-stop, 1.0 at extend-stop, for
    // whichever sensor is wired, with the direction-flip applied. Falls back to the
    // legacy minStop/maxStop pot path until a JointCal is loaded (calValid).
    float getPos()
    {
        if (!calValid)
        {
            float range = maxStop - minStop;
            if (range == 0)
                return 0.5;
            return ((int)avgPot - minStop) / range;
        }
        int32_t pos = getRawPos();
        int32_t lo  = applyFlip(sensorType == SENSOR_HALL ? calHallMin : (int32_t)calPotMin);
        int32_t hi  = applyFlip(sensorType == SENSOR_HALL ? calHallMax : (int32_t)calPotMax);
        int32_t range = hi - lo;
        if (range == 0)
            return 0.5f;
        float p = (float)(pos - lo) / (float)range;
        return p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
    }

    // Load a recorded per-joint cal (from an EEPROM JointCalBlock) into this actuator.
    // Until called (or with no valid EEPROM cal) the joint uses the legacy pot path.
    // liveFrame: true when called right after a fresh cal (the live Hall count is still
    // in the cal's frame → already anchored). false on EEPROM load at boot (the Hall
    // count frame reset → Hall comes up PARTIAL and must self-heal).
    void applyJointCal(const JointCal& jc, bool liveFrame = false)
    {
        if (!jc.calibrated) { calValid = false; calibrationState = CAL_STATE_UNCALIBRATED; return; }
        sensorType     = jc.sensorType;
        sensorReversed = jc.sensorReversed;
        calPotMin      = jc.potMin;
        calPotMax      = jc.potMax;
        calHallMin     = jc.hallMin;
        calHallMax     = jc.hallMax;
        calValid       = true;
        hallOffset     = 0;
        notCalErrSent  = false;
        // Pot is absolute by physics; a just-finished cal is in-frame; an EEPROM-loaded
        // Hall joint knows only its span until it touches an end-stop (§6.5).
        calibrationState = (sensorType == SENSOR_POT || liveFrame)
                               ? CAL_STATE_FULL : CAL_STATE_PARTIAL;
    }

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
        lastJogMs = millis();   // pet the jog watchdog on every jog refresh (incl. the pwm==0 stop)
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

        // No target: the motor is either idle or being jogged open-loop by manualDrive().
        // Jog watchdog: a jog is momentary and must not outlive its commander. If a jog is
        // active (currentPwm != 0) and no fresh jog command has arrived within jogWatchdogMs,
        // coast the motor — this is what stops a runaway when the host crashes, the process is
        // killed, or the cable is pulled mid-jog. Position targets (hasTarget) are exempt: they
        // latch deliberately and settle near their setpoint rather than running away.
        if (!hasTarget)
        {
            // Hall self-heal (Task 2 §6.5 / 2h): a PARTIALLY_CALIBRATED joint anchors its
            // boot-relative count the first time a jog drives it into an end-stop. We know
            // the true count there (calHallMax extending, calHallMin retracting), so set
            // hallOffset to map the live count onto it. (The spec also gates this on
            // avgIS≈0 to tell an end-stop from a mid-travel jam; the current-sense channel
            // reads ~0 on this bench, so that split is deferred until IS wiring is fixed.)
            if (calibrationState == CAL_STATE_PARTIAL && currentPwm != 0 &&
                isStalled(SELF_HEAL_STALL_MS))
            {
                hallOffset = ((currentPwm > 0) ? calHallMax : calHallMin) - hallSignedCount();
                calibrationState = CAL_STATE_FULL;
                notCalErrSent = false;
            }
            if (currentPwm != 0 && controlConfig.jogWatchdogMs > 0 &&
                millis() - lastJogMs > controlConfig.jogWatchdogMs)
            {
                manualDrive(0);  // coast
            }
            return;
        }

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

    // Stall-detection state — PER ACTUATOR (was function-static, which leaked across
    // joints and across the retract→extend transition, instant-stalling the 2nd sweep).
    // Call resetStall() when you start driving a fresh direction.
    long          stallLastPos      = 0;   // holds getRawPos() (int32_t — Hall counts exceed int)
    unsigned long stallLastMoveTime = 0;
    bool          stallInited       = false;  // explicit init flag — getRawPos() can be negative (Hall)

    void resetStall() { stallInited = false; stallLastMoveTime = millis(); }

    // Returns true if motor is powered but position hasn't changed for 'timeout' ms.
    bool isStalled(unsigned long timeout)
    {
        if (abs(currentPwm) < 50)
        { // Not trying to move
            stallLastMoveTime = millis();
            return false;
        }

        if (!stallInited || labs(getRawPos() - stallLastPos) > 6)
        { // Moved (or first reading after a reset). >6 tolerates a few EMI-induced Hall
          // counts at a hard stall so the timer isn't perpetually reset (real motion is
          // hundreds of counts/sweep, so this never masks genuine movement).
            stallLastPos = getRawPos();
            stallLastMoveTime = millis();
            stallInited = true;
            return false;
        }

        if (millis() - stallLastMoveTime > timeout)
            return true;
        return false;
    }

    // JT wire format: "<role>; <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <hallEdges> <calState>;"
    // e.g. 'FRONT; FLHY 0.123 0 12 1 1 0 120 0 2; FRHY 0.234 0 13 1 1 0 130 0 0; ...'
    // calState: 0=UNCAL, 1=PARTIAL (Hall, unanchored), 2=FULL. Keep in sync with
    // firmware/interfaces/joint_telemetry.py. Kept super simple (no string parsing / libs).
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
        out.print(' ');
        out.print(calibrationState);   // 0=UNCAL, 1=PARTIAL, 2=FULL
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
        loadJointCals();
    }

    // Load per-joint calibration from EEPROM (Task 2) and distribute it to actuators
    // by slot index. A blank/invalid block leaves every actuator on the legacy pot
    // path (calValid stays false), so an uncalibrated board behaves exactly as before.
    void loadJointCals()
    {
        JointCalBlock blk;
        if (!jointCalLoad(blk))
            return;
        for (size_t i = 0; i < count && i < JOINTCAL_SLOTS; i++)
            actuators[i]->applyJointCal(blk.joints[i]);
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

    // ===== Per-joint calibration (M17 Task 2 §2/§3) =====
    // Single-joint state machine: nudge to auto-detect sensor type + direction, sweep
    // to both end-stops, record min/max, persist one JointCal slot. Drives only the
    // target joint (others idle). Task 3's whole-robot sequence calls this per joint.
    enum JointCalState : uint8_t {
        JC_NUDGE_FWD_DRIVE, JC_NUDGE_FWD_EVAL,
        JC_NUDGE_REV_DRIVE, JC_NUDGE_REV_EVAL,
        JC_RETRACT, JC_EXTEND, JC_SAVE, JC_DONE,
    };

    static constexpr int           JC_NUDGE_PWM            = 120;  // detect nudge: must exceed these actuators' static friction (~100), not the spec's optimistic 30
    static constexpr int           JC_SWEEP_PWM            = 150;  // sweep-to-stop drive
    static constexpr unsigned long JC_NUDGE_MS             = 250;  // nudge drive duration
    static constexpr unsigned long JC_SETTLE_MS            = 50;   // settle before measuring
    static constexpr unsigned long JC_STALL_MS             = 250;  // isStalled() end-stop window
    static constexpr unsigned long JC_SWEEP_GRACE_MS       = 600;  // drive this long before stall-checking (let it accelerate off a stop)
    static constexpr int32_t       JC_NUDGE_THRESHOLD      = 20;   // raw ADC: pot moved
    static constexpr int32_t       JC_HALL_NUDGE_THRESHOLD = 4;    // counts: Hall moved
    static constexpr int32_t       JC_POT_MIN_SPAN         = 50;   // min retract..extend ADC span
    static constexpr int32_t       JC_HALL_MIN_SPAN        = 10;   // min retract..extend count span
    // Hall auto-detect is ON now that hallSignedCount() is real A/B quadrature (§5/2c):
    // the SIGNED count accumulates with direction, so genuine Hall motion stands out
    // while EMI on an unwired pin nets to ~0 — robust enough to distinguish a Hall
    // actuator from a pot one. (The earlier edge-count placeholder counted noise, which
    // is why this was temporarily disabled.)
    static constexpr bool          JC_HALL_DETECT          = true;

    void setErrorOutput(Print* p) { errOut = p; }  // where ERR <joint> <code> lines go
    bool jointCalActive() const { return jcActive; }

    // Q <name>: print this board's stored calibration for one joint (no-op if not ours).
    // Reads back from EEPROM so it reflects exactly what's persisted, including a failed
    // cal (cal 0). Reply line: "CAL <name> type <POT|HALL> rev <0|1> min <n> max <n> cal <0|1>".
    void queryCalByName(const String& name, Print& out)
    {
        for (size_t i = 0; i < count; i++)
            if (String(actuators[i]->name) == name) { printJointCal((uint8_t)i, out); return; }
    }

    void printJointCal(uint8_t idx, Print& out)
    {
        JointCalBlock blk;
        jointCalLoad(blk);  // invalid → zeroed (every slot cal 0)
        const JointCal& jc = blk.joints[idx];
        out.print("CAL ");
        out.print(actuators[idx]->name);
        out.print(jc.sensorType == SENSOR_HALL ? " type HALL" : " type POT");
        out.print(" rev "); out.print(jc.sensorReversed);
        if (jc.sensorType == SENSOR_HALL) {
            out.print(" min "); out.print(jc.hallMin);
            out.print(" max "); out.print(jc.hallMax);
        } else {
            out.print(" min "); out.print(jc.potMin);
            out.print(" max "); out.print(jc.potMax);
        }
        out.print(" cal "); out.print(jc.calibrated);
        // runtime calibration_state (Task 2 §6.5) — from the live actuator, not EEPROM
        out.print(" state "); out.println(calStateName(actuators[idx]->calibrationState));
    }

    // K <name>: full-sweep calibration of one joint by name (unknown name = no-op,
    // matching the silent-drop convention; the SDK validates client-side).
    void calibrateJointByName(const String& name)
    {
        for (size_t i = 0; i < count; i++)
            if (String(actuators[i]->name) == name) { calibrateJoint((uint8_t)i); return; }
    }

    // Begin calibrating actuator `idx`. Stops every joint first; only `idx` moves.
    void calibrateJoint(uint8_t idx)
    {
        if (idx >= count) return;
        for (size_t i = 0; i < count; i++) actuators[i]->manualDrive(0);
        jcIndex      = idx;
        jcActive     = true;
        jcState      = JC_NUDGE_FWD_DRIVE;
        jcTimer      = millis();
        jcPotBefore  = (int32_t)actuators[idx]->avgPot;
        jcHallBefore = actuators[idx]->hallSignedCount();
        jcSensorType = SENSOR_POT;
        jcReversed   = 0;
    }

    // Advance the cal state machine one tick (called from updateAll while jcActive).
    // Non-blocking: timed nudges + isStalled()-gated sweeps.
    void updateJointCal()
    {
        LinearActuator* a = actuators[jcIndex];
        switch (jcState)
        {
        case JC_NUDGE_FWD_DRIVE:
            a->manualDrive(JC_NUDGE_PWM);
            if (millis() - jcTimer >= JC_NUDGE_MS) { a->manualDrive(0); jcState = JC_NUDGE_FWD_EVAL; jcTimer = millis(); }
            break;
        case JC_NUDGE_FWD_EVAL:
            if (millis() - jcTimer < JC_SETTLE_MS) break;
            if (!jcEvalNudge(a, /*forward=*/true)) { jcState = JC_NUDGE_REV_DRIVE; jcTimer = millis(); }
            break;
        case JC_NUDGE_REV_DRIVE:
            a->manualDrive(-JC_NUDGE_PWM);
            if (millis() - jcTimer >= JC_NUDGE_MS) { a->manualDrive(0); jcState = JC_NUDGE_REV_EVAL; jcTimer = millis(); }
            break;
        case JC_NUDGE_REV_EVAL:
            if (millis() - jcTimer < JC_SETTLE_MS) break;
            if (!jcEvalNudge(a, /*forward=*/false)) {  // neither direction moved the sensor
                emitJointErr(jcIndex, "motor_did_not_move");
                a->manualDrive(0); jcActive = false; jcState = JC_DONE;
            }
            break;
        case JC_RETRACT:
            a->manualDrive(-JC_SWEEP_PWM);
            // Grace period: don't accept a stall until the actuator has had time to
            // accelerate off the start point, or a slow ramp reads as an early stop.
            if (millis() - jcTimer > JC_SWEEP_GRACE_MS && a->isStalled(JC_STALL_MS)) {
                a->manualDrive(0);
                jcPotMin  = (uint16_t)a->avgPot;
                jcHallMin = a->hallSignedCount();
                a->resetStall();  // fresh stall state for the extend sweep
                jcState = JC_EXTEND; jcTimer = millis();
            }
            break;
        case JC_EXTEND:
            a->manualDrive(JC_SWEEP_PWM);
            if (millis() - jcTimer > JC_SWEEP_GRACE_MS && a->isStalled(JC_STALL_MS)) {
                a->manualDrive(0);
                jcPotMax  = (uint16_t)a->avgPot;
                jcHallMax = a->hallSignedCount();
                jcState = JC_SAVE;
            }
            break;
        case JC_SAVE: {
            // Sweep-range sanity check: a real sweep crosses the joint's whole travel,
            // so the recorded span should be large. A tiny span means the sensor never
            // tracked the motion (e.g. a floating/intermittent pot) — the nudge can still
            // pass on noise, so this is the gate that stops us silently saving garbage.
            int32_t span = (jcSensorType == SENSOR_HALL)
                               ? labs(jcHallMax - jcHallMin)
                               : labs((int32_t)jcPotMax - (int32_t)jcPotMin);
            int32_t minSpan = (jcSensorType == SENSOR_HALL) ? JC_HALL_MIN_SPAN : JC_POT_MIN_SPAN;
            bool ok = span >= minSpan;

            JointCalBlock blk;
            jointCalLoad(blk);  // invalid → zero-inited; we overwrite this one slot
            JointCal& jc = blk.joints[jcIndex];
            jc.potMin = jcPotMin; jc.potMax = jcPotMax;
            jc.hallMin = jcHallMin; jc.hallMax = jcHallMax;
            jc.sensorType = jcSensorType; jc.sensorReversed = jcReversed;
            jc.calibrated = ok ? 1 : 0;   // failed range check → not trusted
            jointCalSave(blk);            // still record the values (cal=0) so GET shows them
            a->applyJointCal(jc, /*liveFrame=*/true);  // in-frame → FULL; cal=0 reverts to legacy
            if (!ok)
                emitJointErr(jcIndex, jcSensorType == SENSOR_HALL ? "hall_no_edges" : "pot_value_invalid");
            jcActive = false; jcState = JC_DONE;
            break;
        }
        default:
            jcActive = false;
            break;
        }
    }

    void emitJointErr(uint8_t idx, const char* code)
    {
        if (!errOut) return;
        errOut->print("ERR ");
        errOut->print(actuators[idx]->name);
        errOut->print(' ');
        errOut->println(code);
    }

    void jcBeginSweep()
    {
        actuators[jcIndex]->manualDrive(0);
        actuators[jcIndex]->resetStall();  // fresh stall state for the retract sweep
        jcState = JC_RETRACT;
        jcTimer = millis();
    }

    // Did the nudge move the sensor enough to detect type + direction? `forward` =
    // the nudge was the +extend drive (the sign convention flips for the retract nudge,
    // Task 2 §3 step 7). Sets sensorType/sensorReversed and begins the sweep on success.
    bool jcEvalNudge(LinearActuator* a, bool forward)
    {
        int32_t potDelta  = (int32_t)a->avgPot - jcPotBefore;
        int32_t hallDelta = a->hallSignedCount() - jcHallBefore;
        // Check HALL first: on a Hall actuator the shared A1 pin carries HallB (a square
        // wave that can masquerade as pot movement on avgPot), so the unambiguous signal
        // is the signed Hall count. On a pot actuator HallA is unwired → the signed count
        // nets ~0, so this falls through to the pot check.
        if (JC_HALL_DETECT && labs(hallDelta) > JC_HALL_NUDGE_THRESHOLD) {
            jcSensorType = SENSOR_HALL;
            jcReversed   = (forward ? (hallDelta < 0) : (hallDelta > 0)) ? 1 : 0;
            jcApplyDetectedSensor(a);
            jcBeginSweep();
            return true;
        }
        if (labs(potDelta) > JC_NUDGE_THRESHOLD) {
            jcSensorType = SENSOR_POT;
            jcReversed   = (forward ? (potDelta < 0) : (potDelta > 0)) ? 1 : 0;
            jcApplyDetectedSensor(a);
            jcBeginSweep();
            return true;
        }
        return false;
    }

    // Push the detected sensor type/flip onto the actuator *now*, before the sweep, so
    // getRawPos()/isStalled() read the correct sensor while sweeping (e.g. a Hall joint
    // must stall-detect on the signed count, not on avgPot — which is HallB noise there).
    void jcApplyDetectedSensor(LinearActuator* a)
    {
        a->sensorType     = jcSensorType;
        a->sensorReversed = jcReversed;
    }

    void updateAll()
    {
        if (jcActive)                  // single-joint calibration (M17 Task 2)
            updateJointCal();
        else if (calState != CAL_IDLE) // legacy multi-joint calibration
            updateCalibration();
        else
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
                    // A position target needs absolute position. A PARTIALLY_CALIBRATED
                    // Hall joint doesn't have it yet → drop the target + emit one
                    // not_calibrated ERR (Task 2 §6.5). Jogs (J/B) bypass this and are
                    // how the operator drives it to an end-stop to self-heal.
                    if (actuators[j]->calibrationState == CAL_STATE_PARTIAL)
                    {
                        if (!actuators[j]->notCalErrSent)
                        {
                            emitJointErr((uint8_t)j, "not_calibrated");
                            actuators[j]->notCalErrSent = true;
                        }
                    }
                    else
                    {
                        actuators[j]->setTarget(cmd.val);
                    }
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

    // Calibration limits are held in RAM only — they are not persisted to EEPROM.
    // EEPROM address 0 is reserved for the board config struct (EepromLayout, in
    // firmware/arduino/eeprom_layout.h), so these must not write there. Per-joint
    // calibration persistence is intended to be added as a field of that struct.
    void saveCalibration()
    {
        Serial.println("Calibration complete (held in RAM; not persisted).");
    }

    void loadCalibration()
    {
        // Nothing to load: calibration is not persisted (see saveCalibration).
    }

private:
    LinearActuator **actuators;
    size_t count;

    // Single-joint calibration state (M17 Task 2)
    Print*        errOut       = nullptr;  // ERR <joint> <code> sink (set by setErrorOutput)
    bool          jcActive     = false;
    uint8_t       jcIndex      = 0;
    JointCalState jcState      = JC_DONE;
    unsigned long jcTimer      = 0;
    int32_t       jcPotBefore  = 0;
    int32_t       jcHallBefore = 0;
    uint8_t       jcSensorType = SENSOR_POT;
    uint8_t       jcReversed   = 0;
    uint16_t      jcPotMin     = 0;
    uint16_t      jcPotMax     = 1023;
    int32_t       jcHallMin    = 0;
    int32_t       jcHallMax    = 0;
};
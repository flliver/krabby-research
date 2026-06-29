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
    int32_t currentTarget = 0;       // Target in getRawPos() units (calibrated frame, or raw ADC if uncal); only used when hasTarget
    float lastSetVal = 0.0f;         // last normalized [0,1] target passed to setTarget(); for atTarget()
    bool hasTarget = false;          // True only after a T (target) command; if false, motor stays idle
    unsigned long lastRampTime = 0;  // Last time PWM ramp was updated, in millis, used along with rampIntervalMs to control ramp timing
    unsigned long lastJogMs = 0;     // millis() of the last manualDrive() (jog) command; drives the jog watchdog in update()
    float avgPot = 0.0;              // Global state variable to track smoothed potentiometer value
    float avgIS = 0.0;               // Global state variable to track smoothed current sense value

    // --- Per-joint calibration (M17 Task 2) ---
    // Defaults give the legacy minStop/maxStop pot path with no flip; applyJointCal()
    // overrides these once a recorded JointCal is loaded from EEPROM (calValid=true).
    uint8_t  sensorType     = SENSOR_POT;  // fixed per-joint property, set at construction (HL/HY=Hall, KL=pot)
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
    // sType is the joint's FIXED position-sensor type — a property of the hardware on that
    // slot, not something discovered at runtime. HL/HY joints carry a Hall encoder, KL joints
    // a potentiometer. Calibration verifies the expected sensor moved; it never re-guesses type.
    LinearActuator(const char *n, int pR, int pL, int en, int isPin, int pot, int8_t hallIdx = -1, SensorType sType = SENSOR_POT)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEn(en), pinIS(isPin), pinPot(pot), hallSlot(hallIdx), sensorType((uint8_t)sType) {}
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
        // sensorType is fixed at construction; a stored cal recorded against a different
        // sensor type is stale (e.g. left from before HL/KL types were corrected) and its
        // min/max mean nothing for this sensor — reject it rather than trust garbage.
        if (jc.sensorType != sensorType) { calValid = false; calibrationState = CAL_STATE_UNCALIBRATED; return; }
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
    // Drive to a normalized [0,1] position. On a calibrated joint this maps onto the
    // recorded travel in the SAME flip-corrected frame getRawPos()/getPos() report, so a
    // target of 0.5 settles where the telemetry reads 0.5 (and 0.0/1.0 = the cal stops),
    // for either sensor type. Uncalibrated joints fall back to the legacy raw-ADC scale.
    void setTarget(float val)
    {
        val = constrain(val, 0.0, 1.0);
        if (calValid)
        {
            int32_t lo = applyFlip(sensorType == SENSOR_HALL ? calHallMin : (int32_t)calPotMin);
            int32_t hi = applyFlip(sensorType == SENSOR_HALL ? calHallMax : (int32_t)calPotMax);
            currentTarget = lo + (int32_t)(val * (hi - lo));
        }
        else
        {
            currentTarget = minStop + (int32_t)(val * (maxStop - minStop));
        }
        lastSetVal = val;
        hasTarget = true;
    }

    // True once the joint has driven within `tol` (normalized) of its target — the settle
    // test moveJointTo() / the Task-3 pose transitions poll. Only meaningful with a target.
    bool atTarget(float tol = 0.02f)
    {
        return hasTarget && fabs(getPos() - lastSetVal) <= tol;
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

        int32_t error = currentTarget - getRawPos();
        if (labs(error) < controlConfig.pwmErrDeadband)
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

    // --- Continuous runtime sensor-health monitoring (Task 2 §2f) ---
    // While a joint is driven during NORMAL operation but its sensor stops following,
    // return a one-shot reason code (nullptr when healthy or suppressed). The manager
    // turns this into an `ERR <joint> <code>` line. Throttled to one report per stall
    // event (re-arms when the joint moves again or drive stops). Legitimate end-stops
    // are NOT faults: a PARTIAL joint self-heals on stop contact (handled in update()),
    // and a FULL joint sitting at a known limit is expected travel, not a stall.
    static constexpr unsigned long HEALTH_STALL_MS       = 600;  // driven-but-pinned window (> SELF_HEAL_STALL_MS so self-heal anchors first)
    static constexpr int           JAM_CURRENT_THRESHOLD = 100;  // avgIS counts; ≳ this while pinned = pushing hard (jam) vs an unresponsive motor
    static constexpr float         HEALTH_AT_LIMIT_EPS   = 0.03; // normalized: within this of 0.0/1.0 = at a known end-stop
    bool healthErrSent = false;  // throttle: one ERR per stall event

    const char* checkRuntimeHealth()
    {
        if (currentPwm == 0)            { healthErrSent = false; return nullptr; }  // not driven → re-arm
        if (!isStalled(HEALTH_STALL_MS)){ healthErrSent = false; return nullptr; }  // sensor following → re-arm
        if (healthErrSent)               return nullptr;                            // already reported this event

        // Suppress legitimate end-stops. PARTIAL anchors via self-heal; a FULL joint at a
        // known limit is expected. (An UNCAL joint has no known limits, so a genuine stall
        // there is always reported.)
        if (calibrationState == CAL_STATE_PARTIAL) return nullptr;
        if (calibrationState == CAL_STATE_FULL) {
            float p = getPos();
            if (p <= HEALTH_AT_LIMIT_EPS || p >= 1.0f - HEALTH_AT_LIMIT_EPS) return nullptr;
        }

        healthErrSent = true;
        // High current while pinned = pushing against an obstacle (jam); low current = the
        // motor isn't engaging at all (broken wiring / disconnected motor / dead sensor).
        // NOTE: current sense reads ~0 on this bench (IS-line fault), so the jam branch is
        // dormant here — every runtime stall classifies as motor_did_not_move until IS is fixed.
        return (avgIS >= JAM_CURRENT_THRESHOLD) ? "motor_jammed" : "motor_did_not_move";
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
        JC_RETRACT, JC_EXTEND, JC_RETRACT_AGAIN, JC_SAVE, JC_DONE,
    };

    // Which end(s) a single calibrateJoint run sweeps. NONE = full both-ends sweep (an
    // operator's `calibrate-joint <name>`); RETRACT/EXTEND = one end-stop, parked there
    // (Task 3's whole-robot sequence cals one DOF at a time). Yaw left/right map onto
    // RETRACT/EXTEND at the wire-parse layer (parseCalDirection).
    enum CalDirection : uint8_t { CAL_DIR_NONE = 0, CAL_DIR_RETRACT, CAL_DIR_EXTEND };

    static constexpr int           JC_NUDGE_PWM            = 180;  // detect nudge: must break a joint OFF a hard stop (a prior cal step parks it there at JC_SWEEP_PWM=150), so > sweep; bench jog at 200 frees it, 120 did not
    static constexpr int           JC_SWEEP_PWM            = 150;  // sweep-to-stop drive
    static constexpr unsigned long JC_NUDGE_MS             = 250;  // nudge drive duration
    static constexpr unsigned long JC_SETTLE_MS            = 50;   // settle before measuring
    static constexpr unsigned long JC_STALL_MS             = 250;  // isStalled() end-stop window
    static constexpr unsigned long JC_SWEEP_GRACE_MS       = 600;  // drive this long before stall-checking (let it accelerate off a stop)
    static constexpr int32_t       JC_NUDGE_THRESHOLD      = 20;   // raw ADC: pot moved
    static constexpr int32_t       JC_HALL_NUDGE_THRESHOLD = 4;    // counts: a wired Hall's own motion
    // Bar for declaring an UNEXPECTED Hall present on a pot-declared joint. Must clear the
    // EMI a floating (unwired) HallA pin picks up under motor-drive current — measured to
    // occasionally exceed the 4-count motion threshold — while staying well under a real
    // Hall's ~20-30 counts/nudge, so a Hall in a pot slot is still caught (sensor_type_mismatch).
    static constexpr int32_t       JC_HALL_PRESENT_THRESHOLD = 15;
    static constexpr int32_t       JC_POT_MIN_SPAN         = 50;   // min retract..extend ADC span
    static constexpr int32_t       JC_HALL_MIN_SPAN        = 10;   // min retract..extend count span
    static constexpr int32_t       JC_HALL_DRIFT_TOL       = 4;    // counts: max |hallMin_2 - hallMin_1| over repeat sweeps (2c)

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

    // GET calibration (Task 4 §4): one positional line per joint for the whole board,
    //   CAL <joint> <POT|HALL> <min> <max> <reversed>
    // the raw JointCal tuple from EEPROM. Leaner than the labelled per-joint Q reply
    // (printJointCal) — this is the operator-facing after-the-fact inspection dump.
    void printAllCal(Print& out)
    {
        JointCalBlock blk;
        jointCalLoad(blk);  // invalid → zeroed
        for (size_t i = 0; i < count; i++)
        {
            const JointCal& jc = blk.joints[i];
            out.print("CAL ");
            out.print(actuators[i]->name);
            out.print(jc.sensorType == SENSOR_HALL ? " HALL " : " POT ");
            if (jc.sensorType == SENSOR_HALL) { out.print(jc.hallMin); out.print(' '); out.print(jc.hallMax); }
            else                              { out.print(jc.potMin);  out.print(' '); out.print(jc.potMax); }
            out.print(' ');
            out.println(jc.sensorReversed);
        }
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

    // K <name> [extend|retract|left|right]: calibrate one joint by name. Empty direction =
    // full both-ends sweep. The directional forms drive a single end-stop (Task 3 cals one
    // DOF at a time). Linear joints take extend/retract, yaw joints take left/right; a
    // mismatched pairing (or unknown name) is silently dropped per Task 1's no-reply
    // convention — the SDK validates client-side.
    void calibrateJointByName(const String& name, const String& dirStr = String())
    {
        for (size_t i = 0; i < count; i++) {
            if (String(actuators[i]->name) != name) continue;
            CalDirection dir;
            if (!parseCalDirection(name, dirStr, dir)) return;  // bad type/direction pairing → drop
            calibrateJoint((uint8_t)i, dir);
            return;
        }
    }

    // Map a wire direction token to a physical sweep, enforcing the joint-type pairing.
    // Yaw joints (name ends in 'Y') take left/right; linear joints take extend/retract.
    // left==retract==−PWM, right==extend==+PWM (a convention; sensorReversed normalizes
    // the reported position regardless). Returns false to drop a mismatched/unknown token.
    static bool parseCalDirection(const String& name, const String& dirStr, CalDirection& out)
    {
        bool yaw = (name.length() >= 4 && name.charAt(3) == 'Y');
        if (dirStr.length() == 0)        { out = CAL_DIR_NONE;    return true; }
        if (!yaw && dirStr == "retract") { out = CAL_DIR_RETRACT; return true; }
        if (!yaw && dirStr == "extend")  { out = CAL_DIR_EXTEND;  return true; }
        if ( yaw && dirStr == "left")    { out = CAL_DIR_RETRACT; return true; }
        if ( yaw && dirStr == "right")   { out = CAL_DIR_EXTEND;  return true; }
        return false;
    }

    // Begin calibrating actuator `idx`. Stops every joint first; only `idx` moves.
    void calibrateJoint(uint8_t idx, CalDirection dir = CAL_DIR_NONE)
    {
        if (idx >= count) return;
        for (size_t i = 0; i < count; i++) actuators[i]->manualDrive(0);
        jcIndex      = idx;
        jcDir        = dir;
        jcActive     = true;
        jcState      = JC_NUDGE_FWD_DRIVE;
        jcTimer      = millis();
        // Baselines for BOTH sensors so the nudge can cross-check the wrong-sensor case
        // (e.g. a Hall actuator in the knee slot). Pot is read directly here because Hall
        // joints skip the pot read in normal operation, leaving avgPot stale.
        jcHallBefore = actuators[idx]->hallSignedCount();
        jcPotBefore  = analogRead(actuators[idx]->pinPot);
        jcSensorType = (SensorType)actuators[idx]->sensorType;  // fixed per-joint, not guessed
        jcReversed   = 0;
        jcMismatch   = false;
        jcLastFailed = false;   // cleared now; set by jcAbortCal / a failing JC_SAVE
    }

    // Advance the cal state machine one tick (called from updateAll while jcActive).
    // Non-blocking: timed nudges + isStalled()-gated sweeps.
    void updateJointCal()
    {
        LinearActuator* a = actuators[jcIndex];
        // Keep the calibrating joint's smoothed sensors live. The normal update() loop is
        // bypassed during cal, so without this avgPot stays frozen the whole sweep and a pot
        // joint records min==max (span 0). Hall joints are unaffected (updateSensors skips
        // the pot read on a Hall joint, and the Hall count comes from the ISR regardless).
        a->updateSensors();
        switch (jcState)
        {
        case JC_NUDGE_FWD_DRIVE:
            a->manualDrive(JC_NUDGE_PWM);
            if (millis() - jcTimer >= JC_NUDGE_MS) { a->manualDrive(0); jcState = JC_NUDGE_FWD_EVAL; jcTimer = millis(); }
            break;
        case JC_NUDGE_FWD_EVAL:
            if (millis() - jcTimer < JC_SETTLE_MS) break;
            if (jcEvalNudge(a, /*forward=*/true)) break;     // expected sensor moved → sweeping
            if (jcMismatch) { jcAbortCal(a, "sensor_type_mismatch"); break; }  // wrong sensor on this slot
            jcState = JC_NUDGE_REV_DRIVE; jcTimer = millis();
            break;
        case JC_NUDGE_REV_DRIVE:
            a->manualDrive(-JC_NUDGE_PWM);
            if (millis() - jcTimer >= JC_NUDGE_MS) { a->manualDrive(0); jcState = JC_NUDGE_REV_EVAL; jcTimer = millis(); }
            break;
        case JC_NUDGE_REV_EVAL:
            if (millis() - jcTimer < JC_SETTLE_MS) break;
            if (jcEvalNudge(a, /*forward=*/false)) break;    // expected sensor moved → sweeping
            if (jcMismatch) { jcAbortCal(a, "sensor_type_mismatch"); break; }
            jcAbortCal(a, "motor_did_not_move");             // neither sensor moved → dead joint
            break;
        case JC_RETRACT:
            a->manualDrive(-JC_SWEEP_PWM);
            // Grace period: don't accept a stall until the actuator has had time to
            // accelerate off the start point, or a slow ramp reads as an early stop.
            if (millis() - jcTimer > JC_SWEEP_GRACE_MS && a->isStalled(JC_STALL_MS)) {
                a->manualDrive(0);
                jcPotMin  = (uint16_t)a->avgPot;
                jcHallMin = a->hallSignedCount();
                if (jcDir == CAL_DIR_RETRACT) { jcState = JC_SAVE; break; }  // single end → done
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
                if (jcDir != CAL_DIR_NONE) { jcState = JC_SAVE; break; }  // single end → done
                // Hall joints get a second retract to confirm the min count is reproducible
                // (2c drift check). A pot's reading is absolute, so one sweep is enough.
                if (jcSensorType == SENSOR_HALL) {
                    a->resetStall();  // fresh stall state for the repeat retract
                    jcState = JC_RETRACT_AGAIN; jcTimer = millis();
                } else {
                    jcState = JC_SAVE;
                }
            }
            break;
        case JC_RETRACT_AGAIN:
            a->manualDrive(-JC_SWEEP_PWM);
            if (millis() - jcTimer > JC_SWEEP_GRACE_MS && a->isStalled(JC_STALL_MS)) {
                a->manualDrive(0);
                jcHallMin2 = a->hallSignedCount();  // compared against jcHallMin in JC_SAVE (2c)
                jcState = JC_SAVE;
            }
            break;
        case JC_SAVE: {
            JointCalBlock blk;
            jointCalLoad(blk);  // invalid → zero-inited; preserves the orthogonal end on a directional cal
            JointCal& jc = blk.joints[jcIndex];

            // Merge in only the end(s) this run swept; a directional cal keeps the other
            // end from a prior cal. Track which ends have ever been recorded so a single
            // directional stroke on a fresh joint can't masquerade as fully calibrated.
            if (jcDir != CAL_DIR_EXTEND)  { jc.potMin = jcPotMin; jc.hallMin = jcHallMin; jc.endsRecorded |= JOINTCAL_END_MIN; }
            if (jcDir != CAL_DIR_RETRACT) { jc.potMax = jcPotMax; jc.hallMax = jcHallMax; jc.endsRecorded |= JOINTCAL_END_MAX; }
            jc.sensorType = jcSensorType; jc.sensorReversed = jcReversed;

            // Sweep-range sanity check, computed from the now-merged pair: a real sweep
            // crosses the whole travel, so the span should be large. A tiny span means the
            // sensor never tracked the motion (floating/intermittent) — the nudge can pass
            // on noise, so this gate stops us silently saving garbage.
            bool bothEnds = (jc.endsRecorded & (JOINTCAL_END_MIN | JOINTCAL_END_MAX))
                                == (JOINTCAL_END_MIN | JOINTCAL_END_MAX);
            int32_t span = (jcSensorType == SENSOR_HALL)
                               ? labs((int32_t)jc.hallMax - (int32_t)jc.hallMin)
                               : labs((int32_t)jc.potMax - (int32_t)jc.potMin);
            int32_t minSpan = (jcSensorType == SENSOR_HALL) ? JC_HALL_MIN_SPAN : JC_POT_MIN_SPAN;
            // Failure modes in order: (1) only one end recorded so far (directional cal
            // mid-sequence — not an error, just not-yet-trusted), (2) the sweep never
            // tracked the motion (tiny span), (3) Hall-only full sweep, the repeat retract
            // drifted from the first. Only (2)/(3) emit an ERR; (1) is silent.
            const char* failCode = nullptr;
            if (bothEnds && span < minSpan)
                failCode = (jcSensorType == SENSOR_HALL) ? "hall_no_edges" : "pot_value_invalid";
            else if (bothEnds && jcDir == CAL_DIR_NONE && jcSensorType == SENSOR_HALL
                     && labs(jcHallMin2 - jcHallMin) > JC_HALL_DRIFT_TOL)
                failCode = "hall_drift";  // repeat retract didn't reproduce hallMin (2c)
            bool ok = bothEnds && (failCode == nullptr);

            jc.calibrated = ok ? 1 : 0;   // one end only, or failed range/drift → not trusted
            jointCalSave(blk);            // still record the values (cal=0) so GET shows them
            a->applyJointCal(jc, /*liveFrame=*/true);  // in-frame → FULL; cal=0 reverts to legacy
            if (failCode)
                emitJointErr(jcIndex, failCode);
            // A directional cal that recorded its one end with no error is a SUCCESS for the
            // sequencer (calibrated may still be 0 until the partner end runs); only a real
            // failCode marks the step failed.
            jcLastFailed = (failCode != nullptr);
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
        actuators[jcIndex]->resetStall();  // fresh stall state for the first sweep
        // Extend-only directional cal starts straight at the extend stroke; everything
        // else (full sweep, retract-only) begins by retracting.
        jcState = (jcDir == CAL_DIR_EXTEND) ? JC_EXTEND : JC_RETRACT;
        jcTimer = millis();
    }

    // Did the nudge move this joint's EXPECTED sensor enough to confirm it's live and
    // learn the direction-flip? `forward` = the nudge was the +extend drive (the sign
    // convention flips for the retract nudge, Task 2 §3 step 7). The sensor TYPE is a
    // fixed per-joint property (jcSensorType is seeded from the actuator), so the expected
    // sensor is the one we sweep on. Returns true (and begins the sweep) only when that
    // sensor moved.
    //
    // It ALSO cross-checks for a wrong-sensor slot (e.g. a Hall actuator plugged into the
    // pot knee — the bench mistake that created "two Halls on a leg"). The HallA signed
    // count is the unambiguous discriminator: it only accumulates when a Hall encoder is
    // physically present (a pot actuator leaves HallA unwired → ~0). The pot pin can't
    // play that role — it's shared with HallB, so a real Hall makes it swing too. So:
    //   - Hall count moved on a POT-declared joint  → a Hall is wired where a pot belongs.
    //   - Hall count stayed put on a HALL-declared joint but the pot pin tracked cleanly
    //     (no Hall present to confound it) → a pot is wired where a Hall belongs.
    // Either sets jcMismatch; the caller aborts with sensor_type_mismatch.
    bool jcEvalNudge(LinearActuator* a, bool forward)
    {
        int32_t hallDelta = a->hallSignedCount() - jcHallBefore;       // read count first,
        int32_t potDelta  = (int32_t)analogRead(a->pinPot) - jcPotBefore; // then the ADC
        bool hallMoved = labs(hallDelta) > JC_HALL_NUDGE_THRESHOLD;
        bool potMoved  = labs(potDelta)  > JC_NUDGE_THRESHOLD;

        if (jcSensorType == SENSOR_HALL) {
            if (hallMoved) {                          // expected sensor live → sweep
                jcReversed = (forward ? (hallDelta < 0) : (hallDelta > 0)) ? 1 : 0;
                jcApplyNudgeResult(a); jcBeginSweep(); return true;
            }
            // No Hall pulses AND zero rotation (hallDelta==0) but the pin tracks → a pot is
            // wired here. The hallDelta==0 guard matters: a real Hall that barely moved
            // (1-3 counts, e.g. starting near a stop) still flips HallB on the shared pin,
            // which must NOT be read as a pot — only true zero rotation clears it.
            if (potMoved && hallDelta == 0) jcMismatch = true;
        } else {  // SENSOR_POT
            // A pot joint's HallA pin floats and picks up motor-drive EMI, so use the higher
            // present-threshold here (not the 4-count motion bar) to avoid false mismatches.
            if (labs(hallDelta) > JC_HALL_PRESENT_THRESHOLD) { jcMismatch = true; return false; }  // a real Hall is present
            if (potMoved) {                           // genuine pot motion (no Hall to confound)
                jcReversed = (forward ? (potDelta < 0) : (potDelta > 0)) ? 1 : 0;
                jcApplyNudgeResult(a); jcBeginSweep(); return true;
            }
        }
        return false;
    }

    // Stop the joint and end the cal run with one ERR. Used for every nudge-stage failure
    // (motor_did_not_move, sensor_type_mismatch) so they share the same teardown.
    void jcAbortCal(LinearActuator* a, const char* code)
    {
        emitJointErr(jcIndex, code);
        jcLastFailed = true;
        a->manualDrive(0); jcActive = false; jcState = JC_DONE;
    }

    // Push the (fixed) sensor type + learned flip onto the actuator *now*, before the
    // sweep, so getRawPos()/isStalled() read the correct sensor while sweeping (e.g. a
    // Hall joint must stall-detect on the signed count, not on avgPot = HallB noise).
    void jcApplyNudgeResult(LinearActuator* a)
    {
        a->sensorType     = jcSensorType;
        a->sensorReversed = jcReversed;
    }

    // ========================================================================
    // Whole-robot calibration sequence (M17 Task 3) — a non-blocking executor over a
    // small generated step list. Each step is one Task-2 directional calibrateJoint
    // (CA_CAL) or a closed-loop move-to-normalized-position (CA_MOVE — the moveJointTo
    // pose primitive). It chains the standard per-leg cal sequence (§3) for THIS board's
    // legs; on any per-joint cal failure it halts and holds all motors (3f).
    //
    // Scope note: this runs the LOCAL board's legs. Whole-robot orchestration across
    // boards — auto-squat splay (3c), middles-first cross-board ordering (3d), and
    // sequencing one board at a time so multiple boards don't stall the shared 24V rail
    // at once — layers on top of this and is chassis-gated.
    // ========================================================================
    // CA_MOVE = moveJointTo (one joint → posPct). CA_NEUTRAL = all calibrated joints →
    // posPct at once (the Task-4 §4b neutral pose). CA_RECORD_UNLOADED / CA_EVAL = the
    // Task-4 current-sense lift markers (jointIdx carries the leg index): record this leg's
    // unloaded avgIS, then load + evaluate vs the unloaded baseline.
    enum CalStepOp : uint8_t { CA_CAL, CA_MOVE, CA_NEUTRAL, CA_RECORD_UNLOADED, CA_EVAL };
    struct CalStep {
        uint8_t op;        // CalStepOp
        uint8_t jointIdx;  // 0..count-1 (CA_CAL/CA_MOVE) or leg index (CA_RECORD/CA_EVAL)
        uint8_t dir;       // CalDirection (CA_CAL only)
        uint8_t posPct;    // normalized target * 100 (CA_MOVE/CA_NEUTRAL), 0..100
    };
    static constexpr uint8_t       CA_MAX_STEPS      = 64;    // 2 legs of cal + neutral + lifts
    static constexpr unsigned long CA_MOVE_SETTLE_MS = 250;   // hold within tol this long = settled
    static constexpr unsigned long CA_MOVE_TIMEOUT_MS= 6000;  // give up a move after this, then advance
    static constexpr float         CA_MOVE_TOL       = 0.03f;

    void caAdd(CalStepOp op, uint8_t jointIdx, CalDirection dir, uint8_t posPct)
    {
        if (caStepCount >= CA_MAX_STEPS) return;
        caSteps[caStepCount].op = (uint8_t)op;
        caSteps[caStepCount].jointIdx = jointIdx;
        caSteps[caStepCount].dir = (uint8_t)dir;
        caSteps[caStepCount].posPct = posPct;
        caStepCount++;
    }

    // Standard per-leg cal sequence (§3) for each of this board's legs (3 joints/leg:
    // slot+0 = hip-yaw, +1 = hip-lift, +2 = knee). includeYaw=false skips the yaw steps
    // (bench-without-yaw fallback, spec §8). validate appends the Task-4 neutral pose +
    // current-sense lifts.
    void buildLocalSequence(bool includeYaw, bool validate)
    {
        caStepCount = 0;
        uint8_t nLegs = (uint8_t)(count / 3);
        for (uint8_t leg = 0; leg < nLegs; leg++)
        {
            uint8_t hy = leg * 3 + 0, hl = leg * 3 + 1, kl = leg * 3 + 2;
            caAdd(CA_CAL,  hl, CAL_DIR_RETRACT, 0);   // 1. hip min
            caAdd(CA_CAL,  kl, CAL_DIR_RETRACT, 0);   // 2. knee min
            caAdd(CA_CAL,  kl, CAL_DIR_EXTEND,  0);   // 3. knee max
            if (includeYaw) {
                caAdd(CA_CAL,  hy, CAL_DIR_RETRACT, 0);  // 4. yaw left
                caAdd(CA_CAL,  hy, CAL_DIR_EXTEND,  0);  // 5. yaw right
                caAdd(CA_MOVE, hy, CAL_DIR_NONE,   50);  // 6. yaw center
            }
            caAdd(CA_CAL,  hl, CAL_DIR_EXTEND,  0);   // 7. hip max
            caAdd(CA_MOVE, hl, CAL_DIR_NONE,   50);   // return leg to neutral mid-travel
            caAdd(CA_MOVE, kl, CAL_DIR_NONE,   50);
        }
        if (validate) appendNeutralAndValidation();
    }

    // Task 4 §3: neutral pose (every calibrated joint → 0.5) then the per-leg body-lift
    // current-sense check. Shared by calibrate-all (validate) and validate-current-sense.
    void appendNeutralAndValidation()
    {
        uint8_t nLegs = (uint8_t)(count / 3);
        caAdd(CA_NEUTRAL, 0, CAL_DIR_NONE, 50);       // 4b: all FULL joints → 0.5, settle
        for (uint8_t leg = 0; leg < nLegs; leg++)
        {
            uint8_t hl = leg * 3 + 1, kl = leg * 3 + 2;
            caAdd(CA_MOVE,            hl,  CAL_DIR_NONE, 90);  // lift hip (unload the leg)
            caAdd(CA_RECORD_UNLOADED, leg, CAL_DIR_NONE, 0);  // record unloaded avgIS
            caAdd(CA_MOVE,            kl,  CAL_DIR_NONE, 100); // extend knee (load body half)
            caAdd(CA_EVAL,            leg, CAL_DIR_NONE, 0);   // record loaded + evaluate
            caAdd(CA_MOVE,            hl,  CAL_DIR_NONE, 50);  // lower back to neutral
            caAdd(CA_MOVE,            kl,  CAL_DIR_NONE, 50);
        }
    }

    // Per-joint current-sense evaluation (Task 4 §3 / 4d). Thresholds are inline literals
    // to tune on the real robot, not named constants.
    void evalCurrentSense(uint8_t idx, float unloaded, float loaded)
    {
        int32_t delta = (int32_t)loaded - (int32_t)unloaded;
        if (delta < 20)          emitJointErr(idx, "current_sense_no_signal");
        else if (loaded < 100)   emitJointErr(idx, "current_sense_no_spike");
        else if (loaded > 800)   emitJointErr(idx, "current_sense_no_spike");
    }

    // Entry point (wire `CALL [noyaw] [skipval]`): run the local board's cal sequence,
    // then (unless skipval) the neutral pose + current-sense validation. includeYaw per §8.
    void calibrateAll(bool includeYaw = true, bool validate = true)
    {
        if (jcActive) return;            // don't stomp a single-joint cal in progress
        holdAll();
        buildLocalSequence(includeYaw, validate);
        caStepIdx = 0;
        caStepStarted = false;
        caFailed = false;
        caActive = (caStepCount > 0);
    }

    // Entry point (wire `VCS`): run the neutral pose + current-sense validation standalone,
    // assuming per-joint cal already ran (Task 4 §6 `validate-current-sense`).
    void validateCurrentSense()
    {
        if (jcActive) return;
        holdAll();
        caStepCount = 0;
        appendNeutralAndValidation();
        caStepIdx = 0;
        caStepStarted = false;
        caFailed = false;
        caActive = (caStepCount > 0);
    }

    bool calibrateAllActive() const { return caActive; }

    void caAdvance() { caStepIdx++; caStepStarted = false; }

    // Advance the sequence executor one tick. Called from updateAll while caActive and no
    // single-joint cal is mid-run (a CA_CAL step starts a calibrateJoint, which sets
    // jcActive and takes over the tick until it finishes).
    void updateCalibrateAll()
    {
        if (caStepIdx >= caStepCount) { caActive = false; holdAll(); return; }  // done (3g)
        CalStep& s = caSteps[caStepIdx];

        if (s.op == CA_CAL)
        {
            if (!caStepStarted) {
                calibrateJoint(s.jointIdx, (CalDirection)s.dir);  // sets jcActive
                caStepStarted = true;
                return;  // subsequent ticks run updateJointCal until the cal clears
            }
            // Back here only once the cal finished (jcActive cleared).
            if (jcLastFailed) { caActive = false; caFailed = true; holdAll(); return; }  // 3f
            caAdvance();
            return;
        }

        if (s.op == CA_RECORD_UNLOADED) {   // Task 4: snapshot this leg's unloaded current
            uint8_t leg = s.jointIdx;
            cvUnloadedHL = actuators[leg * 3 + 1]->avgIS;
            cvUnloadedKL = actuators[leg * 3 + 2]->avgIS;
            caAdvance();
            return;
        }
        if (s.op == CA_EVAL) {              // Task 4: loaded current vs the unloaded baseline
            uint8_t leg = s.jointIdx;
            evalCurrentSense(leg * 3 + 1, cvUnloadedHL, actuators[leg * 3 + 1]->avgIS);
            evalCurrentSense(leg * 3 + 2, cvUnloadedKL, actuators[leg * 3 + 2]->avgIS);
            caAdvance();
            return;
        }

        // CA_MOVE (one joint) / CA_NEUTRAL (every FULL joint) — drive the PID to the
        // normalized target, advance on settle (within tol for CA_MOVE_SETTLE_MS) or timeout.
        bool neutral = (s.op == CA_NEUTRAL);
        if (!caStepStarted) {
            if (neutral) {
                for (size_t i = 0; i < count; i++)
                    if (actuators[i]->calibrationState == CAL_STATE_FULL)
                        actuators[i]->setTarget(s.posPct / 100.0f);
            } else {
                actuators[s.jointIdx]->setTarget(s.posPct / 100.0f);
            }
            caStepStarted = true;
            caTimer = millis();
            caSettleStart = 0;
        }
        for (size_t i = 0; i < count; i++) actuators[i]->update();  // only targeted joints drive
        bool atTgt;
        if (neutral) {
            atTgt = true;
            for (size_t i = 0; i < count; i++)
                if (actuators[i]->calibrationState == CAL_STATE_FULL && !actuators[i]->atTarget(CA_MOVE_TOL))
                    atTgt = false;
        } else {
            atTgt = actuators[s.jointIdx]->atTarget(CA_MOVE_TOL);
        }
        if (atTgt) {
            if (caSettleStart == 0) caSettleStart = millis();
        } else {
            caSettleStart = 0;
        }
        bool settled = caSettleStart != 0 && (millis() - caSettleStart >= CA_MOVE_SETTLE_MS);
        if (settled || millis() - caTimer >= CA_MOVE_TIMEOUT_MS) {
            if (!neutral) actuators[s.jointIdx]->stopMotor();
            caAdvance();
        }
    }

    void updateAll()
    {
        if (jcActive)                  // single-joint calibration (M17 Task 2)
            updateJointCal();
        else if (caActive)             // whole-board calibration sequence (M17 Task 3)
            updateCalibrateAll();
        else if (calState != CAL_IDLE) // legacy multi-joint calibration
            updateCalibration();
        else
            for (size_t i = 0; i < count; i++)
            {
                actuators[i]->update();
                // 2f: report a joint that's driven but whose sensor stopped following.
                if (const char* code = actuators[i]->checkRuntimeHealth())
                    emitJointErr((uint8_t)i, code);
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
    bool          jcMismatch   = false;  // nudge saw the wrong sensor for this slot (→ sensor_type_mismatch)
    bool          jcLastFailed = false;  // result of the last calibrateJoint, read by the Task-3 sequencer
    uint16_t      jcPotMin     = 0;
    uint16_t      jcPotMax     = 1023;
    int32_t       jcHallMin    = 0;
    int32_t       jcHallMax    = 0;
    int32_t       jcHallMin2   = 0;  // hallMin from the repeat retract (2c drift check)
    CalDirection  jcDir        = CAL_DIR_NONE;  // which end(s) this run sweeps

    // Whole-board calibration sequence state (M17 Task 3)
    CalStep       caSteps[CA_MAX_STEPS];
    uint8_t       caStepCount  = 0;
    uint8_t       caStepIdx    = 0;
    bool          caStepStarted= false;
    bool          caActive     = false;
    bool          caFailed     = false;
    unsigned long caTimer      = 0;  // CA_MOVE start time (timeout)
    unsigned long caSettleStart= 0;  // when the move first reached tol (settle dwell)
    float         cvUnloadedHL = 0;  // Task 4 current-sense: this leg's unloaded avgIS (hip)
    float         cvUnloadedKL = 0;  // ... and knee, recorded before the body lift
};
#pragma once

#include "src/actuator/actuator_constants.h"
#include <Arduino.h>
#include <EEPROM.h>
#include <stddef.h>

#include "eeprom_layout.h"
#include "command.h"
#include "hall_hw.h"
#include "src/actuator/actuator_status.h"
#include "src/actuator/actuator_current_tracker.h"
#include "src/actuator/actuator_pot_tracker.h"
#include "src/actuator/joint_sensor.h"
#include "src/actuator/leg_calibration.h"
#include "src/controller/board_role.h"
#include "src/telemetry.h"

// Linear actuator controller with pot or hall encoder feedback.
//
// sensor says which one the joint has (src/actuator/joint_sensor.h). Pot joints
// read avgPot as today. Hall (and Unknown) joints use the encoder frame
// x = ACTUATOR_HALL_EXTEND_COUNT_SIGN * (count - hallHome), higher = more
// extended: the hip curl end (its curl switch) is 0 with limits
// [0, CAL_HIP_STROKE_EDGES] (its extend switch, derived from FL's measured
// stroke unless a hollow-stand sweep found it), the knee curl end is 0 with
// limits [-S, 0]. The counts restart at every boot, so a Hall joint has a
// position only after a curl proved its home this boot (ACTUATOR_FLAG_HOMED) and
// a saved SWEEP gave its stroke (ACTUATOR_FLAG_LIMITS_VALID). Unknown joints
// never have a position. On builds without quadrature halls every joint is Pot.
// Hall T targets stay ACTUATOR_ENC_SWITCH_INSET_EDGES inside minStop (the
// retracted end, an actuator limit switch) and, inside maxStop,
// ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES for a hip (its extend switch, a hall
// joint with minStop 0) or ACTUATOR_ENC_STOP_INSET_EDGES for a knee (its
// compliant leg stop, maxStop 0): hip [8, 1059], knee [-S + 8, -25]. Nothing
// here keeps a hip clear of the body, a stand or the floor at a given knee:
// with the knee curled the tibia meets the body at about hip 941-1003.
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
    const int pinPot;  // Analog pot pin (0 retracted .. 1023 extended); hall B on quadrature builds
    const int8_t hallSlot; // 0–5 = HallA index for this joint (-1 = no Hall telemetry)

    // Motion limits. Pot: raw ADC (0/1023 by default). Hall: the encoder frame
    // (0/0 until a stroke is known). T targets stay inside them.
    int minStop = 0;
    int maxStop = 1023;

    // Control states
    int currentPwm = 0;              // Current PWM being applied to motor (-255 to 255)
    int currentTarget = 0;           // Target position (raw ADC or frame); only used when hasTarget is true
    bool hasTarget = false;          // True only after a T (target) command; if false, motor stays idle
    unsigned long lastRampTime = 0;  // Last time PWM ramp was updated, in millis, used along with rampIntervalMs to control ramp timing
    float avgPot = 0.0;              // Global state variable to track smoothed potentiometer value
    float avgIS = 0.0;               // Global state variable to track smoothed current sense value

    ActuatorPotTracker potTracker;
    ActuatorCurrentTracker currentTracker;

    JointSensor sensor = JointSensor::Pot;  // init(): Unknown on quadrature builds
    uint8_t encFlags = 0;                   // ACTUATOR_FLAG_* bits
    int32_t hallHome = 0;                   // raw count at the proven curl end
    int16_t guardPos = 0;                   // T stall guard baseline (frame or pot counts)
    uint16_t guardMs = 0;                   // T stall guard window start (wrapping ms)
    // Direction (+1 raises the position, -1 lowers it) a T stall guard trip
    // stopped; 0 when none. Until a T reverses it, a jog or a stop (H), a T in
    // that direction is accepted but leaves the joint stopped. After a quiet
    // stop (below ACTUATOR_T_STALL_MIN_PWM) only a T within the error deadband
    // of the stopped target (stallLatchTarget) is held; a different target
    // re-arms the guard and drives at the PWM its own error gives.
    int8_t stallLatchSign = 0;
    bool stallLatchQuiet = false;
    int stallLatchTarget = 0;

    // Shared physical slot pins; role election only changes the joint name.
    struct SlotConfig
    {
        int pinPwmR;
        int pinPwmL;
        int pinEn;
        int pinIS;
        int pinPot;
        int8_t hallSlot;
    };

    LinearActuator(const char *n, int pR, int pL, int en, int isPin, int pot, int8_t hallIdx = -1)
        : name(n), pinPwmR(pR), pinPwmL(pL), pinEn(en), pinIS(isPin), pinPot(pot), hallSlot(hallIdx) {}

    LinearActuator(const char *n, const SlotConfig &slot, const ControlConfig &cfg)
        : name(n),
          pinPwmR(slot.pinPwmR),
          pinPwmL(slot.pinPwmL),
          pinEn(slot.pinEn),
          pinIS(slot.pinIS),
          pinPot(slot.pinPot),
          hallSlot(slot.hallSlot),
          controlConfig(cfg) {}

    const char *getName() const
    {
        return name;
    }
    void setControlConfig(const ControlConfig &cfg) { controlConfig = cfg; }

    void init()
    {
        // Every joint starts Unknown on a quadrature build (proven by a
        // calibration move or a stored record), Pot otherwise.
        sensor = hallHwHasQuadrature() ? JointSensor::Unknown : JointSensor::Pot;
        encFlags = 0;
        hallHome = 0;
        if (sensor != JointSensor::Pot)
        {
            minStop = 0;
            maxStop = 0;
        }

        // Configure pin modes. A hall B line keeps its pull-up.
        pinMode(pinPwmR, OUTPUT);
        pinMode(pinPwmL, OUTPUT);
        pinMode(pinEn, OUTPUT);
        pinMode(pinIS, INPUT);
        pinMode(pinPot, sensor == JointSensor::Pot ? INPUT : INPUT_PULLUP);

        // Safe startup: explicitly drive PWM low and EN low.
        // EN will only be driven HIGH later when we actually command motion
        // via driveActuator() (in update() / manualDrive()).
        analogWrite(pinPwmR, 0);
        analogWrite(pinPwmL, 0);
        digitalWrite(pinEn, LOW);

        // Initialize averaging
        avgPot = analogRead(pinPot);
        avgIS = analogRead(pinIS);
        potTracker.reset(avgPot);
        currentTracker.reset();
        hasTarget = false; // No target until host sends T command
    }

    // Called during update to calculate new smoothed sensor readings, called internally on a fixed interval to exponentially average pot/IS readings
    void updateSensors()
    {
        int rawPot = analogRead(pinPot);
        int rawIS = analogRead(pinIS);
        const bool isDriving = digitalRead(pinEn) == HIGH;

        // Exponential Moving Average
        avgPot = (avgPot * (1.0 - controlConfig.alphaPot)) + (rawPot * controlConfig.alphaPot);
        avgIS = (avgIS * (1.0 - controlConfig.alphaIS)) + (rawIS * controlConfig.alphaIS);

        // The pot validity check and its pull-up probe run only for a proven pot:
        // the probe would strip a hall B pull-up and misread the line as open.
        if (sensor == JointSensor::Pot)
        {
            const uint32_t nowMilliseconds = millis();
            potTracker.update(
                avgPot,
                isDriving,
                nowMilliseconds,
                [this]() { return readPotentiometerProbeRise(); });
        }

        currentTracker.update(
            isDriving,
            currentPwm,
            avgIS);
    }

    // Raw signed hall count of this joint's slot (0 without a slot).
    int32_t hallCount() const
    {
        return (hallSlot >= 0 && hallSlot < 6) ? hallHwGetEdgeCount(static_cast<uint8_t>(hallSlot)) : 0;
    }

    // Encoder frame: higher = more extended, the home at 0, clamped to fit int16.
    int hallPosition() const
    {
        const int32_t x = static_cast<int32_t>(ACTUATOR_HALL_EXTEND_COUNT_SIGN) * (hallCount() - hallHome);
        const int32_t limit = ACTUATOR_HALL_POSITION_LIMIT;
        return static_cast<int>(constrain(x, -limit, limit));
    }

    // True when getRawPos() is a position: a valid pot, or a Hall joint homed
    // this boot with a known stroke. Unknown never is.
    bool isPositionReadable() const
    {
        if (sensor == JointSensor::Pot)
            return potTracker.isValid();
        const uint8_t ready = ACTUATOR_FLAG_HOMED | ACTUATOR_FLAG_LIMITS_VALID;
        return sensor == JointSensor::Hall && (encFlags & ready) == ready;
    }

    // Returns normalized position [0.0,1.0], or NaN when there is no position.
    float getPos() const
    {
        if (!isPositionReadable())
            return (float)NAN;
        float range = maxStop - minStop;
        if (range == 0)
            return 0.5;
        return (getRawPos() - minStop) / range;
    }

    float readPotentiometerProbeRise() const
    {
        int32_t baselineSum = 0;
        for (unsigned int sample = 0; sample < POT_PROBE_SAMPLE_COUNT; ++sample)
            baselineSum += analogRead(pinPot);

        pinMode(pinPot, INPUT_PULLUP);
        delayMicroseconds(POT_PROBE_SETTLE_US);

        int32_t pullUpSum = 0;
        for (unsigned int sample = 0; sample < POT_PROBE_SAMPLE_COUNT; ++sample)
            pullUpSum += analogRead(pinPot);

        pinMode(pinPot, INPUT);
        return (pullUpSum - baselineSum) / float(POT_PROBE_SAMPLE_COUNT);
    }

    // Mean of POT_PROBE_SAMPLE_COUNT direct pot reads (calibration, at rest).
    int readPotSnapshot() const
    {
        int32_t sum = 0;
        for (unsigned int sample = 0; sample < POT_PROBE_SAMPLE_COUNT; ++sample)
            sum += analogRead(pinPot);
        return static_cast<int>(sum / static_cast<int32_t>(POT_PROBE_SAMPLE_COUNT));
    }

    // Smoothed pot (Pot) or the encoder frame (Hall, Unknown).
    int getRawPos() const { return sensor == JointSensor::Pot ? (int)avgPot : hallPosition(); }

    // Changes the sensor kind (forced to Pot without quadrature halls). On
    // quadrature builds the pot pin is INPUT for a pot and keeps its pull-up for
    // hall B. A change resets the encoder state and the limits and stops a target.
    void setSensor(JointSensor kind)
    {
        if (!hallHwHasQuadrature())
            kind = JointSensor::Pot;
        else
            pinMode(pinPot, kind == JointSensor::Pot ? INPUT : INPUT_PULLUP);
        if (kind == sensor)
            return;
        sensor = kind;
        encFlags = 0;
        hallHome = 0;
        minStop = 0;
        maxStop = kind == JointSensor::Pot ? 1023 : 0;
        if (kind == JointSensor::Pot)
            potTracker.reset(avgPot);
        if (hasTarget)
            stopMotor();
    }

    // The proven curl end is the current count.
    void setHome()
    {
        hallHome = hallCount();
        encFlags = static_cast<uint8_t>((encFlags | ACTUATOR_FLAG_HOMED) & ~ACTUATOR_FLAG_HOME_LOST);
    }

    // A home no longer trusted (a failed move or a guard) is marked lost.
    void clearHome()
    {
        if (encFlags & ACTUATOR_FLAG_HOMED)
            encFlags = static_cast<uint8_t>((encFlags & ~ACTUATOR_FLAG_HOMED) | ACTUATOR_FLAG_HOME_LOST);
    }

    // SENS state 0-5, the index of its code in sensorStateCode().
    uint8_t sensorState() const
    {
        if (sensor == JointSensor::Pot)
            return 1;
        if (sensor != JointSensor::Hall)
            return 0;
        if (encFlags & ACTUATOR_FLAG_HOMED)
            return (encFlags & ACTUATOR_FLAG_LIMITS_VALID) ? 5 : 4;
        return (encFlags & ACTUATOR_FLAG_HOME_LOST) ? 3 : 2;
    }

    // SENS state code: '?' Unknown, 'P' Pot; Hall: 'e' unhomed, 'x' home lost,
    // 's' homed without a stroke, 'E' homed with a stroke (ready for T).
    char sensorStateCode() const
    {
        return KRABBY_FLASH_CHAR(KRABBY_FLASH_LITERAL("?PexsE") + sensorState());
    }

    // A hall range of the hip form [0, N] (a knee's is [-S, 0]).
    bool isHallHipRange() const { return minStop == 0 && maxStop > 0; }

    // Set position target (T command only). Only this sets hasTarget = true.
    // Refused (false) for NaN or without a position; a refusal stops an active target.
    // The stall guard window restarts only for a new target (none before) or one
    // that reverses the drive: a T re-sent while the joint is blocked never
    // postpones the guard. After an ERR STALL trip, a T in the stopped direction
    // is accepted but keeps the joint stopped (a held key 0 or a streaming host
    // re-sends T): only a T the other way, a jog or a stop (H) re-arms it. After
    // a quiet stop only the stopped target (within the error deadband, same
    // side) stays stopped: a different target re-arms the guard.
    bool setTarget(float val)
    {
        if (val != val || !isPositionReadable())  // val != val: NaN
        {
            if (hasTarget)
                stopMotor();
            return false;
        }
        val = constrain(val, 0.0, 1.0);
        long target = minStop + (long)(val * (maxStop - minStop));
        if (sensor == JointSensor::Hall)
            target = constrain(target,
                               (long)minStop + ACTUATOR_ENC_SWITCH_INSET_EDGES,
                               (long)maxStop - (isHallHipRange() ? ACTUATOR_ENC_HIP_EXTEND_INSET_EDGES
                                                                 : ACTUATOR_ENC_STOP_INSET_EDGES));
        const long pos = getRawPos();
        if (stallLatchSign != 0)
        {
            const int8_t dir = target > pos ? 1 : (target < pos ? -1 : 0);
            const long deadband = sensor == JointSensor::Hall ? ACTUATOR_ENC_ERR_DEADBAND_EDGES
                                                              : controlConfig.pwmErrDeadband;
            if (dir == stallLatchSign &&
                (!stallLatchQuiet || labs(target - (long)stallLatchTarget) < deadband))
                return true;
            stallLatchSign = 0;
        }
        const bool reverses = (target > pos) != ((long)currentTarget > pos);
        if (!hasTarget || reverses)
        {
            guardPos = static_cast<int16_t>(pos);
            guardMs = static_cast<uint16_t>(millis());
        }
        currentTarget = (int)target;
        hasTarget = true;
        return true;
    }

    // Hold: just stop the motor. No target is set or updated; a stall guard
    // latch is released.
    void stopMotor()
    {
        currentPwm = 0;
        driveActuator(0, controlConfig.pwmDeadband);
        hasTarget = false;
        stallLatchSign = 0;
    }

    // Jog: direct PWM. Clears any target, so a stale T never overrides a jog;
    // when pwm is 0 we just stop.
    void manualDrive(int pwm)
    {
        hasTarget = false;
        stallLatchSign = 0;
        pwm = constrain(pwm, -ACTUATOR_PWM_MAXIMUM_MAGNITUDE, ACTUATOR_PWM_MAXIMUM_MAGNITUDE);
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

    // Drives actuator to desired position using controlConfig; call frequently in main loop.
    // Never runs during calibration (ActuatorManager::updateAll), so no guard here
    // can stop a calibration move.
    void update()
    {
        updateSensors(); // Always update sensors to recalculate avgPot/avgIS

        // A pot never counts hall edges: a Pot joint that does was swapped for an
        // encoder and loses its kind, limits and target.
        if (sensor == JointSensor::Pot && hallHwHasQuadrature())
        {
            const int32_t count = hallCount();
            if (count >= CAL_HALL_DETECT_EDGES || count <= -CAL_HALL_DETECT_EDGES)
                setSensor(JointSensor::Unknown);
        }

        // Range guard: a homed encoder clearly outside its stroke has a wrong
        // home. A hall hip's range is its whole stroke, curl switch to extend
        // switch [0, CAL_HIP_STROKE_EDGES], a knee's [-S, 0].
        if (sensor == JointSensor::Hall && isPositionReadable())
        {
            const int x = hallPosition();
            if (x < minStop - ACTUATOR_ENC_RANGE_MARGIN_EDGES || x > maxStop + ACTUATOR_ENC_RANGE_MARGIN_EDGES)
            {
                clearHome();
                encFlags |= ACTUATOR_FLAG_ERR_HOME;
                if (hasTarget)
                    stopMotor();
            }
        }

        // No target: motor chills electrically. manualDrive() directly sets PWM/EN
        // when jogging, and in the no-target state we do not override that here.
        if (!hasTarget)
            return;
        if (!isPositionReadable())
        {
            stopMotor();
            return;
        }

        const bool enc = sensor == JointSensor::Hall;
        // long: frame values reach +-30000, beyond a 16-bit int difference.
        long error = (long)currentTarget - getRawPos();
        if (labs(error) < (enc ? ACTUATOR_ENC_ERR_DEADBAND_EDGES : controlConfig.pwmErrDeadband))
            error = 0;

        const int cap = enc ? ACTUATOR_ENC_MAX_PWM : ACTUATOR_PWM_MAXIMUM_MAGNITUDE;
        int desiredPwm = (int)constrain(error * (enc ? ACTUATOR_ENC_KP : controlConfig.Kp), (float)-cap, (float)cap);

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

        // T stall guard: driven at ACTUATOR_T_STALL_MIN_PWM or more without
        // progress for the window stops the joint. A stalled encoder may have
        // lost edges, so its home is no longer trusted. Driven above the drive
        // deadband but below that PWM, ACTUATOR_T_STALL_SLOW_WINDOW_MS without
        // progress stops the joint quietly (a target it cannot reach at low PWM,
        // or blocked near it): nothing stays energised without moving.
        const uint16_t now16 = static_cast<uint16_t>(millis());
        const int pos = getRawPos();
        const int pwmMagnitude = abs(currentPwm);
        const int8_t driveSign = (long)currentTarget > pos ? 1 : -1;
        const uint16_t stalledMs = static_cast<uint16_t>(now16 - guardMs);
        if (pwmMagnitude < controlConfig.pwmDeadband ||
            labs((long)pos - guardPos) >= ACTUATOR_T_STALL_MIN_PROGRESS)
        {
            guardPos = static_cast<int16_t>(pos);
            guardMs = now16;
        }
        else if (pwmMagnitude >= ACTUATOR_T_STALL_MIN_PWM)
        {
            if (stalledMs >= ACTUATOR_T_STALL_WINDOW_MS)
            {
                stopMotor();
                stallLatchSign = driveSign;
                stallLatchQuiet = false;
                encFlags |= ACTUATOR_FLAG_ERR_STALL;
                if (enc)
                    clearHome();
            }
        }
        else if (stalledMs >= ACTUATOR_T_STALL_SLOW_WINDOW_MS)
        {
            stopMotor();
            stallLatchSign = driveSign;
            stallLatchQuiet = true;
            stallLatchTarget = currentTarget;
        }
    }

    // Joint segment: "<name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <hallEdges> <conn>".
    // The manager owns separators between segments; the loop owns role prefix,
    // optional sensor segments, and the one line ending.
    // Keep in sync with firmware/interfaces/joint_telemetry.py
    // Keeping it super simple to avoid any string parsing and external library overhead
    void printTelemetry(Print& out) const
    {
        out.print(name);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(getPos(), 3);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print((int)avgPot);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print((int)avgIS);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        int en = digitalRead(pinEn);
        out.print(en);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(en);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(currentPwm < 0 ? abs(currentPwm) : 0);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(currentPwm > 0 ? currentPwm : 0);
        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(hallCount());

        out.print(TELEMETRY_FIELD_SEPARATOR);
        out.print(static_cast<int>(getConnectionState()));
    }

    ActuatorConnection getConnectionState() const
    {
        return determineActuatorConnectionState(isPositionReadable(), currentTracker.evidence());
    }

    ActuatorStatus getStatus() const
    {
        ActuatorStatus status;
        status.actuatorId = parseActuatorId(name);
        status.connectionState = getConnectionState();
        status.commandedPwm = static_cast<int16_t>(currentPwm);
        return status;
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

// CalibrationTextSink over an Arduino Print: USB, a follower link or an uplink.
class PrintCalibrationSink final : public CalibrationTextSink
{
public:
    explicit PrintCalibrationSink(Print *out = nullptr) : out_(out) {}

    void setOutput(Print *out) { out_ = out; }
    Print *output() const { return out_; }

    void printFlash(const char *flashText) override
    {
#ifdef __AVR__
        out_->print(reinterpret_cast<const __FlashStringHelper *>(flashText));
#else
        out_->print(flashText);
#endif
    }
    void printText(const char *text) override { out_->print(text); }
    void printNumber(long value) override { out_->print(value); }
    void printChar(char value) override { out_->print(value); }
    void endLine() override { out_->println(); }

private:
    Print *out_;
};

// The board's actuators, its calibration pass, and its operator lines. Lines go
// through the calibration sink (setCalibrationOutput): "CAL ..." from the
// sequencer, "SENS <ROLE> <NAME>:<c> ..." (quadrature builds: at boot, on a
// change and every ACTUATOR_SENS_PERIOD_MS; codes in sensorStateCode()),
// "ERR STALL <name>", "ERR HOME <name> lost (x=<frame>)", "ERR T refused
// <names> (no position)" and the EEPROM load and save messages.
class ActuatorManager
{
public:
    ActuatorManager(LinearActuator **actsArray, size_t actsCount)
        : actuators(actsArray), count(actsCount) {}

    void initAll()
    {
        for (size_t i = 0; i < count; i++)
            actuators[i]->init();
        // Line state as at construction, so a re-init (a sim reset) prints
        // the first SENS line and the first T refusal like a boot does.
        sensSig_ = 0xFFFFFFFFUL;
        lastSensMs_ = 0;
        lastRefusedMs_ = static_cast<uint16_t>(0U - ACTUATOR_T_REFUSED_LOG_INTERVAL_MS);
    }

    void handleJog(String name, int pwm)
    {
        // TODO: Improve brute force O(N) lookup
        for (size_t i = 0; i < count; i++)
        {
            if (String(actuators[i]->getName()) == name)
            {
                actuators[i]->manualDrive(pwm);
                return;
            }
        }
    }

    void updateAll()
    {
        if (calibration.isRunning())
        {
            // Calibration owns the drive while it runs, but sensors refresh every
            // loop on every build. The T guards in update() never run here.
            for (size_t i = 0; i < count; i++)
                actuators[i]->updateSensors();
            CalibrationPlant plant(*this);
            calibration.update(plant);
        }
        else
        {
            for (size_t i = 0; i < count; i++)
                actuators[i]->update();
        }
        printPendingErrors();
        pollSensorStateLine();
    }

    void applyCommands(const Command *cmds, size_t cmdCount)
    {
        // Refused targets (no position), by actuator index.
        uint32_t refused = 0;
        // TODO: This is O(N^2), but N is small so probably ok for now. Would need to add a map for larger actuator sets.
        for (size_t i = 0; i < cmdCount; i++)
        {
            const auto &cmd = cmds[i];
            for (size_t j = 0; j < count; j++)
            {
                if (cmd.name == actuators[j]->getName())
                {
                    if (!actuators[j]->setTarget(cmd.val) && j < 32)
                        refused |= 1UL << j;
                    break;
                }
            }
        }
        if (refused != 0)
            printRefusedTargets(refused);
    }

    void holdAll()
    {
        // Hold ends a running calibration first; otherwise its sequencer would
        // drive again on the next loop.
        abortCalibration('H');
        // For now, "hold" means fully de‑energize all joints:
        // EN low and PWM 0 on every actuator. This avoids any
        // PID activity that could move other joints when the
        // user expects everything to stay still.
        stopAllActuators();
    }

    // Prints the joint segments only — no line ending. The caller terminates the
    // line so the leader can append sensor segments before the newline.
    void printTelemetry(Print& out) const
    {
        for (size_t i = 0; i < count; i++)
        {
            if (i) out.print(TELEMETRY_SEGMENT_DELIMITER);
            actuators[i]->printTelemetry(out);
        }
    }

    // ==================================================
    // AUTO-CALIBRATION & PERSISTENCE
    // ==================================================
    // Struct to save to EEPROM. Fixed-width fields: the same 30-byte record on
    // the Mega (16-bit int) and on 32-bit host builds, little-endian on both.
    struct CalData
    {
        // TODO: Should be stored with Joint information, so that when joints change this changes, not hardcoded here
        int16_t minVals[6];   // 0   Pot: ADC; Hall: frame edges (hip 0, knee -S)
        int16_t maxVals[6];   // 12  Pot: ADC; Hall: hip its stroke (CAL_HIP_STROKE_EDGES: derived), knee 0
        uint16_t magic;       // 24  CAL_DATA_MAGIC (v1: CAL_DATA_MAGIC_V1, same offset)
        uint16_t sensorBits;  // 26  joint j at bits 2j..2j+1 (JointSensor; 3 reads as Unknown); bits 12-15 zero
        uint16_t check;       // 28  calDataCheck()
    };

    // v2 record (kinds and a check). v1 was the 26-byte pot-only record with the
    // magic at the same offset; the old 0xDEADBEEF magic truncated to 0xBEEF on
    // the Mega and never loaded.
    static constexpr uint16_t CAL_DATA_MAGIC = 0xCA2B;
    static constexpr uint16_t CAL_DATA_MAGIC_V1 = 0xCA1B;
    static constexpr uint16_t CAL_DATA_CHECK_XOR = 0x5A5A;
    static constexpr uint16_t CAL_DATA_RESERVED_SENSOR_BITS = 0xF000;

    static_assert(sizeof(CalData) == EEPROM_ACTUATOR_CAL_SIZE,
                  "actuator calibration EEPROM layout changed");

    // (sum of bytes 0..27 as uint16) ^ CAL_DATA_CHECK_XOR.
    static uint16_t calDataCheck(const CalData &data)
    {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&data);
        uint16_t sum = 0;
        for (size_t i = 0; i < offsetof(CalData, check); i++)
            sum = static_cast<uint16_t>(sum + bytes[i]);
        return static_cast<uint16_t>(sum ^ CAL_DATA_CHECK_XOR);
    }

    // Where this board's "CAL <ROLE> ..." lines go (mainSerial: USB on the
    // leader, the uplink on a follower) and the role they are labelled with.
    void setCalibrationOutput(Print *out, BoardRole role)
    {
        calibrationSink.setOutput(out);
        calibrationRole = role;
    }

    // The role set by setCalibrationOutput() (ROLE_UNKNOWN until then).
    BoardRole calibrationOutputRole() const { return calibrationRole; }

    // Runs the requested passes (CAL_PASS_* bits) at PWM magnitude pwm with the
    // CAL_OPT_* options on this board; see src/actuator/leg_calibration.h.
    // Returns false while one is running, for an invalid request, without six
    // actuators or without an output.
    bool startCalibration(uint8_t passes, int pwm, uint8_t options = CAL_DEFAULT_OPTIONS)
    {
        if (count != LEG_CALIBRATION_JOINT_COUNT || calibrationSink.output() == nullptr)
            return false;
        CalibrationPlant plant(*this);
        return calibration.start(plant, passes, pwm, options);
    }

    // Stops every actuator and ends a running calibration with nothing saved,
    // emitting "CAL <ROLE> ABORTED <cause>". Returns false when none was running.
    bool abortCalibration(char cause)
    {
        if (!calibration.isRunning())
            return false;
        CalibrationPlant plant(*this);
        return calibration.abort(plant, cause);
    }

    // Host-sim reset: forget calibration state without output or motor changes.
    void resetCalibration() { calibration.reset(); }

    bool isCalibrating() const { return calibration.isRunning(); }
    LegCalibrationOutcome calibrationOutcome() const { return calibration.outcome(); }
    const LegCalibration &calibrationSequencer() const { return calibration; }

    // Reason of the last failed or aborted calibration (no prefix, no line end).
    void printCalibrationFailure(CalibrationTextSink &out) const
    {
        calibration.printFailureReason(out);
    }

    void saveCalibration()
    {
        writeCalibrationToEeprom();
        printMessage(KRABBY_FLASH_LITERAL("Limits saved"));
    }

    // Applies the EEPROM record (called once at boot, after initAll): per-joint
    // kinds and limits, each joint validated; an invalid joint gets its kind's
    // defaults. A v1 record is ignored on quadrature builds (it may hold hall B
    // rail values) and read as all-Pot otherwise; so is a v2 record with a Pot
    // yaw, which only a build without quadrature writes. Without a usable record the
    // joints keep the defaults init() set. Every joint's kind is reapplied, so
    // every pot pin mode is set. Homes are never stored.
    void loadCalibration()
    {
        CalData data;
        EEPROM.get(EEPROM_ACTUATOR_CAL_ADDR, data);
        const bool quadrature = hallHwHasQuadrature();
        const bool isV2 = data.magic == CAL_DATA_MAGIC;
        const bool v2Intact = isV2 && data.check == calDataCheck(data) &&
                              (data.sensorBits & CAL_DATA_RESERVED_SENSOR_BITS) == 0;
        const bool isV1 = data.magic == CAL_DATA_MAGIC_V1;
        // A quadrature build never gives a yaw a kind, and a build without
        // quadrature saves every joint as Pot: a Pot yaw on a quadrature build
        // means the record came from a build without halls, and its Pot kinds
        // would remove hall-B pull-ups and servo on hall B. It is ignored like v1.
        bool potYaw = false;
        for (size_t i = 0; i < LEG_CALIBRATION_JOINT_COUNT; i += LEG_CALIBRATION_JOINTS_PER_LEG)
            potYaw = potYaw || storedSensor(data.sensorBits, i) == JointSensor::Pot;
        const bool foreign = quadrature && v2Intact && potYaw;
        // Records the joints read from: v2 (unless foreign), v1 only without quadrature halls.
        const bool useRecord = (v2Intact && !foreign) || (isV1 && !quadrature);

        for (size_t i = 0; i < count; i++)
        {
            LinearActuator &a = *actuators[i];
            if (!useRecord || i >= LEG_CALIBRATION_JOINT_COUNT)
            {
                a.setSensor(a.sensor);  // as init() left it; pin mode only
                continue;
            }
            JointSensor kind = isV1 ? JointSensor::Pot : storedSensor(data.sensorBits, i);
            const int lo = data.minVals[i];
            const int hi = data.maxVals[i];
            const bool valid = isValidStoredJoint(kind, static_cast<uint8_t>(i), lo, hi, quadrature);
            if (!valid)
            {
                kind = quadrature ? JointSensor::Unknown : JointSensor::Pot;
                printJointInvalid(a.getName());
            }
            applyStoredJoint(a, kind, valid && kind != JointSensor::Unknown, lo, hi);
            if (valid && kind == JointSensor::Hall && i % LEG_CALIBRATION_JOINTS_PER_LEG == LEG_CALIBRATION_HIP_OFFSET &&
                hi == CAL_HIP_STROKE_EDGES)
            {
                printHipExtendDerived(a.getName());
            }
        }

        if (foreign)
            printMessage(KRABBY_FLASH_LITERAL("Calibration from a build without halls ignored: run C"));
        else if (v2Intact)
            printMessage(KRABBY_FLASH_LITERAL("Calibration v2 loaded"));
        else if (isV2)
            printMessage(KRABBY_FLASH_LITERAL("Calibration record corrupt: defaults"));
        else if (isV1)
            printMessage(quadrature ? KRABBY_FLASH_LITERAL("Calibration v1 ignored: run C")
                                    : KRABBY_FLASH_LITERAL("Calibration v1 loaded"));
        else
            printMessage(KRABBY_FLASH_LITERAL("No calibration record: defaults"));
    }

    // LegCalibration's plant: this board's six actuators (public for tests).
    class CalibrationPlant final : public LegCalibrationPlant
    {
    public:
        explicit CalibrationPlant(ActuatorManager &owner) : manager(owner) {}

        void drive(uint8_t joint, int pwm) override { act(joint).manualDrive(pwm); }
        void stopAll() override { manager.stopAllActuators(); }
        int rawPot(uint8_t joint) override { return static_cast<int>(act(joint).avgPot); }
        int potSnapshot(uint8_t joint) override { return act(joint).readPotSnapshot(); }
        bool isPotOpen(uint8_t joint) override { return act(joint).potTracker.isPositionOpen(); }
        bool hasQuadratureHall() override { return hallHwHasQuadrature(); }
        int32_t hallCount(uint8_t joint) override { return act(joint).hallCount(); }
        int position(uint8_t joint) override { return act(joint).getRawPos(); }
        JointSensor sensor(uint8_t joint) override { return act(joint).sensor; }
        void setSensor(uint8_t joint, JointSensor kind) override { act(joint).setSensor(kind); }
        bool isHomed(uint8_t joint) override { return (act(joint).encFlags & ACTUATOR_FLAG_HOMED) != 0; }
        void setHome(uint8_t joint) override { act(joint).setHome(); }
        void clearHome(uint8_t joint) override { act(joint).clearHome(); }
        // round(10 * avgIS): the end detector's current, in tenths of an IS count.
        uint16_t motorCurrentTenths(uint8_t joint) override
        {
            return calibrationCurrentTenths(act(joint).avgIS);
        }
        // Latched: the tracker keeps its evidence after the motor stops.
        bool isCurrentAbsent(uint8_t joint) override
        {
            return act(joint).currentTracker.evidence() == ActuatorCurrentEvidence::CurrentAbsent;
        }
        bool measuredLimit(uint8_t joint, bool extended, int &value) override
        {
            const LinearActuator &a = act(joint);
            if (a.sensor != JointSensor::Pot || (a.encFlags & ACTUATOR_FLAG_LIMITS_VALID) == 0)
                return false;
            value = extended ? a.maxStop : a.minStop;
            return true;
        }
        int savedStroke(uint8_t joint) override
        {
            const LinearActuator &a = act(joint);
            if (a.sensor != JointSensor::Hall || (a.encFlags & ACTUATOR_FLAG_LIMITS_VALID) == 0)
                return 0;
            return a.maxStop - a.minStop;
        }
        uint32_t nowMs() override { return millis(); }
        const char *jointName(uint8_t joint) override { return act(joint).getName(); }
        const char *roleLabel() override { return boardRoleLabel(manager.calibrationRole); }
        CalibrationTextSink &out() override { return manager.calibrationSink; }
        void storeLimits(uint8_t joint, int minStop, int maxStop) override
        {
            LinearActuator &a = act(joint);
            a.minStop = minStop;
            a.maxStop = maxStop;
            a.encFlags |= ACTUATOR_FLAG_LIMITS_VALID;
        }
        void saveLimits() override { manager.writeCalibrationToEeprom(); }

    private:
        LinearActuator &act(uint8_t joint) { return *manager.actuators[joint]; }

        ActuatorManager &manager;
    };

private:
    static JointSensor storedSensor(uint16_t bits, size_t joint)
    {
        const uint8_t raw = static_cast<uint8_t>((bits >> (2 * joint)) & 0x3);
        if (raw == static_cast<uint8_t>(JointSensor::Pot))
            return JointSensor::Pot;
        if (raw == static_cast<uint8_t>(JointSensor::Hall))
            return JointSensor::Hall;
        return JointSensor::Unknown;  // 0, and 3 (unused)
    }

    // A stored joint is used only when its limits fit its kind: a pot inside the
    // ADC range; on a quadrature build, a hall hip [0, N] with N its stroke as a
    // table-safe SWEEP saves it (CAL_HIP_STROKE_EDGES, derived, or an extend
    // switch a hollow-stand sweep found within the tolerance of it; an older
    // record's fixed 700-edge sweep is invalid: run C) and a hall knee [-S, 0]
    // with a plausible stroke S. Unknown carries no limits.
    static bool isValidStoredJoint(JointSensor kind, uint8_t joint, int lo, int hi, bool quadrature)
    {
        if (kind == JointSensor::Pot)
            return lo >= 0 && lo < hi && hi <= 1023;
        if (kind == JointSensor::Unknown)
            return true;
        if (!quadrature)
            return false;
        const uint8_t offset = joint % LEG_CALIBRATION_JOINTS_PER_LEG;
        if (offset == LEG_CALIBRATION_HIP_OFFSET)
            return lo == 0 && hi >= CAL_HIP_STROKE_EDGES - CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES &&
                   hi <= CAL_HIP_STROKE_EDGES + CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES;
        const long stroke = (long)hi - lo;
        return offset == LEG_CALIBRATION_KNEE_OFFSET && hi == 0 &&
               stroke >= CAL_HALL_MIN_STROKE_EDGES && stroke <= CAL_HALL_MAX_STROKE_EDGES;
    }

    // Kind first (it resets the limits on a change and sets the pin mode), then
    // the stored limits, or the defaults for the kind.
    static void applyStoredJoint(LinearActuator &a, JointSensor kind, bool hasLimits, int lo, int hi)
    {
        a.setSensor(kind);
        a.encFlags = static_cast<uint8_t>(a.encFlags & ~ACTUATOR_FLAG_LIMITS_VALID);
        if (hasLimits)
        {
            a.minStop = lo;
            a.maxStop = hi;
            a.encFlags |= ACTUATOR_FLAG_LIMITS_VALID;
        }
        else
        {
            a.minStop = 0;
            a.maxStop = a.sensor == JointSensor::Pot ? 1023 : 0;
        }
    }

    // One flash literal and a line end, when there is an output.
    void printMessage(const char *flashText)
    {
        if (calibrationSink.output() == nullptr)
            return;
        calibrationSink.printFlash(flashText);
        calibrationSink.endLine();
    }

    // "Calibration joint <name> extend derived (<N> edges): not switch-verified"
    void printHipExtendDerived(const char *jointName)
    {
        if (calibrationSink.output() == nullptr)
            return;
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL("Calibration joint "));
        calibrationSink.printText(jointName);
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL(" extend derived ("));
        calibrationSink.printNumber(CAL_HIP_STROKE_EDGES);
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL(" edges): not switch-verified"));
        calibrationSink.endLine();
    }

    void printJointInvalid(const char *jointName)
    {
        if (calibrationSink.output() == nullptr)
            return;
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL("Calibration joint "));
        calibrationSink.printText(jointName);
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL(" invalid: defaults"));
        calibrationSink.endLine();
    }

    // "ERR STALL <name>" / "ERR HOME <name> lost (x=<frame>)" for each pending
    // guard trip; the pending bits are cleared.
    void printPendingErrors()
    {
        const uint8_t errBits = ACTUATOR_FLAG_ERR_STALL | ACTUATOR_FLAG_ERR_HOME;
        for (size_t i = 0; i < count; i++)
        {
            LinearActuator &a = *actuators[i];
            const uint8_t pending = a.encFlags & errBits;
            if (pending == 0)
                continue;
            a.encFlags = static_cast<uint8_t>(a.encFlags & ~errBits);
            if (calibrationSink.output() == nullptr)
                continue;
            if (pending & ACTUATOR_FLAG_ERR_STALL)
            {
                calibrationSink.printFlash(KRABBY_FLASH_LITERAL("ERR STALL "));
                calibrationSink.printText(a.getName());
                calibrationSink.endLine();
            }
            if (pending & ACTUATOR_FLAG_ERR_HOME)
            {
                calibrationSink.printFlash(KRABBY_FLASH_LITERAL("ERR HOME "));
                calibrationSink.printText(a.getName());
                calibrationSink.printFlash(KRABBY_FLASH_LITERAL(" lost (x="));
                calibrationSink.printNumber(a.hallPosition());
                calibrationSink.printChar(')');
                calibrationSink.endLine();
            }
        }
    }

    // "ERR T refused <name> ... (no position)", at most once per
    // ACTUATOR_T_REFUSED_LOG_INTERVAL_MS. refused: bits by actuator index.
    void printRefusedTargets(uint32_t refused)
    {
        const uint16_t now16 = static_cast<uint16_t>(millis());
        if (calibrationSink.output() == nullptr ||
            static_cast<uint16_t>(now16 - lastRefusedMs_) < ACTUATOR_T_REFUSED_LOG_INTERVAL_MS)
            return;
        lastRefusedMs_ = now16;
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL("ERR T refused"));
        for (size_t j = 0; j < count && j < 32; j++)
        {
            if ((refused & (1UL << j)) == 0)
                continue;
            calibrationSink.printChar(' ');
            calibrationSink.printText(actuators[j]->getName());
        }
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL(" (no position)"));
        calibrationSink.endLine();
    }

    // "SENS <ROLE> <NAME>:<c> ..." on quadrature builds: when any joint's code
    // changed, else every ACTUATOR_SENS_PERIOD_MS. The signature packs 3 bits per
    // joint (the last ten when there are more).
    void pollSensorStateLine()
    {
        if (!hallHwHasQuadrature() || calibrationSink.output() == nullptr)
            return;
        uint32_t sig = 0;
        for (size_t i = 0; i < count; i++)
            sig = (sig << 3) | actuators[i]->sensorState();
        const uint16_t now16 = static_cast<uint16_t>(millis());
        if (sig == sensSig_ && static_cast<uint16_t>(now16 - lastSensMs_) < ACTUATOR_SENS_PERIOD_MS)
            return;
        sensSig_ = sig;
        lastSensMs_ = now16;
        calibrationSink.printFlash(KRABBY_FLASH_LITERAL("SENS "));
        calibrationSink.printText(boardRoleLabel(calibrationRole));
        for (size_t i = 0; i < count; i++)
        {
            calibrationSink.printChar(' ');
            calibrationSink.printText(actuators[i]->getName());
            calibrationSink.printChar(':');
            calibrationSink.printChar(actuators[i]->sensorStateCode());
        }
        calibrationSink.endLine();
    }

    // Motors off and no position target on every actuator.
    void stopAllActuators()
    {
        for (size_t i = 0; i < count; i++)
            actuators[i]->stopMotor();
    }

    // The v2 record: limits and kinds as they stand, magic and check. The magic
    // is cleared first and written last, so a write cut short (power lost) leaves
    // no record rather than old and new bytes that might still pass the check.
    void writeCalibrationToEeprom()
    {
        CalData data = {};
        for (size_t i = 0; i < count && i < LEG_CALIBRATION_JOINT_COUNT; i++)
        {
            data.minVals[i] = static_cast<int16_t>(actuators[i]->minStop);
            data.maxVals[i] = static_cast<int16_t>(actuators[i]->maxStop);
            data.sensorBits = static_cast<uint16_t>(
                data.sensorBits | (static_cast<uint16_t>(actuators[i]->sensor) << (2 * i)));
        }
        data.magic = CAL_DATA_MAGIC;
        data.check = calDataCheck(data);
        const int magicAddr = static_cast<int>(EEPROM_ACTUATOR_CAL_ADDR + offsetof(CalData, magic));
        EEPROM.put(magicAddr, static_cast<uint16_t>(0));
        const uint16_t magic = data.magic;
        data.magic = 0;
        EEPROM.put(EEPROM_ACTUATOR_CAL_ADDR, data);
        EEPROM.put(magicAddr, magic);
    }

    LinearActuator **actuators;
    size_t count;
    LegCalibration calibration;
    PrintCalibrationSink calibrationSink{&Serial};
    BoardRole calibrationRole = ROLE_UNKNOWN;
    uint32_t sensSig_ = 0xFFFFFFFFUL;  // no SENS line yet: the first poll prints
    uint16_t lastSensMs_ = 0;
    // One interval before 0, so the first refusal prints.
    uint16_t lastRefusedMs_ = static_cast<uint16_t>(0U - ACTUATOR_T_REFUSED_LOG_INTERVAL_MS);
};

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "calibration_constants.h"
#include "calibration_end_detector.h"
#include "calibration_text_sink.h"
#include "joint_sensor.h"

// Per-board actuator calibration: drives one hip or knee at a time to its end
// of travel, slowly, and records each end. Yaw joints are never driven. Joint
// indices follow the board's actuator order: 0 = leg-A yaw, 1 = leg-A hip,
// 2 = leg-A knee, 3 = leg-B yaw, 4 = leg-B hip, 5 = leg-B knee.
//
// SEEK pass, for leg A then leg B: knee -> curl end; hip -> curl end.
// SWEEP pass, for leg A then leg B: knee -> curl end; hip -> curl end;
//   hip -> uncurl end; hip -> curl end; knee -> uncurl end; knee -> curl end.
// A hall hip's uncurl is a fixed sweep: it stops after CAL_HIP_SWEEP_EDGES
// drive edges from its proven curl end (a curled tibia meets the body before
// the hip's own end), fails if it stalls first, and saves [0, CAL_HIP_SWEEP_EDGES]
// with no stroke checks. A pot hip (builds without quadrature) still runs to
// its uncurl end. Knees always sweep their full stroke, with their hip curled.
// A hip only moves with its knee proven curled in this run, and a knee only
// leaves its curl end with its hip proven curled (legCalibrationPartnerReady;
// the knee nudge, CAL_OPT_KNEE_NUDGE and on by default, is the one exception).
//
// Sensors (joint_sensor.h): each hip and knee is a pot, a hall encoder, or not
// known yet. While driven, CAL_HALL_DETECT_EDGES edges in the driven direction
// make it a hall, and as many against the drive fail the move. A pot is told
// only at rest, from two snapshots taken with the motor stopped. On quadrature
// builds the pot pins carry hall B, so a pot reading never stops, classifies or
// fails a joint that is not proven Pot.
//
// One move: stop everything; ramp |PWM| from 0 by CAL_PWM_RAMP_STEP every
// CAL_PWM_RAMP_INTERVAL_MS up to the calibration PWM. On quadrature builds a
// joint not proven Pot (every drive but the nudge out) then runs until the end
// detector (calibration_end_detector.h) decides: a retracting rod's limit
// switch (count stopped, current dropped: "by=switch"), an extending rod's
// mechanical stop (speed collapsed, current rose: "by=stop"), a stall with
// neither signature after a running phase ("by=stuck": the pass fails "stuck
// mid-stroke"), or, for a drive that never ran (a press at an end, a short
// return, a joint slower than arming speed) or a curl that armed only in its
// own end approach, a plain stall. Otherwise (proven pot, builds without quadrature),
// after CAL_START_GRACE_MS every CAL_STALL_WINDOW_MS window must advance the
// joint in the driven direction (a proven pot by CAL_STALL_MIN_PROGRESS counts
// of its smoothed reading, any other joint by CAL_HALL_STALL_MIN_EDGES edges);
// the first window that does not is the end of travel. Either way: stop, wait
// CAL_SETTLE_MS, read the end. A move still running CAL_MOVE_TIMEOUT_MS after
// it started fails.
//
// A curl that moved (CAL_HALL_CURL_EVIDENCE_EDGES edges or CAL_MIN_TRAVEL_COUNTS
// counts) and then stalled is at its curl end; after a boot an obstacle looks
// the same. A curl with no motion is accepted only at a curl end proven earlier
// (a hall home from this boot, a pot end read on an earlier accepted curl, or a
// measured pot limit). Otherwise a hip gets a verification nudge (out, then a
// full curl back) and a knee fails (or, with CAL_OPT_KNEE_NUDGE, gets a shorter
// nudge at no more than CAL_KNEE_NUDGE_MAX_PWM; a hall knee's curl back then
// needs only CAL_KNEE_RETURN_MIN_EDGES of motion, since a knee parked deeper
// than the calibration PWM reaches comes back only a few edges). Ending more
// than the curl-end tolerance short of a curl end proven earlier fails, with or
// without motion; ending deeper is accepted and moves a hall home. A failed pass stops before the partner joint moves. A failure or
// abort that leaves a joint nudged out says so (WARN). Only a completed SWEEP
// saves.

// The board the sequencer drives. Sensors must be refreshed before every update().
class LegCalibrationPlant
{
public:
    // Signed PWM through the actuator's normal drive path (EN high); +PWM extends.
    virtual void drive(uint8_t joint, int pwm) = 0;
    // Every actuator PWM 0 / EN low, no position target.
    virtual void stopAll() = 0;
    // Smoothed pot reading (avgPot); Pot windows and the Pot nudge stop only.
    virtual int rawPot(uint8_t joint) = 0;
    // Mean of POT_PROBE_SAMPLE_COUNT direct pot reads, taken at rest.
    virtual int potSnapshot(uint8_t joint) = 0;
    // Pot input flagged open by the pull-up probe; consulted only for proven Pot.
    virtual bool isPotOpen(uint8_t joint) = 0;
    // hallHwHasQuadrature(): signed hall counts on this build.
    virtual bool hasQuadratureHall() = 0;
    // Raw signed hall count.
    virtual int32_t hallCount(uint8_t joint) = 0;
    // getRawPos(): the smoothed pot for Pot, the hall frame otherwise.
    virtual int position(uint8_t joint) = 0;
    virtual JointSensor sensor(uint8_t joint) = 0;
    virtual void setSensor(uint8_t joint, JointSensor kind) = 0;
    // Hall home proven this boot.
    virtual bool isHomed(uint8_t joint) = 0;
    // Hall home at the current count; clearHome marks the home lost.
    virtual void setHome(uint8_t joint) = 0;
    virtual void clearHome(uint8_t joint) = 0;
    // Current tracker evidence is CurrentAbsent.
    virtual bool isCurrentAbsent(uint8_t joint) = 0;
    // Motor current in tenths of an IS count: round(10 * avgIS), clamped to
    // [0, 65535] (calibrationCurrentTenths). The end detector's input.
    virtual uint16_t motorCurrentTenths(uint8_t joint) = 0;
    // Measured Pot limit (extended: maxStop, else minStop); false unless the
    // joint is Pot with valid limits.
    virtual bool measuredLimit(uint8_t joint, bool extended, int &value) = 0;
    // Saved hall stroke (maxStop - minStop) of a Hall joint with valid limits, else 0.
    virtual int savedStroke(uint8_t joint) = 0;
    virtual uint32_t nowMs() = 0;
    virtual const char *jointName(uint8_t joint) = 0;
    // boardRoleLabel() of this board.
    virtual const char *roleLabel() = 0;
    // The board's line output (mainSerial).
    virtual CalibrationTextSink &out() = 0;
    // Sets the joint's limits and marks them valid.
    virtual void storeLimits(uint8_t joint, int minStop, int maxStop) = 0;
    // Persist all six actuators' limits and kinds as they now stand.
    virtual void saveLimits() = 0;

protected:
    ~LegCalibrationPlant() = default;
};

enum class LegCalibrationOutcome : uint8_t
{
    Idle,       // never started (or reset)
    Running,
    Completed,
    Failed,
    Aborted,
};

enum class LegCalibrationFailure : uint8_t
{
    None,
    PotNotConnected,
    NoEndOfTravel,
    BarelyMoved,
    NotAtCurlEnd,
    WrongDirection,
    PotWrongWay,
    NoFeedback,
    NoMotorCurrent,
    NoMotion,
    ShortOfCurlEnd,
    NoCurlBack,
    StrokeOutOfRange,
    StrokeShort,
    SweepShort,
    StuckMidStroke,
    PartnerNotCurled,
    Aborted,
};

enum class LegCalibrationMovePhase : uint8_t
{
    Ramp,     // magnitude climbing to the calibration PWM
    Grace,    // full PWM, before the first stall window
    Windows,  // stall windows: each must advance the joint
    Detect,   // full PWM, until the end detector decides (hall drives)
    Nudge,    // full PWM, verification nudge until its bound
    Settle,   // motor stopped, waiting for the end reading
};

enum class LegCalibrationMoveStage : uint8_t
{
    Normal,    // a step's move
    NudgeOut,  // the verification nudge away from the curl end
    Return,    // the full curl back after a nudge
};

static constexpr uint8_t LEG_CALIBRATION_JOINT_COUNT = 6;
static constexpr uint8_t LEG_CALIBRATION_JOINTS_PER_LEG = 3;
static constexpr uint8_t LEG_CALIBRATION_HIP_OFFSET = 1;
static constexpr uint8_t LEG_CALIBRATION_KNEE_OFFSET = 2;
static constexpr uint8_t LEG_CALIBRATION_SEEK_STEPS_PER_LEG = 2;
static constexpr uint8_t LEG_CALIBRATION_SWEEP_STEPS_PER_LEG = 6;
static constexpr uint8_t LEG_CALIBRATION_LEG_COUNT = 2;

inline bool isLegCalibrationHip(uint8_t joint)
{
    return joint % LEG_CALIBRATION_JOINTS_PER_LEG == LEG_CALIBRATION_HIP_OFFSET;
}

// PWM sign that moves joint toward its curl end (curl) or away from it.
inline int legCalibrationPwmSign(uint8_t joint, bool curl)
{
    const int curlSign = isLegCalibrationHip(joint)
        ? CAL_HIP_CURL_PWM_SIGN
        : CAL_KNEE_CURL_PWM_SIGN;
    return curl ? curlSign : -curlSign;
}

// Whether joint may be driven (curl or not; isNudge: a verification nudge out)
// given the joints proven curled in this run (curledMask) and the options: a
// hip only with its knee curled; a knee curls freely and leaves its curl end
// only with its hip curled, or as a nudge with CAL_OPT_KNEE_NUDGE.
inline bool legCalibrationPartnerReady(
    uint8_t curledMask,
    uint8_t joint,
    bool curl,
    bool isNudge,
    uint8_t options)
{
    const uint8_t base = joint - joint % LEG_CALIBRATION_JOINTS_PER_LEG;
    if (isLegCalibrationHip(joint))
        return (curledMask & (1U << (base + LEG_CALIBRATION_KNEE_OFFSET))) != 0;
    if (curl)
        return true;
    if (isNudge && (options & CAL_OPT_KNEE_NUDGE) != 0)
        return true;
    return (curledMask & (1U << (base + LEG_CALIBRATION_HIP_OFFSET))) != 0;
}

inline void printLegCalibrationDirection(CalibrationTextSink &out, bool curl)
{
    out.printFlash(curl ? KRABBY_FLASH_LITERAL("curl") : KRABBY_FLASH_LITERAL("uncurl"));
}

inline void printLegCalibrationUnit(CalibrationTextSink &out, bool pot)
{
    out.printFlash(pot ? KRABBY_FLASH_LITERAL(" counts") : KRABBY_FLASH_LITERAL(" edges"));
}

// A failure reason without prefix or line end (LegCalibration::printFailureReason).
// kind sets the units: counts for Pot, edges otherwise. curl: the failed move's
// direction; value, value2: the failure's numbers (Aborted: value is the cause);
// pwm: the calibration PWM. Each fits CAL_ROBOT_REASON_MAX_CHARS with a
// 4-character joint name and 5-digit signed numbers.
inline void printLegCalibrationFailureReason(
    CalibrationTextSink &out,
    LegCalibrationFailure failure,
    const char *joint,
    JointSensor kind,
    bool curl,
    int32_t value,
    int value2,
    int pwm)
{
    if (failure == LegCalibrationFailure::None)
        return;
    if (failure == LegCalibrationFailure::Aborted)
    {
        out.printFlash(KRABBY_FLASH_LITERAL("ABORTED "));
        out.printChar(static_cast<char>(value));
        return;
    }
    const bool pot = kind == JointSensor::Pot;
    out.printText(joint);
    switch (failure)
    {
        case LegCalibrationFailure::PotNotConnected:
            out.printFlash(KRABBY_FLASH_LITERAL(" pot not connected"));
            return;
        case LegCalibrationFailure::NoEndOfTravel:
            out.printChar(' ');
            printLegCalibrationDirection(out, curl);
            out.printFlash(KRABBY_FLASH_LITERAL(": no end of travel in "));
            out.printNumber(static_cast<long>(CAL_MOVE_TIMEOUT_MS / 1000UL));
            out.printFlash(KRABBY_FLASH_LITERAL(" s"));
            return;
        case LegCalibrationFailure::BarelyMoved:
            out.printFlash(KRABBY_FLASH_LITERAL(" barely moved ("));
            out.printNumber(value);
            printLegCalibrationUnit(out, pot);
            out.printFlash(KRABBY_FLASH_LITERAL(") at PWM "));
            out.printNumber(pwm);
            out.printFlash(pot
                ? KRABBY_FLASH_LITERAL(": check pot wiring or raise speed")
                : KRABBY_FLASH_LITERAL(": check hall wiring or raise speed"));
            return;
        case LegCalibrationFailure::NotAtCurlEnd:
            out.printFlash(pot
                ? KRABBY_FLASH_LITERAL(" did not reach its curl end (pot=")
                : KRABBY_FLASH_LITERAL(" did not reach its curl end (hall="));
            out.printNumber(value);
            out.printChar(')');
            return;
        case LegCalibrationFailure::WrongDirection:
            out.printFlash(KRABBY_FLASH_LITERAL(
                " hall counts the wrong way: check motor or hall A/B wiring"));
            return;
        case LegCalibrationFailure::PotWrongWay:
            out.printFlash(KRABBY_FLASH_LITERAL(
                " pot moves the wrong way: check motor or pot end wires"));
            return;
        case LegCalibrationFailure::NoFeedback:
            // value: the PWM of a nudge run below CAL_DEFAULT_PWM (a stored
            // joint may not break free there), else 0.
            if (value != 0)
            {
                out.printFlash(KRABBY_FLASH_LITERAL(" no feedback at PWM "));
                out.printNumber(value);
                out.printFlash(KRABBY_FLASH_LITERAL(": check sensor cable, or calibrate at PWM "));
                out.printNumber(CAL_DEFAULT_PWM);
                return;
            }
            out.printFlash(KRABBY_FLASH_LITERAL(
                " no feedback: no hall edges, pot not moving (check sensor cable)"));
            return;
        case LegCalibrationFailure::NoMotorCurrent:
            out.printFlash(KRABBY_FLASH_LITERAL(" no motor current at PWM "));
            out.printNumber(pwm);
            out.printFlash(KRABBY_FLASH_LITERAL(": check motor wiring"));
            return;
        case LegCalibrationFailure::NoMotion:
            out.printFlash(KRABBY_FLASH_LITERAL(
                " no motion seen: jog it off its curl end, then retry"));
            return;
        case LegCalibrationFailure::ShortOfCurlEnd:
            out.printFlash(KRABBY_FLASH_LITERAL(" stopped "));
            out.printNumber(value);
            printLegCalibrationUnit(out, pot);
            out.printFlash(KRABBY_FLASH_LITERAL(" short of its curl end"));
            return;
        case LegCalibrationFailure::NoCurlBack:
            out.printFlash(KRABBY_FLASH_LITERAL(" curled back only "));
            out.printNumber(value);
            printLegCalibrationUnit(out, pot);
            out.printFlash(KRABBY_FLASH_LITERAL(" after its nudge"));
            return;
        case LegCalibrationFailure::StrokeOutOfRange:
            out.printFlash(KRABBY_FLASH_LITERAL(" stroke "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges, outside "));
            out.printNumber(CAL_HALL_MIN_STROKE_EDGES);
            out.printChar('-');
            out.printNumber(CAL_HALL_MAX_STROKE_EDGES);
            return;
        case LegCalibrationFailure::StrokeShort:
            out.printFlash(KRABBY_FLASH_LITERAL(" stroke "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges, saved "));
            out.printNumber(value2);
            out.printFlash(KRABBY_FLASH_LITERAL(": obstacle?"));
            return;
        case LegCalibrationFailure::SweepShort:
            out.printFlash(KRABBY_FLASH_LITERAL(" stopped "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges short of its "));
            out.printNumber(CAL_HIP_SWEEP_EDGES);
            out.printFlash(KRABBY_FLASH_LITERAL("-edge sweep: obstacle?"));
            return;
        case LegCalibrationFailure::StuckMidStroke:
            out.printFlash(KRABBY_FLASH_LITERAL(" stuck mid-stroke after "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges at PWM "));
            out.printNumber(pwm);
            out.printFlash(KRABBY_FLASH_LITERAL(": raise PWM or check load"));
            return;
        default:  // PartnerNotCurled
            out.printFlash(KRABBY_FLASH_LITERAL(" partner not proven curled"));
            return;
    }
}

inline uint8_t legCalibrationStepCount(uint8_t pass)
{
    return LEG_CALIBRATION_LEG_COUNT * (pass == CAL_PASS_SEEK
        ? LEG_CALIBRATION_SEEK_STEPS_PER_LEG
        : LEG_CALIBRATION_SWEEP_STEPS_PER_LEG);
}

struct LegCalibrationStep
{
    uint8_t joint;
    bool curl;
    // SWEEP only: this move completes the joint's uncurl-and-back measurement.
    bool completesJoint;
};

// Step `step` (0-based) of `pass` (CAL_PASS_SEEK or CAL_PASS_SWEEP).
inline LegCalibrationStep legCalibrationStep(uint8_t pass, uint8_t step)
{
    const uint8_t perLeg = pass == CAL_PASS_SEEK
        ? LEG_CALIBRATION_SEEK_STEPS_PER_LEG
        : LEG_CALIBRATION_SWEEP_STEPS_PER_LEG;
    const uint8_t base = (step / perLeg) * LEG_CALIBRATION_JOINTS_PER_LEG;
    const uint8_t k = step % perLeg;
    LegCalibrationStep result;
    if (pass == CAL_PASS_SEEK)
    {
        // knee curl, hip curl
        result.joint = base + (k == 0 ? LEG_CALIBRATION_KNEE_OFFSET : LEG_CALIBRATION_HIP_OFFSET);
        result.curl = true;
        result.completesJoint = false;
        return result;
    }
    // knee curl, hip curl, hip uncurl, hip curl, knee uncurl, knee curl
    const bool isHip = k >= 1 && k <= 3;
    result.joint = base + (isHip ? LEG_CALIBRATION_HIP_OFFSET : LEG_CALIBRATION_KNEE_OFFSET);
    result.curl = k != 2 && k != 4;
    result.completesJoint = k == 3 || k == 5;
    return result;
}

class LegCalibration
{
public:
    LegCalibration() { reset(); }

    // Silently returns to Idle; does not touch the plant (host-sim reset).
    void reset()
    {
        outcome_ = LegCalibrationOutcome::Idle;
        failure_ = LegCalibrationFailure::None;
        phase_ = LegCalibrationMovePhase::Ramp;
        stage_ = LegCalibrationMoveStage::Normal;
        passes_ = 0;
        pass_ = 0;
        step_ = 0;
        pwm_ = 0;
        options_ = 0;
        joint_ = 0;
        curl_ = false;
        repeated_ = false;
        nudgeCurrentAbsent_ = false;
        sweepReached_ = false;
        detecting_ = false;
        endKind_ = CalibrationEnd::None;
        nudgeDriveMs_ = 0;
        sign_ = 0;
        rampPwm_ = 0;
        curledMask_ = 0;
        curlEndKnownMask_ = 0;
        reportedMask_ = 0;
        nudgeKind_ = JointSensor::Unknown;
        failKind_ = JointSensor::Unknown;
        failCurl_ = false;
        startHall_ = 0;
        startSnap_ = 0;
        startPot_ = 0;
        windowStartPos_ = 0;
        preNudgePos_ = 0;
        nudgeEdges_ = 0;
        failValue_ = 0;
        failValue2_ = 0;
        moveStartMs_ = 0;
        phaseStartMs_ = 0;
        failJoint_ = nullptr;
        for (uint8_t joint = 0; joint < LEG_CALIBRATION_JOINT_COUNT; ++joint)
        {
            stagedMin_[joint] = 0;
            stagedMax_[joint] = 0;
            curlEnd_[joint] = 0;
        }
    }

    // Starts the requested passes (SEEK before SWEEP) at PWM magnitude pwm with
    // the CAL_OPT_* bits in options: emits "CAL <ROLE> PASS <SEEK|SWEEP> pwm=<p>"
    // and begins the first move. Returns false, without touching the plant,
    // while running, when isValidCalibrationRequest(passes, pwm) is false, or
    // when options has a bit outside CAL_OPT_ALL.
    bool start(LegCalibrationPlant &plant, uint8_t passes, int pwm,
               uint8_t options = CAL_DEFAULT_OPTIONS)
    {
        if (outcome_ == LegCalibrationOutcome::Running ||
            !isValidCalibrationRequest(passes, pwm) ||
            (options & ~CAL_OPT_ALL) != 0)
        {
            return false;
        }

        passes_ = passes;
        pwm_ = static_cast<uint8_t>(pwm);
        options_ = options;
        curledMask_ = 0;
        reportedMask_ = 0;
        repeated_ = false;
        failure_ = LegCalibrationFailure::None;
        failJoint_ = nullptr;
        outcome_ = LegCalibrationOutcome::Running;
        beginPass(plant, (passes & CAL_PASS_SEEK) != 0 ? CAL_PASS_SEEK : CAL_PASS_SWEEP);
        return true;
    }

    // Advances the running pass; call once per loop after refreshing sensors.
    void update(LegCalibrationPlant &plant)
    {
        if (outcome_ != LegCalibrationOutcome::Running)
            return;

        const uint32_t now = plant.nowMs();
        if (phase_ != LegCalibrationMovePhase::Settle)
        {
            if (now - moveStartMs_ >= CAL_MOVE_TIMEOUT_MS)
            {
                fail(plant, LegCalibrationFailure::NoEndOfTravel, 0);
                return;
            }
            if (!checkHallEvidence(plant, now))
                return;
            if (stage_ == LegCalibrationMoveStage::NudgeOut && nudgeDone(plant, now))
            {
                stopToSettle(plant, now);
                return;
            }
            // A hall hip's fixed sweep stops at its target, never at a stall.
            if (isHallHipSweep(plant) &&
                hallDriveEdges(sign_, plant.hallCount(joint_) - startHall_) >= CAL_HIP_SWEEP_EDGES)
            {
                sweepReached_ = true;
                stopToSettle(plant, now);
                return;
            }
            if (detecting_ && (phase_ == LegCalibrationMovePhase::Ramp ||
                               phase_ == LegCalibrationMovePhase::Detect))
            {
                const CalibrationEnd kind = end_.update(
                    now, hallDriveEdges(sign_, plant.hallCount(joint_) - startHall_),
                    plant.motorCurrentTenths(joint_), phase_ != LegCalibrationMovePhase::Ramp);
                if (kind != CalibrationEnd::None)
                {
                    endKind_ = kind;
                    stopToSettle(plant, now);
                    return;
                }
            }
        }

        switch (phase_)
        {
            case LegCalibrationMovePhase::Ramp:
            {
                if (now - phaseStartMs_ < CAL_PWM_RAMP_INTERVAL_MS)
                    return;
                phaseStartMs_ = now;
                const uint8_t top = movePwm();
                rampPwm_ = rampPwm_ + CAL_PWM_RAMP_STEP >= top
                    ? top
                    : static_cast<uint8_t>(rampPwm_ + CAL_PWM_RAMP_STEP);
                plant.drive(joint_, sign_ * static_cast<int>(rampPwm_));
                if (rampPwm_ == top)
                {
                    phase_ = stage_ == LegCalibrationMoveStage::NudgeOut ? LegCalibrationMovePhase::Nudge
                        : detecting_                                     ? LegCalibrationMovePhase::Detect
                                                                         : LegCalibrationMovePhase::Grace;
                }
                return;
            }

            case LegCalibrationMovePhase::Grace:
                if (now - phaseStartMs_ < CAL_START_GRACE_MS)
                    return;
                startWindow(plant, now);
                phase_ = LegCalibrationMovePhase::Windows;
                return;

            case LegCalibrationMovePhase::Windows:
            {
                if (now - phaseStartMs_ < CAL_STALL_WINDOW_MS)
                    return;
                // The driven direction is fixed by the PWM sign for every kind.
                const int32_t progress = static_cast<int32_t>(sign_) *
                    (static_cast<int32_t>(plant.position(joint_)) - windowStartPos_);
                const int threshold = plant.sensor(joint_) == JointSensor::Pot
                    ? CAL_STALL_MIN_PROGRESS
                    : CAL_HALL_STALL_MIN_EDGES;
                if (progress >= threshold)
                {
                    startWindow(plant, now);
                    return;
                }
                // End of travel: stop, then let the reading settle.
                stopToSettle(plant, now);
                return;
            }

            case LegCalibrationMovePhase::Detect:  // the end detector ends it
            case LegCalibrationMovePhase::Nudge:   // nudgeDone() ends it
                return;

            default:  // Settle
                if (now - phaseStartMs_ < CAL_SETTLE_MS)
                    return;
                if (stage_ == LegCalibrationMoveStage::NudgeOut)
                    finishNudge(plant);
                else
                    finishMove(plant);
                return;
        }
    }

    // Stops every actuator and ends a running calibration with nothing saved.
    // Hall homes are kept (the counts keep tracking). Emits
    // "CAL <ROLE> ABORTED <cause>". Returns false (and does nothing) when idle.
    bool abort(LegCalibrationPlant &plant, char cause)
    {
        if (outcome_ != LegCalibrationOutcome::Running)
            return false;
        plant.stopAll();
        warnIfNudgedOut(plant);
        failure_ = LegCalibrationFailure::Aborted;
        failJoint_ = nullptr;
        failValue_ = cause;
        outcome_ = LegCalibrationOutcome::Aborted;
        CalibrationTextSink &out = plant.out();
        printPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL("ABORTED "));
        out.printChar(cause);
        out.endLine();
        return true;
    }

    // Reason of the last Failed or Aborted run, without prefix or line end, for
    // example "<joint> pot not connected", "<joint> no motion seen: jog it off
    // its curl end, then retry" or "ABORTED <cause>"
    // (printLegCalibrationFailureReason). Prints nothing otherwise.
    void printFailureReason(CalibrationTextSink &out) const
    {
        printLegCalibrationFailureReason(out, failure_, failJoint_, failKind_, failCurl_,
                                         failValue_, failValue2_, pwm_);
    }

    bool isRunning() const { return outcome_ == LegCalibrationOutcome::Running; }
    LegCalibrationOutcome outcome() const { return outcome_; }
    LegCalibrationFailure failure() const { return failure_; }
    // Current pass bit (CAL_PASS_SEEK / CAL_PASS_SWEEP) and step within it.
    uint8_t currentPass() const { return pass_; }
    uint8_t currentStep() const { return step_; }
    // Joint of the current (or last) move and its direction.
    uint8_t currentJoint() const { return joint_; }
    bool isCurrentMoveCurl() const { return curl_; }
    LegCalibrationMovePhase movePhase() const { return phase_; }
    LegCalibrationMoveStage moveStage() const { return stage_; }
    // The current (or last) drive runs the end detector, and what it decided
    // (None: the drive ended otherwise, or is still running).
    bool isDetectingEnd() const { return detecting_; }
    CalibrationEnd lastEnd() const { return endKind_; }
    const CalibrationEndDetector &endDetector() const { return end_; }
    // Ramped PWM magnitude most recently applied in the current move.
    int rampPwm() const { return rampPwm_; }
    int pwm() const { return pwm_; }
    uint8_t passes() const { return passes_; }
    // CAL_OPT_* bits of the current (or last) run.
    uint8_t options() const { return options_; }
    // Bit j set: joint j's most recent completed move was accepted at its curl
    // end in this run and it has not been driven since.
    uint8_t curledMask() const { return curledMask_; }
    bool isJointCurled(uint8_t joint) const { return (curledMask_ & (1U << joint)) != 0; }
    // Staged end readings of the running pass: min = end reached with -PWM,
    // max = with +PWM; for a hall joint, canonical [0, CAL_HIP_SWEEP_EDGES] (hip) or [-S, 0] (knee)
    // once it completes.
    int stagedMin(uint8_t joint) const { return stagedMin_[joint]; }
    int stagedMax(uint8_t joint) const { return stagedMax_[joint]; }

private:
    static void printPrefix(LegCalibrationPlant &plant, CalibrationTextSink &out)
    {
        out.printFlash(KRABBY_FLASH_LITERAL("CAL "));
        out.printText(plant.roleLabel());
        out.printChar(' ');
    }

    static void printKind(CalibrationTextSink &out, JointSensor kind)
    {
        out.printFlash(kind == JointSensor::Pot
            ? KRABBY_FLASH_LITERAL("pot")
            : KRABBY_FLASH_LITERAL("hall"));
    }

    // "CAL <ROLE> <word><joint>" without line end.
    void printJointLine(LegCalibrationPlant &plant, CalibrationTextSink &out, const char *flashWord)
    {
        printPrefix(plant, out);
        out.printFlash(flashWord);
        out.printText(plant.jointName(joint_));
    }

    static int32_t magnitude(int32_t value) { return value < 0 ? -value : value; }

    // Curl-end tolerance and least travel away from the uncurl end, per kind.
    static int32_t curlEndTolerance(JointSensor kind)
    {
        return kind == JointSensor::Pot ? CAL_CURL_END_TOLERANCE_COUNTS : CAL_HALL_CURL_END_TOLERANCE_EDGES;
    }

    static int32_t minimumLeave(JointSensor kind)
    {
        return kind == JointSensor::Pot ? CAL_MIN_TRAVEL_COUNTS : CAL_HALL_MIN_STROKE_EDGES;
    }

    void beginPass(LegCalibrationPlant &plant, uint8_t pass)
    {
        pass_ = pass;
        step_ = 0;
        CalibrationTextSink &out = plant.out();
        printPrefix(plant, out);
        out.printFlash(pass_ == CAL_PASS_SEEK
            ? KRABBY_FLASH_LITERAL("PASS SEEK pwm=")
            : KRABBY_FLASH_LITERAL("PASS SWEEP pwm="));
        out.printNumber(pwm_);
        out.endLine();
        beginMove(plant);
    }

    void beginMove(LegCalibrationPlant &plant)
    {
        const LegCalibrationStep step = legCalibrationStep(pass_, step_);
        joint_ = step.joint;
        beginDrive(plant, LegCalibrationMoveStage::Normal, step.curl);
    }

    // Every drive of joint_ starts here: a step's move, a nudge or a return.
    void beginDrive(LegCalibrationPlant &plant, LegCalibrationMoveStage stage, bool curl)
    {
        curl_ = curl;
        sign_ = static_cast<int8_t>(legCalibrationPwmSign(joint_, curl_));
        plant.stopAll();
        // The partner gate cannot fail through the step table; it guards the
        // table. The pot-open probe applies to proven Pot joints only.
        const LegCalibrationFailure gate =
            !legCalibrationPartnerReady(curledMask_, joint_, curl_,
                                        stage == LegCalibrationMoveStage::NudgeOut, options_)
                ? LegCalibrationFailure::PartnerNotCurled
            : plant.sensor(joint_) == JointSensor::Pot && plant.isPotOpen(joint_)
                ? LegCalibrationFailure::PotNotConnected
                : LegCalibrationFailure::None;
        if (gate != LegCalibrationFailure::None)
        {
            fail(plant, gate, 0);
            return;
        }

        curledMask_ &= static_cast<uint8_t>(~(1U << joint_));
        sweepReached_ = false;
        startHall_ = plant.hallCount(joint_);
        startSnap_ = plant.potSnapshot(joint_);
        startPot_ = plant.rawPot(joint_);
        moveStartMs_ = plant.nowMs();
        phaseStartMs_ = moveStartMs_;
        rampPwm_ = 0;
        phase_ = LegCalibrationMovePhase::Ramp;
        stage_ = stage;
        // Hall drives on quadrature builds (Unknown included) end by the
        // detector; the nudge out keeps its own bounds. A Return drives at
        // least as long as its nudge did, so a blind joint is driven back; one
        // whose nudge proved no motion (no feedback: it may have moved blind)
        // drives CAL_STALL_WINDOW_MS longer, since joints curl back slower than
        // they uncurl (hips at 128: 134-144 edges out in 2.0 s, 112-131 back).
        detecting_ = plant.hasQuadratureHall() && plant.sensor(joint_) != JointSensor::Pot &&
                     stage != LegCalibrationMoveStage::NudgeOut;
        endKind_ = CalibrationEnd::None;
        uint32_t minStill = CAL_END_STILL_MIN_DRIVE_MS;
        if (stage == LegCalibrationMoveStage::Return)
        {
            const uint32_t nudged = nudgeKind_ == JointSensor::Unknown
                ? static_cast<uint32_t>(nudgeDriveMs_) + CAL_STALL_WINDOW_MS
                : nudgeDriveMs_;
            if (nudged > minStill)
                minStill = nudged;
        }
        end_.begin(moveStartMs_, sign_ < 0, !curl_, minStill);

        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("MOVE "));
        out.printChar(' ');
        if (stage == LegCalibrationMoveStage::NudgeOut)
            out.printFlash(KRABBY_FLASH_LITERAL("nudge"));
        else
            printLegCalibrationDirection(out, curl_);
        out.endLine();
    }

    void startWindow(LegCalibrationPlant &plant, uint32_t now)
    {
        phaseStartMs_ = now;
        windowStartPos_ = plant.position(joint_);
    }

    void stopToSettle(LegCalibrationPlant &plant, uint32_t now)
    {
        if (stage_ == LegCalibrationMoveStage::NudgeOut)
            nudgeDriveMs_ = static_cast<uint16_t>(now - moveStartMs_);
        plant.stopAll();
        phaseStartMs_ = now;
        phase_ = LegCalibrationMovePhase::Settle;
    }

    // Hall evidence while driven. Returns false when the move failed.
    bool checkHallEvidence(LegCalibrationPlant &plant, uint32_t now)
    {
        const int32_t delta = plant.hallCount(joint_) - startHall_;
        const MoveFeedback feedback = classifyHallMotion(plant.hasQuadratureHall(), sign_, delta);
        if (feedback == MoveFeedback::HallWrongWay)
        {
            fail(plant, LegCalibrationFailure::WrongDirection, hallDriveEdges(sign_, delta));
            return false;
        }
        if (feedback == MoveFeedback::Hall)
        {
            const JointSensor old = plant.sensor(joint_);
            // A pot window was in pot counts: restart it in the hall frame.
            if (noteKind(plant, JointSensor::Hall) && old == JointSensor::Pot &&
                phase_ == LegCalibrationMovePhase::Windows)
            {
                startWindow(plant, now);
            }
        }
        return true;
    }

    // PWM magnitude the current drive ramps to: the calibration PWM, but never
    // more than CAL_KNEE_NUDGE_MAX_PWM for a knee nudge.
    uint8_t movePwm() const
    {
        if (stage_ == LegCalibrationMoveStage::NudgeOut && !isLegCalibrationHip(joint_) &&
            pwm_ > CAL_KNEE_NUDGE_MAX_PWM)
        {
            return static_cast<uint8_t>(CAL_KNEE_NUDGE_MAX_PWM);
        }
        return pwm_;
    }

    // The SWEEP hip uncurl of a hall hip (Normal uncurls exist only in SWEEP):
    // a fixed sweep of CAL_HIP_SWEEP_EDGES drive edges from its curl end.
    bool isHallHipSweep(LegCalibrationPlant &plant) const
    {
        return !curl_ && stage_ == LegCalibrationMoveStage::Normal &&
               isLegCalibrationHip(joint_) && plant.sensor(joint_) == JointSensor::Hall;
    }

    // The verification nudge has gone far enough, or reached its time cap. A
    // knee with no hall feedback runs to its 1 s cap (calibration_constants.h).
    bool nudgeDone(LegCalibrationPlant &plant, uint32_t now)
    {
        const bool hip = isLegCalibrationHip(joint_);
        if (plant.sensor(joint_) == JointSensor::Pot)
        {
            const int32_t moved = static_cast<int32_t>(plant.rawPot(joint_)) - startPot_;
            if (magnitude(moved) >= CAL_POT_NUDGE_COUNTS)
                return true;
        }
        else if (plant.hasQuadratureHall())
        {
            const int32_t edges = hallDriveEdges(sign_, plant.hallCount(joint_) - startHall_);
            if (edges >= (hip ? CAL_HIP_NUDGE_EDGES : CAL_KNEE_NUDGE_EDGES))
                return true;
        }
        return now - moveStartMs_ >= (hip ? CAL_HIP_NUDGE_MAX_MS : CAL_KNEE_NUDGE_MAX_MS);
    }

    // A run that ends during a nudge (before its return was driven) leaves the
    // joint off its curl end: "CAL <ROLE> WARN <j> nudged out, not driven back:
    // jog it to its curl end".
    void warnIfNudgedOut(LegCalibrationPlant &plant)
    {
        if (stage_ != LegCalibrationMoveStage::NudgeOut)
            return;
        CalibrationTextSink &out = plant.out();
        printWarnPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL(" nudged out, not driven back: jog it to its curl end"));
        out.endLine();
    }

    // Records joint_'s kind and prints "CAL <ROLE> SENSOR <j> hall|pot[ (was ...)]"
    // on a change, or once per run when it is confirmed. Returns true on a change.
    bool noteKind(LegCalibrationPlant &plant, JointSensor kind)
    {
        const JointSensor old = plant.sensor(joint_);
        const uint8_t bit = static_cast<uint8_t>(1U << joint_);
        const bool changed = old != kind;
        if (changed)
        {
            plant.setSensor(joint_, kind);
            curlEndKnownMask_ &= static_cast<uint8_t>(~bit);
        }
        else if ((reportedMask_ & bit) != 0)
        {
            return false;
        }
        reportedMask_ |= bit;
        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("SENSOR "));
        out.printChar(' ');
        printKind(out, kind);
        if (changed && old != JointSensor::Unknown)
        {
            out.printFlash(KRABBY_FLASH_LITERAL(" (was "));
            printKind(out, old);
            out.printChar(')');
        }
        out.endLine();
        return changed;
    }

    // At rest after a move or nudge with few edges, on a quadrature build: pot
    // evidence from the two snapshots classifies a joint not yet proven Pot.
    // Returns false when the move failed; sets classified when it became Pot.
    bool classifyAtRest(LegCalibrationPlant &plant, int32_t delta, int endSnap, bool &classified)
    {
        classified = false;
        if (!plant.hasQuadratureHall() || plant.sensor(joint_) == JointSensor::Pot ||
            magnitude(hallDriveEdges(sign_, delta)) >= CAL_HALL_DETECT_EDGES)
        {
            return true;
        }
        const MoveFeedback feedback = classifyPotAtRest(2, sign_, delta, startSnap_, endSnap);
        if (feedback == MoveFeedback::PotWrongWay)
        {
            fail(plant, LegCalibrationFailure::PotWrongWay, 0);
            return false;
        }
        if (feedback == MoveFeedback::Pot)
        {
            noteKind(plant, JointSensor::Pot);
            classified = true;
        }
        return true;
    }

    // Pot evidence of a proven Pot joint's move (one snapshot in band on
    // quadrature builds). Returns false when the move failed.
    bool potMotion(LegCalibrationPlant &plant, int32_t delta, int endSnap, bool &motion)
    {
        const MoveFeedback feedback = classifyPotAtRest(
            plant.hasQuadratureHall() ? 1 : 0, sign_, delta, startSnap_, endSnap);
        if (feedback == MoveFeedback::PotWrongWay)
        {
            fail(plant, LegCalibrationFailure::PotWrongWay, 0);
            return false;
        }
        motion = feedback == MoveFeedback::Pot;
        return true;
    }

    // "CAL <ROLE> END <j> <curl|uncurl|nudge> ..." in the joint's units.
    void printEnd(LegCalibrationPlant &plant, JointSensor kind, int32_t endPos, int32_t travel,
                  int endSnap)
    {
        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("END "));
        out.printChar(' ');
        if (stage_ == LegCalibrationMoveStage::NudgeOut)
            out.printFlash(KRABBY_FLASH_LITERAL("nudge"));
        else
            printLegCalibrationDirection(out, curl_);
        out.printFlash(kind == JointSensor::Pot
            ? KRABBY_FLASH_LITERAL(" pot=")
            : KRABBY_FLASH_LITERAL(" hall="));
        out.printNumber(endPos);
        out.printFlash(KRABBY_FLASH_LITERAL(" travel="));
        out.printNumber(travel);
        if (kind == JointSensor::Unknown)
        {
            out.printFlash(KRABBY_FLASH_LITERAL(" pot="));
            out.printNumber(endSnap);
        }
        // Detector ends: " by=switch|stop|stuck cur=<current>/<reference>" in
        // tenths of an IS count (a plain stall prints nothing).
        if (endKind_ == CalibrationEnd::Switch || endKind_ == CalibrationEnd::Stop ||
            endKind_ == CalibrationEnd::Stuck)
        {
            out.printFlash(endKind_ == CalibrationEnd::Switch ? KRABBY_FLASH_LITERAL(" by=switch cur=")
                           : endKind_ == CalibrationEnd::Stop ? KRABBY_FLASH_LITERAL(" by=stop cur=")
                                                              : KRABBY_FLASH_LITERAL(" by=stuck cur="));
            out.printNumber(static_cast<long>(end_.endCurrent()));
            out.printChar('/');
            out.printNumber(static_cast<long>(end_.endReference()));
        }
        out.endLine();
    }

    void printWarnPrefix(LegCalibrationPlant &plant, CalibrationTextSink &out)
    {
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("WARN "));
    }

    // "CAL <ROLE> WARN <j> home moved <n> edges|counts" when a curl ended more
    // than the tolerance deeper than the curl end proven earlier.
    void warnIfHomeMoved(LegCalibrationPlant &plant, JointSensor kind, int32_t shortBy, int32_t tolerance)
    {
        if (shortBy >= -tolerance)
            return;
        CalibrationTextSink &out = plant.out();
        printWarnPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL(" home moved "));
        out.printNumber(-shortBy);
        printLegCalibrationUnit(out, kind == JointSensor::Pot);
        out.endLine();
    }

    void finishMove(LegCalibrationPlant &plant)
    {
        const uint8_t j = joint_;
        const int32_t delta = plant.hallCount(j) - startHall_;
        const int endSnap = plant.potSnapshot(j);
        const int32_t driveEdges = hallDriveEdges(sign_, delta);

        // (1) Late Pot classification. The move that just ended was windowed on
        // hall edges, so its stop was not an end of travel: repeat the step once.
        bool classified = false;
        if (!classifyAtRest(plant, delta, endSnap, classified))
            return;
        if (classified && stage_ == LegCalibrationMoveStage::Normal && !repeated_)
        {
            repeated_ = true;
            beginDrive(plant, LegCalibrationMoveStage::Normal, curl_);
            return;
        }

        // (2) Evidence of motion for this move, in the joint's units.
        const JointSensor kind = plant.sensor(j);
        bool motion = false;
        int32_t endPos = 0;
        int32_t travel = 0;
        if (kind == JointSensor::Pot)
        {
            if (!potMotion(plant, delta, endSnap, motion))
                return;
            endPos = endSnap;
            travel = static_cast<int32_t>(endSnap) - startSnap_;
        }
        else
        {
            motion = kind == JointSensor::Hall &&
                driveEdges >= (curl_ ? CAL_HALL_CURL_EVIDENCE_EDGES : CAL_HALL_MIN_STROKE_EDGES);
            endPos = plant.position(j);
            travel = ACTUATOR_HALL_EXTEND_COUNT_SIGN * delta;
        }
        printEnd(plant, kind, endPos, travel, endSnap);

        // A stall with neither end signature after a running phase is never
        // an end: the pass stops before the partner joint moves.
        if (endKind_ == CalibrationEnd::Stuck)
        {
            fail(plant, LegCalibrationFailure::StuckMidStroke, driveEdges);
            return;
        }

        // (3) Uncurl: stage the end it reached. A hall hip's fixed sweep must
        // have reached its target: stopping short of it is an obstacle.
        if (!curl_)
        {
            if (isHallHipSweep(plant))
            {
                if (!sweepReached_)
                {
                    fail(plant, LegCalibrationFailure::SweepShort, CAL_HIP_SWEEP_EDGES - driveEdges);
                    return;
                }
            }
            else if (!motion)
            {
                fail(plant, LegCalibrationFailure::BarelyMoved,
                     kind == JointSensor::Pot ? magnitude(travel) : driveEdges);
                return;
            }
            stageEnd(endPos);
            advance(plant);
            return;
        }

        // (4) Curl: compare with the curl end proven earlier, if any.
        bool proven = false;
        int32_t reference = 0;
        if (kind == JointSensor::Hall)
        {
            proven = plant.isHomed(j);
        }
        else if (kind == JointSensor::Pot)
        {
            proven = (curlEndKnownMask_ & (1U << j)) != 0;
            reference = curlEnd_[j];
        }
        // > 0: stopped short of the reference; < 0: deeper.
        const int32_t shortBy = -sign_ * (endPos - reference);
        const int32_t tolerance = curlEndTolerance(kind);

        // (4a) The curl back after a nudge: it must stall within the tolerance
        // of where the joint rested before the nudge (else "stopped N short"),
        // having moved back at least CAL_HALL_DETECT_EDGES for a hall hip,
        // CAL_KNEE_RETURN_MIN_EDGES for a hall knee (a knee parked deeper than
        // this PWM reaches comes back only a few edges) and CAL_MIN_TRAVEL_COUNTS
        // for a pot (else "curled back only N": one that cannot move toward
        // curl may be stuck mid-stroke).
        if (stage_ == LegCalibrationMoveStage::Return)
        {
            if (nudgeKind_ == JointSensor::Unknown)
            {
                // Some edges, too few for evidence: a slow joint, not a dead cable.
                // No edges: a dead cable, or, below CAL_DEFAULT_PWM (where every
                // stored joint was seen to break free), a joint that did not.
                if (!nudgeCurrentAbsent_ && nudgeEdges_ > CAL_POT_DETECT_MAX_EDGES)
                    fail(plant, LegCalibrationFailure::BarelyMoved, nudgeEdges_);
                else if (nudgeCurrentAbsent_)
                    fail(plant, LegCalibrationFailure::NoMotorCurrent, 0);
                else
                    fail(plant, LegCalibrationFailure::NoFeedback,
                         pwm_ < CAL_DEFAULT_PWM ? static_cast<int32_t>(pwm_) : 0);
                return;
            }
            if (kind != nudgeKind_)
            {
                // Classified differently since the nudge: its start is in other units.
                fail(plant, LegCalibrationFailure::NoFeedback, 0);
                return;
            }
            const bool backMotion = kind == JointSensor::Pot
                ? motion
                : driveEdges >= (isLegCalibrationHip(j) ? CAL_HALL_DETECT_EDGES : CAL_KNEE_RETURN_MIN_EDGES);
            const int32_t shortOfStart = -sign_ * (endPos - preNudgePos_);
            if (shortOfStart > tolerance)
            {
                fail(plant, LegCalibrationFailure::ShortOfCurlEnd, shortOfStart);
                return;
            }
            if (!backMotion)
            {
                fail(plant, LegCalibrationFailure::NoCurlBack,
                     kind == JointSensor::Pot ? sign_ * travel : driveEdges);
                return;
            }
            accept(plant, endPos, kind == JointSensor::Hall);
            return;
        }

        // (4b) The SWEEP curl that completes the joint (back from its uncurl end).
        if (pass_ == CAL_PASS_SWEEP && legCalibrationStep(pass_, step_).completesJoint)
        {
            const int32_t uncurlEnd = sign_ > 0 ? stagedMin_[j] : stagedMax_[j];
            if (!proven || sign_ * (endPos - uncurlEnd) < minimumLeave(kind))
            {
                fail(plant, LegCalibrationFailure::NotAtCurlEnd, endPos);
                return;
            }
            if (shortBy > tolerance)
            {
                fail(plant, LegCalibrationFailure::ShortOfCurlEnd, shortBy);
                return;
            }
            if (kind != JointSensor::Hall)
            {
                stageEnd(endPos);
            }
            else if (isLegCalibrationHip(j))
            {
                // The fixed sweep's range: no stroke to check.
                stagedMin_[j] = 0;
                stagedMax_[j] = CAL_HIP_SWEEP_EDGES;
            }
            else if (!checkAndStageStroke(plant, uncurlEnd, endPos, shortBy))
            {
                return;
            }
            reportJoint(plant, kind);
            accept(plant, endPos, kind == JointSensor::Hall && shortBy < 0);
            return;
        }

        // (4c) Any other curl: SEEK steps, SWEEP steps 0 and 1.
        if (motion)
        {
            if (proven && shortBy > tolerance)
            {
                fail(plant, LegCalibrationFailure::ShortOfCurlEnd, shortBy);
                return;
            }
            if (proven)
                warnIfHomeMoved(plant, kind, shortBy, tolerance);
            accept(plant, endPos, kind == JointSensor::Hall && (!proven || shortBy < 0));
            return;
        }
        // No motion: only at a curl end proven earlier. More than the tolerance
        // short of it fails as it does with motion: no nudge, whose return would
        // re-home the joint at whatever holds it there. A pot must also not read
        // deeper than the tolerance (a reading past its end is not proof); a
        // hall reading deeper moves the home, as with motion.
        if (proven && shortBy > tolerance)
        {
            fail(plant, LegCalibrationFailure::ShortOfCurlEnd, shortBy);
            return;
        }
        if (proven && (kind != JointSensor::Pot || shortBy >= -tolerance))
        {
            warnIfHomeMoved(plant, kind, shortBy, tolerance);
            accept(plant, endPos, kind == JointSensor::Hall && shortBy < 0);
            return;
        }
        int limit = 0;
        if (kind == JointSensor::Pot && plant.measuredLimit(j, sign_ > 0, limit) &&
            magnitude(endPos - limit) <= CAL_CURL_END_TOLERANCE_COUNTS)
        {
            accept(plant, endPos, false);
            return;
        }
        // At most one nudge per step: a Return was settled by (4a) above.
        if (isLegCalibrationHip(j) || (options_ & CAL_OPT_KNEE_NUDGE) != 0)
        {
            beginDrive(plant, LegCalibrationMoveStage::NudgeOut, false);
            return;
        }
        fail(plant, plant.isCurrentAbsent(j) ? LegCalibrationFailure::NoMotorCurrent
                                             : LegCalibrationFailure::NoMotion, 0);
    }

    // After the verification nudge settles: record what it proved, where it
    // started, then always drive the full curl back. The settled count is the
    // evidence, spring-out included: a stored knee that stops at its 1 s cap
    // just short of 12 edges and springs out past them is a hall.
    void finishNudge(LegCalibrationPlant &plant)
    {
        const uint8_t j = joint_;
        if (!checkHallEvidence(plant, plant.nowMs()))
            return;
        const int32_t delta = plant.hallCount(j) - startHall_;
        const int endSnap = plant.potSnapshot(j);
        bool classified = false;
        if (!classifyAtRest(plant, delta, endSnap, classified))
            return;

        const JointSensor kind = plant.sensor(j);
        bool evidence = false;
        int32_t endPos = 0;
        int32_t travel = 0;
        if (kind == JointSensor::Pot)
        {
            if (!potMotion(plant, delta, endSnap, evidence))
                return;
            endPos = endSnap;
            travel = static_cast<int32_t>(endSnap) - startSnap_;
            preNudgePos_ = startSnap_;
            nudgeEdges_ = 0;
        }
        else
        {
            // Without quadrature the counts carry no direction: no edges.
            nudgeEdges_ = plant.hasQuadratureHall() ? hallDriveEdges(sign_, delta) : 0;
            evidence = kind == JointSensor::Hall && nudgeEdges_ >= CAL_HALL_DETECT_EDGES;
            endPos = plant.position(j);
            travel = ACTUATOR_HALL_EXTEND_COUNT_SIGN * delta;
            preNudgePos_ = endPos - travel;
        }
        nudgeKind_ = evidence ? kind : JointSensor::Unknown;
        nudgeCurrentAbsent_ = plant.isCurrentAbsent(j);
        printEnd(plant, kind, endPos, travel, endSnap);
        beginDrive(plant, LegCalibrationMoveStage::Return, true);
    }

    // Stage the end reached: min with -PWM, max with +PWM.
    void stageEnd(int32_t endPos)
    {
        if (sign_ > 0)
            stagedMax_[joint_] = static_cast<int>(endPos);
        else
            stagedMin_[joint_] = static_cast<int>(endPos);
    }

    // SWEEP checks on a hall knee's stroke S and its canonical limits [-S, 0]
    // (the curl end is 0). Returns false when the pass failed.
    bool checkAndStageStroke(LegCalibrationPlant &plant, int32_t uncurlEnd, int32_t endPos,
                             int32_t shortBy)
    {
        // Distance from the uncurl end to the home in force after this step. It
        // is at least the distance this curl left the uncurl end, which the
        // caller checked against CAL_HALL_MIN_STROKE_EDGES: only the top can fail.
        const int32_t stroke = magnitude(uncurlEnd - (shortBy < 0 ? endPos : 0));
        if (stroke > CAL_HALL_MAX_STROKE_EDGES)
        {
            fail(plant, LegCalibrationFailure::StrokeOutOfRange, stroke);
            return false;
        }
        const int32_t saved = plant.savedStroke(joint_);
        if (saved > 0 && stroke * 100 < saved * 85)
        {
            fail(plant, LegCalibrationFailure::StrokeShort, stroke, static_cast<int>(saved));
            return false;
        }
        const int32_t warnBy = saved / 10 > 30 ? saved / 10 : 30;
        if (saved > 0 && magnitude(stroke - saved) > warnBy)
        {
            CalibrationTextSink &out = plant.out();
            printWarnPrefix(plant, out);
            out.printFlash(KRABBY_FLASH_LITERAL(" stroke "));
            out.printNumber(saved);
            out.printFlash(KRABBY_FLASH_LITERAL(" -> "));
            out.printNumber(stroke);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges"));
            out.endLine();
        }
        stagedMin_[joint_] = -static_cast<int>(stroke);
        stagedMax_[joint_] = 0;
        return true;
    }

    // joint_ is at its curl end: optionally home it, mark it curled, next step.
    void accept(LegCalibrationPlant &plant, int32_t endPos, bool home)
    {
        const uint8_t j = joint_;
        if (home)
        {
            plant.setHome(j);
            CalibrationTextSink &out = plant.out();
            printJointLine(plant, out, KRABBY_FLASH_LITERAL("HOME "));
            out.printFlash(KRABBY_FLASH_LITERAL(" count="));
            out.printNumber(plant.hallCount(j));
            out.endLine();
        }
        curledMask_ |= static_cast<uint8_t>(1U << j);
        if (plant.sensor(j) == JointSensor::Pot)
        {
            curlEndKnownMask_ |= static_cast<uint8_t>(1U << j);
            curlEnd_[j] = static_cast<int>(endPos);
        }
        if (!(pass_ == CAL_PASS_SWEEP && legCalibrationStep(pass_, step_).completesJoint))
            stageEnd(endPos);
        advance(plant);
    }

    void advance(LegCalibrationPlant &plant)
    {
        ++step_;
        repeated_ = false;
        if (step_ < legCalibrationStepCount(pass_))
        {
            beginMove(plant);
            return;
        }
        finishPass(plant);
    }

    void reportJoint(LegCalibrationPlant &plant, JointSensor kind)
    {
        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("JOINT "));
        out.printFlash(KRABBY_FLASH_LITERAL(" min="));
        out.printNumber(stagedMin_[joint_]);
        out.printFlash(KRABBY_FLASH_LITERAL(" max="));
        out.printNumber(stagedMax_[joint_]);
        if (kind == JointSensor::Hall)
        {
            out.printFlash(isLegCalibrationHip(joint_)
                ? KRABBY_FLASH_LITERAL(" edges (fixed sweep)")
                : KRABBY_FLASH_LITERAL(" edges"));
        }
        out.endLine();
    }

    void finishPass(LegCalibrationPlant &plant)
    {
        CalibrationTextSink &out = plant.out();
        if (pass_ == CAL_PASS_SEEK)
        {
            printPrefix(plant, out);
            out.printFlash(KRABBY_FLASH_LITERAL("PASS_DONE SEEK"));
            out.endLine();
            if ((passes_ & CAL_PASS_SWEEP) != 0)
            {
                beginPass(plant, CAL_PASS_SWEEP);
                return;
            }
            finish(plant);
            return;
        }

        // Only a completed SWEEP changes limits: hips and knees take the staged
        // readings, yaws keep theirs, and the board's record is saved as a whole.
        for (uint8_t joint = 0; joint < LEG_CALIBRATION_JOINT_COUNT; ++joint)
        {
            if (joint % LEG_CALIBRATION_JOINTS_PER_LEG != 0)
                plant.storeLimits(joint, stagedMin_[joint], stagedMax_[joint]);
        }
        plant.saveLimits();
        printPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL("PASS_DONE SWEEP saved"));
        out.endLine();
        finish(plant);
    }

    // A failed joint is not trusted: its hall home is cleared (an abort keeps it).
    void fail(LegCalibrationPlant &plant, LegCalibrationFailure failure, int32_t value,
              int value2 = 0)
    {
        plant.stopAll();
        warnIfNudgedOut(plant);
        failure_ = failure;
        failJoint_ = plant.jointName(joint_);
        failKind_ = plant.sensor(joint_);
        failCurl_ = curl_;
        failValue_ = value;
        failValue2_ = value2;
        outcome_ = LegCalibrationOutcome::Failed;
        if (failKind_ == JointSensor::Hall)
            plant.clearHome(joint_);
        CalibrationTextSink &out = plant.out();
        printPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL("FAIL "));
        printFailureReason(out);
        out.endLine();
    }

    void finish(LegCalibrationPlant &plant)
    {
        plant.stopAll();
        outcome_ = LegCalibrationOutcome::Completed;
    }

    LegCalibrationOutcome outcome_;
    LegCalibrationFailure failure_;
    LegCalibrationMovePhase phase_;
    LegCalibrationMoveStage stage_;
    uint8_t passes_;
    uint8_t pass_;
    uint8_t step_;
    uint8_t pwm_;
    uint8_t options_;      // CAL_OPT_* bits of this run
    uint8_t joint_;
    bool curl_;
    bool repeated_;        // this step was repeated after a late Pot classification
    bool nudgeCurrentAbsent_;
    // The hall hip sweep stopped at its target (the settled reading may relax
    // an edge or two back below it).
    bool sweepReached_;
    bool detecting_;          // this drive ends by the end detector
    CalibrationEnd endKind_;  // what the detector decided for this drive
    uint16_t nudgeDriveMs_;   // drive time of the last nudge out (its Return's minimum)
    int8_t sign_;
    uint8_t rampPwm_;
    uint8_t curledMask_;
    // Bit j set: curlEnd_[j] holds where Pot joint j's curl end read on an
    // accepted curl (kept across start() calls; the pot is absolute).
    uint8_t curlEndKnownMask_;
    uint8_t reportedMask_;  // SENSOR line printed for the joint in this run
    // Kind with which the last nudge showed motion; Unknown: no evidence.
    JointSensor nudgeKind_;
    JointSensor failKind_;  // kind of the failed joint (units of the reason)
    bool failCurl_;
    int32_t startHall_;     // raw count at drive start
    int startSnap_;         // pot snapshot at drive start
    int startPot_;          // smoothed pot at drive start (Pot nudge stop)
    int windowStartPos_;
    int32_t preNudgePos_;   // joint position where the nudge started
    int32_t nudgeEdges_;    // hall drive edges of the last nudge (0 for Pot)
    // Failure values (edges or counts; SweepShort: edges short of the sweep);
    // Aborted: the cause; value2: StrokeShort saved stroke.
    int32_t failValue_;
    int failValue2_;
    uint32_t moveStartMs_;
    uint32_t phaseStartMs_;
    CalibrationEndDetector end_;
    const char *failJoint_;
    int stagedMin_[LEG_CALIBRATION_JOINT_COUNT];
    int stagedMax_[LEG_CALIBRATION_JOINT_COUNT];
    int curlEnd_[LEG_CALIBRATION_JOINT_COUNT];
};

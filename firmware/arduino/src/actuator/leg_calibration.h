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
// SEEK pass, for leg A then leg B: knee -> curl end; hip -> curl end; knee ->
//   curl end again (R3: a knee curled first while its hip rested extended may
//   have stopped on the body; curled again with its hip curled it is proven,
//   and a deeper end moves its home, "home moved").
// SWEEP pass, for leg A then leg B, table-safe (a hall hip, the robot's legs;
//   calibration_constants.h CAL_KNEE_PARK_EDGES): knee -> curl end; hip ->
//   curl end (home 0); knee -> uncurl end (its switch); knee -> park (by count
//   to -CAL_KNEE_PARK_EDGES, then checked inside its band); hip -> sweep (by
//   count to CAL_HIP_SWEEP_EDGES with the knee held parked: any end before it,
//   or the knee leaving its band, is an obstacle, and the hip retracts to its
//   curl switch and the pass fails, with no retry); hip -> curl end (it must
//   re-home within CAL_HIP_REHOME_TOLERANCE_EDGES of 0, else lost counts);
//   knee -> curl end. The hip saves [0, CAL_HIP_STROKE_EDGES] (its extend
//   switch derived from FL's measured stroke, not switch-verified) and the knee
//   [-S, 0]. The hip's extend switch is never sought, except by the
//   opt-in hollow-stand sweep (CAL_OPT_HOLLOW_STAND), which parks the knee at
//   -CAL_KNEE_HOLLOW_PARK_EDGES and sweeps the hip on to its switch.
// A pot hip (builds without quadrature halls) keeps the earlier order: knee ->
//   curl end; hip -> curl end; hip -> uncurl end; hip -> curl end; knee ->
//   uncurl end; knee -> curl end.
// A hip moves only with its knee proven curled or proven parked in this run,
// and a table-safe hip sweep only with its knee parked; a knee leaves its curl
// end (uncurl, park) or curls back from its park only with its hip proven
// curled (legCalibrationPartnerReady; the knee nudge, CAL_OPT_KNEE_NUDGE and on
// by default, is the one exception).
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
// detector (calibration_end_detector.h) decides: a limit switch (count
// stopped, current dropped to the switch-open level: "by=switch"), an
// extending rod's mechanical stop (speed collapsed, current rose: "by=stop"),
// a stall with neither signature ("by=stuck": the pass fails "stuck
// mid-stroke"; on a drive into a limit switch that includes every stall
// without the switch-open signature, however short the drive), or, for an
// extend drive that never ran or armed only in its own end approach, or a
// switch-end drive that shows the switch-open signature, a plain stall. A
// park or sweep also stops at its count. Otherwise (proven pot, builds without quadrature),
// after CAL_START_GRACE_MS every CAL_STALL_WINDOW_MS window must advance the
// joint in the driven direction (a proven pot by CAL_STALL_MIN_PROGRESS counts
// of its smoothed reading, any other joint by CAL_HALL_STALL_MIN_EDGES edges);
// the first window that does not is the end of travel. Either way: stop, wait
// CAL_SETTLE_MS, read the end. A move still running CAL_MOVE_TIMEOUT_MS after
// it started fails.
//
// A curl that moved (CAL_HALL_CURL_EVIDENCE_EDGES edges or CAL_MIN_TRAVEL_COUNTS
// counts) and then stopped at its end is at its curl end (a hip's: at its limit
// switch, shown by the switch-open current; a knee's at its stop: after a boot
// an obstacle looks the same). A curl with no motion is accepted only at a curl
// end proven earlier (a hall home from this boot, kept through a failed run as
// below, a pot end read on an earlier accepted curl, or a measured pot limit).
// Otherwise a hip gets a verification nudge (out at no more than
// CAL_HIP_NUDGE_MAX_PWM, then a full curl back) and a knee fails (or, with
// CAL_OPT_KNEE_NUDGE, gets a shorter nudge at no more than
// CAL_KNEE_NUDGE_MAX_PWM; a hall knee's curl back then needs only
// CAL_KNEE_RETURN_MIN_EDGES of motion, since a knee parked deeper than the
// calibration PWM reaches comes back only a few edges). Ending more than the
// curl-end tolerance short of a curl end proven earlier fails, with or without
// motion; ending deeper is accepted and moves a hall home. A failed hall
// joint's home is cleared, but kept as the reference of its later curls in
// this boot (V1): a rerun that ends far short of it fails instead of homing
// there. A failed pass stops before the partner joint moves. A failure or
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
    SweepShort,      // the hip sweep ended before its target: obstacle (retracted)
    StuckMidStroke,
    PartnerNotCurled,
    Aborted,
    LostCounts,      // a hip switch (curl or extend) found away from where its counts put it
    KneeOutOfBand,   // a parked knee outside its band (after its park, or during the hip sweep)
    ParkShort,       // the knee's park drive ended before its target: obstacle
    NotRetracted,    // after an obstacle the hip did not reach its curl switch
    TableSafeNeedsHall,  // a hall hip whose knee is not a hall encoder: no table-safe park
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
    Retract,   // a hip's curl back to its switch after an obstacle (the pass then fails)
};

// What a step's drive does: run to an end (Seek), or stop at a count (Park: a
// knee to its table-safe park; Sweep: a hip's table-safe sweep).
enum class LegCalibrationAction : uint8_t
{
    Seek,
    Park,
    Sweep,
};

static constexpr uint8_t LEG_CALIBRATION_JOINT_COUNT = 6;
static constexpr uint8_t LEG_CALIBRATION_JOINTS_PER_LEG = 3;
static constexpr uint8_t LEG_CALIBRATION_HIP_OFFSET = 1;
static constexpr uint8_t LEG_CALIBRATION_KNEE_OFFSET = 2;
static constexpr uint8_t LEG_CALIBRATION_SEEK_STEPS_PER_LEG = 3;
// Table-safe SWEEP of a hall-hip leg, and the earlier SWEEP of a pot-hip leg.
static constexpr uint8_t LEG_CALIBRATION_SWEEP_STEPS_PER_LEG = 7;
static constexpr uint8_t LEG_CALIBRATION_POT_SWEEP_STEPS_PER_LEG = 6;
static constexpr uint8_t LEG_CALIBRATION_LEG_COUNT = 2;

inline bool isLegCalibrationHip(uint8_t joint)
{
    return joint % LEG_CALIBRATION_JOINTS_PER_LEG == LEG_CALIBRATION_HIP_OFFSET;
}

// The knee of joint's leg (a hip's partner).
inline uint8_t legCalibrationKneeOf(uint8_t joint)
{
    return static_cast<uint8_t>(joint - joint % LEG_CALIBRATION_JOINTS_PER_LEG + LEG_CALIBRATION_KNEE_OFFSET);
}

// The park of a knee in the hall frame ([-S, 0]): its target and band, in
// edges (negative: uncurled from the curl stop). hollow: the hollow-stand park.
inline int legCalibrationParkTarget(bool hollow)
{
    return -(hollow ? CAL_KNEE_HOLLOW_PARK_EDGES : CAL_KNEE_PARK_EDGES);
}

inline bool isLegCalibrationParkInBand(int32_t position, bool hollow)
{
    const int32_t out = -position;
    return hollow ? out >= CAL_KNEE_HOLLOW_BAND_MIN_EDGES && out <= CAL_KNEE_HOLLOW_BAND_MAX_EDGES
                  : out >= CAL_KNEE_PARK_BAND_MIN_EDGES && out <= CAL_KNEE_PARK_BAND_MAX_EDGES;
}

// PWM sign that moves joint toward its curl end (curl) or away from it.
inline int legCalibrationPwmSign(uint8_t joint, bool curl)
{
    const int curlSign = isLegCalibrationHip(joint)
        ? CAL_HIP_CURL_PWM_SIGN
        : CAL_KNEE_CURL_PWM_SIGN;
    return curl ? curlSign : -curlSign;
}

// Whether joint may be driven (curl or not; isNudge: a verification nudge out;
// action: a Park or Sweep step's drive, else Seek) given the joints proven
// curled (curledMask) and the knees proven parked (parkedMask) in this run, and
// the options. A hip: its table-safe sweep only with its knee parked; any
// other uncurl (a pot hip's) only with its knee curled; a curl, return,
// retract or nudge with its knee curled or parked. A knee: a curl freely
// (unless it is parked); its park, its uncurl, and a curl back from its park
// only with its hip curled; a nudge with CAL_OPT_KNEE_NUDGE.
inline bool legCalibrationPartnerReady(
    uint8_t curledMask,
    uint8_t parkedMask,
    uint8_t joint,
    bool curl,
    bool isNudge,
    LegCalibrationAction action,
    uint8_t options)
{
    const uint8_t base = joint - joint % LEG_CALIBRATION_JOINTS_PER_LEG;
    const uint8_t kneeBit = static_cast<uint8_t>(1U << (base + LEG_CALIBRATION_KNEE_OFFSET));
    if (isLegCalibrationHip(joint))
    {
        const bool kneeCurled = (curledMask & kneeBit) != 0;
        const bool kneeParked = (parkedMask & kneeBit) != 0;
        if (action == LegCalibrationAction::Sweep)
            return kneeParked;
        if (!curl && !isNudge)
            return kneeCurled;
        return kneeCurled || kneeParked;
    }
    if (curl && action != LegCalibrationAction::Park && (parkedMask & kneeBit) == 0)
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
            // value: edges short of the sweep's target; value2: the target.
            out.printFlash(KRABBY_FLASH_LITERAL(" stopped "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges short of its "));
            out.printNumber(value2);
            out.printFlash(KRABBY_FLASH_LITERAL("-edge sweep: obstacle? check its home"));
            return;
        case LegCalibrationFailure::LostCounts:
            // value: where the switch was found; value2: where it was expected.
            out.printFlash(KRABBY_FLASH_LITERAL(" lost counts: switch at "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges, expected "));
            out.printNumber(value2);
            return;
        case LegCalibrationFailure::KneeOutOfBand:
            // value: the knee's position; value2: 1 for the hollow-stand band.
            out.printFlash(KRABBY_FLASH_LITERAL(" at "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges, outside its park band "));
            out.printNumber(-(value2 != 0 ? CAL_KNEE_HOLLOW_BAND_MAX_EDGES : CAL_KNEE_PARK_BAND_MAX_EDGES));
            out.printFlash(KRABBY_FLASH_LITERAL(".."));
            out.printNumber(-(value2 != 0 ? CAL_KNEE_HOLLOW_BAND_MIN_EDGES : CAL_KNEE_PARK_BAND_MIN_EDGES));
            return;
        case LegCalibrationFailure::ParkShort:
            // value: edges short of the park; value2: the park target.
            out.printFlash(KRABBY_FLASH_LITERAL(" stopped "));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL(" edges short of its park at "));
            out.printNumber(value2);
            out.printFlash(KRABBY_FLASH_LITERAL(": obstacle?"));
            return;
        case LegCalibrationFailure::NotRetracted:
            out.printFlash(KRABBY_FLASH_LITERAL(" not back at its curl switch (hall="));
            out.printNumber(value);
            out.printFlash(KRABBY_FLASH_LITERAL("): jog it home"));
            return;
        case LegCalibrationFailure::TableSafeNeedsHall:
            out.printFlash(KRABBY_FLASH_LITERAL(" and its knee must both be hall encoders"));
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

// Steps per leg of `pass`; tableSafe (SWEEP): the leg's hip is a hall hip.
inline uint8_t legCalibrationLegStepCount(uint8_t pass, bool tableSafe)
{
    if (pass == CAL_PASS_SEEK)
        return LEG_CALIBRATION_SEEK_STEPS_PER_LEG;
    return tableSafe ? LEG_CALIBRATION_SWEEP_STEPS_PER_LEG : LEG_CALIBRATION_POT_SWEEP_STEPS_PER_LEG;
}

struct LegCalibrationStep
{
    uint8_t joint;   // within the leg: LEG_CALIBRATION_HIP_OFFSET or _KNEE_OFFSET
    bool curl;
    // SWEEP only: this move completes the joint's uncurl-and-back measurement.
    bool completesJoint;
    LegCalibrationAction action;
};

// Step `legStep` (0-based) of one leg in `pass` (CAL_PASS_SEEK or
// CAL_PASS_SWEEP); tableSafe (SWEEP): the table-safe order of a hall-hip leg.
inline LegCalibrationStep legCalibrationStep(uint8_t pass, uint8_t legStep, bool tableSafe)
{
    static constexpr uint8_t H = LEG_CALIBRATION_HIP_OFFSET;
    static constexpr uint8_t K = LEG_CALIBRATION_KNEE_OFFSET;
    LegCalibrationStep result;
    result.completesJoint = false;
    result.action = LegCalibrationAction::Seek;
    result.curl = true;
    if (pass == CAL_PASS_SEEK)
    {
        // knee curl, hip curl, knee curl
        result.joint = legStep == 1 ? H : K;
        return result;
    }
    if (tableSafe)
    {
        // knee curl, hip curl, knee uncurl, knee park, hip sweep, hip curl, knee curl
        static const uint8_t kJoint[LEG_CALIBRATION_SWEEP_STEPS_PER_LEG] = {K, H, K, K, H, H, K};
        result.joint = kJoint[legStep < LEG_CALIBRATION_SWEEP_STEPS_PER_LEG ? legStep : 0];
        result.curl = legStep != 2 && legStep != 4;
        result.completesJoint = legStep == 5 || legStep == 6;
        result.action = legStep == 3 ? LegCalibrationAction::Park
            : legStep == 4           ? LegCalibrationAction::Sweep
                                     : LegCalibrationAction::Seek;
        return result;
    }
    // knee curl, hip curl, hip uncurl, hip curl, knee uncurl, knee curl
    const bool isHip = legStep >= 1 && legStep <= 3;
    result.joint = isHip ? H : K;
    result.curl = legStep != 2 && legStep != 4;
    result.completesJoint = legStep == 3 || legStep == 5;
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
        leg_ = 0;
        legStep_ = 0;
        tableSafe_ = false;
        action_ = LegCalibrationAction::Seek;
        parkedMask_ = 0;
        keptMask_ = 0;
        verifiedMask_ = 0;
        pendingFailure_ = LegCalibrationFailure::None;
        pendingValue_ = 0;
        pendingValue2_ = 0;
        pendingJoint_ = 0;
        kneeLeft_ = false;
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
            keptHome_[joint] = 0;
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
        parkedMask_ = 0;
        verifiedMask_ = 0;
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
            // A park or a table-safe sweep stops at its count. During the
            // sweep the knee must stay inside its park band.
            if (stage_ == LegCalibrationMoveStage::Normal && action_ != LegCalibrationAction::Seek)
            {
                const bool hollow = (options_ & CAL_OPT_HOLLOW_STAND) != 0;
                const int32_t x = plant.position(joint_);
                const bool reached = action_ == LegCalibrationAction::Sweep
                    ? x >= sweepTarget()
                    : x >= legCalibrationParkTarget(hollow);
                if (reached)
                {
                    sweepReached_ = true;
                    stopToSettle(plant, now);
                    return;
                }
                if (action_ == LegCalibrationAction::Sweep &&
                    !isLegCalibrationParkInBand(plant.position(legCalibrationKneeOf(joint_)), hollow))
                {
                    kneeLeft_ = true;
                    stopToSettle(plant, now);
                    return;
                }
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
        warnIfLeftOut(plant);
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
    // Current pass bit (CAL_PASS_SEEK / CAL_PASS_SWEEP), the steps begun in it
    // before the current one, and the current leg (0 = A) and its step.
    uint8_t currentPass() const { return pass_; }
    uint8_t currentStep() const { return step_; }
    uint8_t currentLeg() const { return leg_; }
    uint8_t currentLegStep() const { return legStep_; }
    // The current leg's SWEEP runs the table-safe order (its hip is a hall hip).
    bool isTableSafeLeg() const { return tableSafe_; }
    // What the current (or last) step's drive does.
    LegCalibrationAction currentAction() const { return action_; }
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
    // Bit j set: knee j was parked (inside its band) in this run and has not
    // been driven since.
    uint8_t parkedMask() const { return parkedMask_; }
    // Bit j set: hall joint j failed while homed this boot; keptHome(j) is the
    // raw count of that home, the reference of its later curls.
    uint8_t keptHomeMask() const { return keptMask_; }
    int32_t keptHome(uint8_t joint) const { return keptHome_[joint]; }
    // Bit j set: hip j's extend switch was found by the hollow-stand sweep.
    uint8_t switchVerifiedMask() const { return verifiedMask_; }
    // Staged end readings of the running pass: min = end reached with -PWM,
    // max = with +PWM; for a hall joint, canonical [0, CAL_HIP_STROKE_EDGES] (hip;
    // a hollow-stand sweep's extend switch instead) or [-S, 0] (knee) once it
    // completes.
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
        leg_ = 0;
        legStep_ = 0;
        tableSafe_ = false;
        CalibrationTextSink &out = plant.out();
        printPrefix(plant, out);
        out.printFlash(pass_ == CAL_PASS_SEEK
            ? KRABBY_FLASH_LITERAL("PASS SEEK pwm=")
            : KRABBY_FLASH_LITERAL("PASS SWEEP pwm="));
        out.printNumber(pwm_);
        out.endLine();
        beginMove(plant);
    }

    // The current step of the pass.
    LegCalibrationStep currentStepOf() const
    {
        return legCalibrationStep(pass_, legStep_, tableSafe_);
    }

    void beginMove(LegCalibrationPlant &plant)
    {
        const uint8_t base = static_cast<uint8_t>(leg_ * LEG_CALIBRATION_JOINTS_PER_LEG);
        // The SWEEP orders agree up to the hip's curl; from there a hall hip
        // takes the table-safe order (its knee must be a hall encoder too).
        if (pass_ == CAL_PASS_SWEEP && legStep_ == 2)
        {
            const uint8_t hip = static_cast<uint8_t>(base + LEG_CALIBRATION_HIP_OFFSET);
            tableSafe_ = plant.sensor(hip) == JointSensor::Hall;
            if (tableSafe_ && plant.sensor(static_cast<uint8_t>(base + LEG_CALIBRATION_KNEE_OFFSET)) != JointSensor::Hall)
            {
                joint_ = hip;
                fail(plant, LegCalibrationFailure::TableSafeNeedsHall, 0);
                return;
            }
        }
        const LegCalibrationStep step = currentStepOf();
        joint_ = static_cast<uint8_t>(base + step.joint);
        action_ = step.action;
        beginDrive(plant, LegCalibrationMoveStage::Normal, step.curl);
    }

    // The table-safe sweep's stop count: CAL_HIP_SWEEP_EDGES, or for the
    // hollow-stand sweep the furthest its extend switch may be.
    int32_t sweepTarget() const
    {
        return (options_ & CAL_OPT_HOLLOW_STAND) != 0
            ? CAL_HIP_STROKE_EDGES + CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES
            : CAL_HIP_SWEEP_EDGES;
    }

    // Every drive of joint_ starts here: a step's move, a nudge, a return or a
    // retract.
    void beginDrive(LegCalibrationPlant &plant, LegCalibrationMoveStage stage, bool curl)
    {
        curl_ = curl;
        sign_ = static_cast<int8_t>(legCalibrationPwmSign(joint_, curl_));
        plant.stopAll();
        // The partner gate cannot fail through the step table; it guards the
        // table. The pot-open probe applies to proven Pot joints only.
        const LegCalibrationAction action = stage == LegCalibrationMoveStage::Normal
            ? action_
            : LegCalibrationAction::Seek;
        const LegCalibrationFailure gate =
            !legCalibrationPartnerReady(curledMask_, parkedMask_, joint_, curl_,
                                        stage == LegCalibrationMoveStage::NudgeOut, action, options_)
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
        parkedMask_ &= static_cast<uint8_t>(~(1U << joint_));
        sweepReached_ = false;
        kneeLeft_ = false;
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
        // A hip's extended end is a limit switch too (R2).
        end_.begin(moveStartMs_, sign_ < 0, !curl_, minStill, isLegCalibrationHip(joint_) && sign_ > 0);

        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("MOVE "));
        out.printChar(' ');
        if (stage == LegCalibrationMoveStage::NudgeOut)
            out.printFlash(KRABBY_FLASH_LITERAL("nudge"));
        else
            printLegCalibrationDirection(out, curl_);
        if (stage == LegCalibrationMoveStage::Retract)
            out.printFlash(KRABBY_FLASH_LITERAL(" retract"));
        else if (stage == LegCalibrationMoveStage::Normal && action_ == LegCalibrationAction::Park)
            out.printFlash(KRABBY_FLASH_LITERAL(" park"));
        else if (stage == LegCalibrationMoveStage::Normal && action_ == LegCalibrationAction::Sweep)
            out.printFlash(KRABBY_FLASH_LITERAL(" sweep"));
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
    // more than CAL_KNEE_NUDGE_MAX_PWM for a knee nudge or CAL_HIP_NUDGE_MAX_PWM
    // for a hip nudge.
    uint8_t movePwm() const
    {
        if (stage_ == LegCalibrationMoveStage::NudgeOut)
        {
            const int cap = isLegCalibrationHip(joint_) ? CAL_HIP_NUDGE_MAX_PWM : CAL_KNEE_NUDGE_MAX_PWM;
            if (pwm_ > cap)
                return static_cast<uint8_t>(cap);
        }
        return pwm_;
    }

    // The verification nudge has gone far enough, or reached its time cap. A
    // knee with no hall feedback runs to its 1 s cap, a hip with no hall edge
    // stops at CAL_HIP_NUDGE_BLIND_MS (calibration_constants.h).
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
            const int32_t delta = plant.hallCount(joint_) - startHall_;
            const int32_t edges = hallDriveEdges(sign_, delta);
            if (edges >= (hip ? CAL_HIP_NUDGE_EDGES : CAL_KNEE_NUDGE_EDGES))
                return true;
            if (hip && magnitude(delta) <= CAL_POT_DETECT_MAX_EDGES && now - moveStartMs_ >= CAL_HIP_NUDGE_BLIND_MS)
                return true;
        }
        return now - moveStartMs_ >= (hip ? CAL_HIP_NUDGE_MAX_MS : CAL_KNEE_NUDGE_MAX_MS);
    }

    // A run that ends while a hip is out on its table-safe sweep (or retract)
    // or a knee is out on its park leaves the leg off its curl ends:
    // "CAL <ROLE> WARN <j> left out at <x> edges: jog it to its curl end".
    void warnIfLeftOut(LegCalibrationPlant &plant)
    {
        const bool out = stage_ == LegCalibrationMoveStage::Retract ||
                         (stage_ == LegCalibrationMoveStage::Normal && action_ != LegCalibrationAction::Seek);
        if (!out)
            return;
        CalibrationTextSink &text = plant.out();
        printWarnPrefix(plant, text);
        text.printFlash(KRABBY_FLASH_LITERAL(" left out at "));
        text.printNumber(plant.position(joint_));
        text.printFlash(KRABBY_FLASH_LITERAL(" edges: jog it to its curl end"));
        text.endLine();
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
        // tenths of an IS count, then on a drive into a limit switch (a
        // retract, a hip extend) " floor=<floor>"; a plain stall prints
        // nothing, or on such a drive " open=<current>/<floor>" (it showed the
        // switch-open signature).
        const bool switchEnd = sign_ < 0 || isLegCalibrationHip(joint_);
        if (endKind_ == CalibrationEnd::Switch || endKind_ == CalibrationEnd::Stop ||
            endKind_ == CalibrationEnd::Stuck)
        {
            out.printFlash(endKind_ == CalibrationEnd::Switch ? KRABBY_FLASH_LITERAL(" by=switch cur=")
                           : endKind_ == CalibrationEnd::Stop ? KRABBY_FLASH_LITERAL(" by=stop cur=")
                                                              : KRABBY_FLASH_LITERAL(" by=stuck cur="));
            out.printNumber(static_cast<long>(end_.endCurrent()));
            out.printChar('/');
            out.printNumber(static_cast<long>(end_.endReference()));
            if (switchEnd && end_.currentFloor() != 0xFFFFU)
            {
                out.printFlash(KRABBY_FLASH_LITERAL(" floor="));
                out.printNumber(static_cast<long>(end_.currentFloor()));
            }
        }
        else if (endKind_ == CalibrationEnd::Still && switchEnd && end_.currentFloor() != 0xFFFFU)
        {
            out.printFlash(KRABBY_FLASH_LITERAL(" open="));
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

        // A retract after an obstacle, a park and a table-safe sweep each
        // settle their own way.
        if (stage_ == LegCalibrationMoveStage::Retract)
        {
            finishRetract(plant, endPos);
            return;
        }
        if (stage_ == LegCalibrationMoveStage::Normal && action_ == LegCalibrationAction::Park)
        {
            finishPark(plant, endPos);
            return;
        }
        if (stage_ == LegCalibrationMoveStage::Normal && action_ == LegCalibrationAction::Sweep)
        {
            finishSweep(plant, endPos);
            return;
        }

        // A stall with no end signature (on a drive into a limit switch: no
        // switch-open current) is never an end: the pass stops before the
        // partner joint moves.
        if (endKind_ == CalibrationEnd::Stuck)
        {
            fail(plant, LegCalibrationFailure::StuckMidStroke, driveEdges);
            return;
        }

        // (3) Uncurl (a knee's, or a pot hip's): stage the end it reached.
        if (!curl_)
        {
            if (!motion)
            {
                fail(plant, LegCalibrationFailure::BarelyMoved,
                     kind == JointSensor::Pot ? magnitude(travel) : driveEdges);
                return;
            }
            stageEnd(endPos);
            advance(plant);
            return;
        }

        // (4) Curl: compare with the curl end proven earlier, if any: a hall
        // home from this boot, or one kept through a failed run (kept: the
        // joint is homed again where it is accepted).
        bool proven = false;
        bool kept = false;
        int32_t reference = 0;
        int32_t fromReference = endPos;
        if (kind == JointSensor::Hall)
        {
            proven = plant.isHomed(j);
            if (!proven && (keptMask_ & (1U << j)) != 0)
            {
                proven = true;
                kept = true;
                fromReference = static_cast<int32_t>(ACTUATOR_HALL_EXTEND_COUNT_SIGN) *
                                (plant.hallCount(j) - keptHome_[j]);
            }
        }
        else if (kind == JointSensor::Pot)
        {
            proven = (curlEndKnownMask_ & (1U << j)) != 0;
            reference = curlEnd_[j];
        }
        // > 0: stopped short of the reference; < 0: deeper.
        const int32_t shortBy = -sign_ * (fromReference - reference);
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
        if (pass_ == CAL_PASS_SWEEP && currentStepOf().completesJoint)
        {
            const int32_t uncurlEnd = sign_ > 0 ? stagedMin_[j] : stagedMax_[j];
            if (!proven || kept || sign_ * (endPos - uncurlEnd) < minimumLeave(kind))
            {
                fail(plant, LegCalibrationFailure::NotAtCurlEnd, endPos);
                return;
            }
            const bool hallHip = kind == JointSensor::Hall && isLegCalibrationHip(j);
            // A table-safe hip comes back to its curl switch: where its counts
            // put the home, or they were lost.
            if (hallHip && magnitude(shortBy) > CAL_HIP_REHOME_TOLERANCE_EDGES)
            {
                fail(plant, LegCalibrationFailure::LostCounts, endPos, 0);
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
            else if (hallHip)
            {
                // Its curl switch to its extend switch: found by a hollow-stand
                // sweep, else derived (CAL_HIP_STROKE_EDGES).
                stagedMin_[j] = 0;
                if ((verifiedMask_ & (1U << j)) == 0)
                    stagedMax_[j] = CAL_HIP_STROKE_EDGES;
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
            accept(plant, endPos, kind == JointSensor::Hall && (!proven || kept || shortBy < 0));
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
            accept(plant, endPos, kind == JointSensor::Hall && (kept || shortBy < 0));
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

    // The knee's park settled: it must have reached its target (any detector
    // end before it is an obstacle) and rest inside its band. "CAL <ROLE> PARK
    // <j> hall=<x> target=<t> (FL values)".
    void finishPark(LegCalibrationPlant &plant, int32_t endPos)
    {
        const bool hollow = (options_ & CAL_OPT_HOLLOW_STAND) != 0;
        const int target = legCalibrationParkTarget(hollow);
        if (!sweepReached_)
        {
            fail(plant, LegCalibrationFailure::ParkShort, target - endPos, target);
            return;
        }
        if (!isLegCalibrationParkInBand(endPos, hollow))
        {
            fail(plant, LegCalibrationFailure::KneeOutOfBand, endPos, hollow ? 1 : 0);
            return;
        }
        parkedMask_ |= static_cast<uint8_t>(1U << joint_);
        CalibrationTextSink &out = plant.out();
        printJointLine(plant, out, KRABBY_FLASH_LITERAL("PARK "));
        out.printFlash(KRABBY_FLASH_LITERAL(" hall="));
        out.printNumber(endPos);
        out.printFlash(KRABBY_FLASH_LITERAL(" target="));
        out.printNumber(target);
        out.printFlash(KRABBY_FLASH_LITERAL(" (FL values)"));
        out.endLine();
        advance(plant);
    }

    // The table-safe hip sweep settled. Reaching its count is the solid-stand
    // sweep's end (staged as the uncurl end); the hollow-stand sweep must find
    // its extend switch instead, within the tolerance of CAL_HIP_STROKE_EDGES.
    // Anything else (an obstacle's stop or stall, the knee leaving its band, an
    // extend switch found where the counts do not put it) retracts the hip to
    // its curl switch, and the pass then fails.
    void finishSweep(LegCalibrationPlant &plant, int32_t endPos)
    {
        const bool hollow = (options_ & CAL_OPT_HOLLOW_STAND) != 0;
        const int32_t target = sweepTarget();
        if (kneeLeft_)
        {
            const uint8_t knee = legCalibrationKneeOf(joint_);
            retractAfter(plant, LegCalibrationFailure::KneeOutOfBand, plant.position(knee), hollow ? 1 : 0, knee);
            return;
        }
        const bool atSwitch = endKind_ == CalibrationEnd::Switch || endKind_ == CalibrationEnd::Still;
        if (sweepReached_ && !hollow)
        {
            stageEnd(endPos);
            advance(plant);
            return;
        }
        if (sweepReached_ || atSwitch)
        {
            if (!sweepReached_ && hollow &&
                magnitude(endPos - CAL_HIP_STROKE_EDGES) <= CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES)
            {
                verifiedMask_ |= static_cast<uint8_t>(1U << joint_);
                stageEnd(endPos);
                advance(plant);
                return;
            }
            retractAfter(plant, LegCalibrationFailure::LostCounts, endPos, CAL_HIP_STROKE_EDGES, joint_);
            return;
        }
        retractAfter(plant, LegCalibrationFailure::SweepShort, target - endPos, static_cast<int>(target), joint_);
    }

    // "CAL <ROLE> WARN <j> stopped at <x> edges: retracting" and the hip's
    // curl back to its switch; the pass then fails with `failure` (see
    // finishRetract).
    void retractAfter(LegCalibrationPlant &plant, LegCalibrationFailure failure, int32_t value,
                      int value2, uint8_t failJoint)
    {
        pendingFailure_ = failure;
        pendingValue_ = value;
        pendingValue2_ = value2;
        pendingJoint_ = failJoint;
        CalibrationTextSink &out = plant.out();
        printWarnPrefix(plant, out);
        out.printFlash(KRABBY_FLASH_LITERAL(" stopped at "));
        out.printNumber(plant.position(joint_));
        out.printFlash(KRABBY_FLASH_LITERAL(" edges: retracting"));
        out.endLine();
        beginDrive(plant, LegCalibrationMoveStage::Retract, true);
    }

    // The retract settled: at its curl switch (the detector's switch or
    // switch-open stall, within the curl-end tolerance of its home) the pass
    // fails with the obstacle's failure; otherwise "not back at its curl
    // switch" (jog it home).
    void finishRetract(LegCalibrationPlant &plant, int32_t endPos)
    {
        const bool home = (endKind_ == CalibrationEnd::Switch || endKind_ == CalibrationEnd::Still) &&
                          magnitude(endPos) <= CAL_HALL_CURL_END_TOLERANCE_EDGES;
        if (!home)
        {
            fail(plant, LegCalibrationFailure::NotRetracted, endPos);
            return;
        }
        // Back at its curl switch: not left out.
        stage_ = LegCalibrationMoveStage::Normal;
        action_ = LegCalibrationAction::Seek;
        fail(plant, pendingFailure_, pendingValue_, pendingValue2_, pendingJoint_);
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
            keptMask_ &= static_cast<uint8_t>(~(1U << j));
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
        if (!(pass_ == CAL_PASS_SWEEP && currentStepOf().completesJoint))
            stageEnd(endPos);
        advance(plant);
    }

    void advance(LegCalibrationPlant &plant)
    {
        ++step_;
        ++legStep_;
        repeated_ = false;
        if (legStep_ >= legCalibrationLegStepCount(pass_, tableSafe_))
        {
            ++leg_;
            legStep_ = 0;
            tableSafe_ = false;
        }
        if (leg_ < LEG_CALIBRATION_LEG_COUNT)
        {
            beginMove(plant);
            return;
        }
        finishPass(plant);
    }

    // "CAL <ROLE> JOINT <j> min=<a> max=<b>[ edges...]": a hall hip's extend
    // end is derived from FL's measured stroke unless a hollow-stand sweep
    // found its switch.
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
            out.printFlash(!isLegCalibrationHip(joint_) ? KRABBY_FLASH_LITERAL(" edges")
                           : (verifiedMask_ & (1U << joint_)) != 0
                               ? KRABBY_FLASH_LITERAL(" edges (extend switch-verified)")
                               : KRABBY_FLASH_LITERAL(" edges (extend derived from FL, not switch-verified)"));
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

    // A failed joint (joint_, or failJoint) is not trusted: its hall home is
    // cleared (an abort keeps it). A home proven this boot is kept as the
    // reference of the joint's later curls (V1: a rerun that stops far short of
    // it fails instead of homing there), unless its counts were lost.
    void fail(LegCalibrationPlant &plant, LegCalibrationFailure failure, int32_t value,
              int value2 = 0)
    {
        fail(plant, failure, value, value2, joint_);
    }

    void fail(LegCalibrationPlant &plant, LegCalibrationFailure failure, int32_t value, int value2,
              uint8_t failJoint)
    {
        plant.stopAll();
        warnIfNudgedOut(plant);
        warnIfLeftOut(plant);
        failure_ = failure;
        failJoint_ = plant.jointName(failJoint);
        failKind_ = plant.sensor(failJoint);
        failCurl_ = curl_;
        failValue_ = value;
        failValue2_ = value2;
        outcome_ = LegCalibrationOutcome::Failed;
        if (failKind_ == JointSensor::Hall)
        {
            const uint8_t bit = static_cast<uint8_t>(1U << failJoint);
            if (failure == LegCalibrationFailure::LostCounts)
            {
                keptMask_ &= static_cast<uint8_t>(~bit);
            }
            else if (plant.isHomed(failJoint))
            {
                keptHome_[failJoint] = plant.hallCount(failJoint) -
                    static_cast<int32_t>(ACTUATOR_HALL_EXTEND_COUNT_SIGN) * plant.position(failJoint);
                keptMask_ |= bit;
            }
            plant.clearHome(failJoint);
        }
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
    uint8_t step_;      // steps begun in this pass before the current one
    uint8_t leg_;       // 0 = leg A, 1 = leg B
    uint8_t legStep_;   // the current leg's step (legCalibrationStep)
    bool tableSafe_;    // the current leg's SWEEP runs the table-safe order
    LegCalibrationAction action_;  // the current step's action
    uint8_t parkedMask_;  // knees parked in band this run, not driven since
    uint8_t keptMask_;    // hall joints whose home this boot is kept after a failure
    uint8_t verifiedMask_;  // hips whose extend switch the hollow-stand sweep found
    // The obstacle failure a retract ends in (finishRetract).
    LegCalibrationFailure pendingFailure_;
    int32_t pendingValue_;
    int pendingValue2_;
    uint8_t pendingJoint_;
    bool kneeLeft_;  // the sweep stopped because its knee left its park band
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
    int32_t keptHome_[LEG_CALIBRATION_JOINT_COUNT];  // raw count of a kept home
};

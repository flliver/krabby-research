#pragma once

// Records every actuator drive made by the host-sim firmware, per board, and
// checks the calibration safety invariants at every single pin write, not just
// at the end of a run. It also is the boards' plant: a rod per joint that moves
// with the board's signed drive between limit-switch ends and reports through a
// pot, a quadrature hall encoder, or nothing. A rod can behave as the robot's
// actuators were measured to on 2026-11-23 (robotLikeKnee/robotLikeHip): a
// curled knee needs time to break free and springs out after its nudge, a knee
// parked by a PWM 200 curl sits deeper than PWM 80 reaches, and a hip with its
// knee curled stops where the curled tibia meets the body (a violation).
// Every rod also drives its board's IS pin as the 2026-11-24 end-current runs
// measured: 0 with EN low, a running level while driven, the joint's
// switch-open level once driven back into its retracted end (the actuator's
// limit switch), and a rising level pressed into an extended end (the knee's
// compliant leg stop, its taper zone, or the body).
//
// The sim's three boards share one pin map; PinStateModel tags each write with
// the board whose firmware made it and gives each board its own analog inputs
// (host_multi_board.cpp ScopedHostBoard), so the timeline keeps one shadow pin
// bank per board. Hall counts are kept per board by hall_sim.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "EEPROM.h"
#include "hall_sim.hpp"
#include "pin_state.hpp"
#include "sim_clock.hpp"

namespace calibration_timeline
{

// The spec's numbers, written out rather than taken from calibration_constants.h
// so that a changed firmware constant fails these checks instead of moving them.
constexpr int kSpecRampStep = 5;
constexpr uint32_t kSpecRampIntervalMs = 20;
constexpr uint32_t kSpecSettleMs = 300;
constexpr uint32_t kSpecMoveTimeoutMs = 120000;
constexpr uint32_t kSpecFollowerSilenceMs = 135000;
// Confirmed on the robot: +PWM extends a rod; hips curl by retracting, knees by extending.
constexpr int kSpecHipCurlSign = -1;
constexpr int kSpecKneeCurlSign = +1;
// Hips and knees count down while extending (retract counts up).
constexpr int kSpecHallExtendCountSign = -1;
// With CAL_OPT_KNEE_NUDGE only: a knee may leave its curl end before its hip is
// proven curled for at most this long and this far, then curls straight back.
// The owner approved it as the shipped default (2026-11-23), so every run started
// through the C command uses it; a strict run passes options 0 explicitly.
constexpr bool kSpecDefaultKneeNudge = true;
constexpr uint32_t kSpecKneeNudgeMaxMs = 1000;
// The knee nudge drive stops at 12 edges (16 allows a tick of travel and the
// coast, also at PWM 128). A curled knee then springs out up to 9 more edges,
// which the owner accepted: the knee may end the nudge up to 25 edges out
// before it curls back.
constexpr double kSpecKneeNudgeMaxEdges = 16.0;
constexpr double kSpecKneeSpringMaxEdges = 9.0;
// A knee whose hall A is dead counts no edge, so only the 1 s cap stops its
// nudge. At the knee nudge's PWM 128 cap (CAL_KNEE_NUDGE_MAX_PWM follows
// CAL_DEFAULT_PWM), measured 2026-11-24: FLKL driven out of its curl end at 128
// had moved 72 edges 1.0 s after its drive started (a 0.64 s ramp; the
// firmware's is 0.52 s), the FR knees ran about 120-130 edges/s, and a knee
// coasted 3-9 edges after its stop. A blind knee that breaks free at once thus
// drives up to about 90 edges by the cap, 95 with its coast, and its spring-out
// (a curled knee's, up to 9) may add to that before it curls back (and fails:
// no feedback). At PWM 80 that was 36 / 45. Its hip is not proven curled: the
// knee nudge is the one move allowed before that.
constexpr double kSpecBlindKneeNudgeMaxEdges = 95.0;
// The hip verification nudge: out at most this long (ramp included).
constexpr uint32_t kSpecHipNudgeMaxMs = 2000;
// The end detector (2026-11-24): a drive into a limit switch stops at most this
// long after the rod reaches it, one into a mechanical stop at most this long
// after it, and a press of a joint already at its end at most this long.
constexpr uint32_t kSpecSwitchStopMaxMs = 250;
constexpr uint32_t kSpecMechanicalStopMaxMs = 400;
constexpr uint32_t kSpecPressMaxMs = 1100;
// A drive that reached its end within 1.2 s of its start (it started 0-130
// edges from it: no running phase, or one learned in the end approach) stops
// at most this long after reaching it, by a signature or by the plain stall
// (no edge for 1 s), and the run goes on (never "stuck mid-stroke").
constexpr uint32_t kSpecEarlyEndStopMaxMs = 1150;
// The owner's fixed hip sweep: a hall hip uncurls this many edges from its curl
// end, never to where its curled tibia meets the body.
constexpr double kSpecHipSweepEdges = 700.0;

constexpr int kBoards = 3;
constexpr int kJoints = 6;
constexpr int kPins = krabby::sim::PinStateModel::kMaxMegaPin + 1;
// Rod speed: pot counts per PWM count per millisecond (80 PWM -> 320 counts/s).
constexpr double kCountsPerPwmMs = 0.004;
// A joint "at its curl end" is within this many counts of it (or deeper).
constexpr double kAtEndTolerance = 2.0;
// EEPROM_ACTUATOR_CAL_ADDR / _SIZE: CalData v2 is the same 30-byte record at
// address 0 on AVR and host builds (int16 fields). Watch all of it.
constexpr int kCalDataBytes = 30;
// Hall B on the pot pin, as read at rest on the robot: low and high levels.
constexpr int kHallBLow = 19;
constexpr int kHallBHigh = 1016;
// An unplugged hall: both lines pulled up.
constexpr int kUnpluggedPotReading = 1020;
// Loaded knee taper: over the last kKneeTaperEdges edges before the curl end the
// rate falls to kKneeTaperFloorEdgesPerS, so a full-stroke approach stops
// shallower than a short one (fact 3: the PWM 80 end varies 3-24 edges).
constexpr double kKneeTaperEdges = 8.0;
constexpr double kKneeTaperFloorEdgesPerS = 2.0;
// Compliant knee curl end: deeper by kKneeComplianceEdges per kKneeCompliancePwm
// of drive (fact 3: a PWM 200 curl parks a knee 17 edges deeper than PWM 80 reaches).
constexpr double kKneeComplianceEdges = 17.0;
constexpr double kKneeCompliancePwm = 120.0;
// Travel after a drive stops runs at this rate (counts per ms).
constexpr double kAfterStopCountsPerMs = 0.1;
// Hall hips set up by setHallRods: a stroke longer than the reach with the knee
// curled, and the curled tibia meeting the body at 941 edges (fact 2: FLHL
// stopped at 941 with its knee 24 edges off its curl end, 1003 curled tight).
constexpr double kHallHipStrokeEdges = 1200.0;
constexpr double kBodyContactEdges = 941.0;

// Every board's sim actuators use the same pins: PWM_R, PWM_L, EN, POT, IS per slot.
struct JointPins
{
    int pwmR;
    int pwmL;
    int en;
    int pot;
    int is;
};
static const JointPins kJointPins[kJoints] = {
    {2, 3, 22, 0, 6}, {4, 5, 24, 1, 7}, {6, 7, 26, 2, 8},
    {8, 9, 23, 3, 9}, {10, 11, 25, 4, 10}, {12, 13, 27, 5, 11},
};

static const char *const kJointNames[kBoards][kJoints] = {
    {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"},
    {"RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL"},
    {"RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL"},
};

inline bool isYaw(int joint) { return joint % 3 == 0; }
inline bool isHip(int joint) { return joint % 3 == 1; }
inline bool isKnee(int joint) { return joint % 3 == 2; }
inline int curlSign(int joint) { return isHip(joint) ? kSpecHipCurlSign : kSpecKneeCurlSign; }

// What a rod reports through: a pot on the pot pin, a quadrature hall (A counts,
// B on the pot pin), or nothing (an unplugged hall: no edges, pot pin pulled up).
enum class RodSensor
{
    Pot,
    Hall,
    None,
};

// One rod. Its position is in pot counts; +drive extends it toward `high`.
struct Rod
{
    double position = 500.0;
    double low = 150.0;   // fully retracted (limit switch)
    double high = 850.0;  // fully extended (limit switch)
    double speedScale = 1.0;
    bool isStuck = false;
    // +1 / -1: the rod does not move under drive of that sign (a loaded actuator
    // below its dead zone, or held at a limit switch); 0: never.
    int stallSign = 0;
    bool isReversedPot = false;  // pot wired backwards: reads lower when extended
    bool isOpenPot = false;      // wiper disconnected
    bool isDeadMotor = false;    // never moves and draws no current (IS reads 0)
    bool motorReversed = false;  // motor leads swapped: +drive retracts

    RodSensor sensor = RodSensor::Pot;
    double edgesPerCount = 1.0;  // hall edges per pot count of travel
    bool hallReversed = false;   // A/B swapped: counts run against the travel
    bool deadA = false;          // no A edges; B still toggles
    bool deadB = false;          // B stuck: A edges alternate -1, +1
    bool kneeTaper = false;      // knee slows to ~2 edges/s over its last taperEdges edges
    // The taper's length in edges (kKneeTaperEdges; longer: a compliant end
    // approached over 0.3-0.5 s, its current rising, as the robot's at 120-150).
    double taperEdges = kKneeTaperEdges;
    bool kneeCompliance = false; // knee curl end deeper with more PWM
    bool startAtCurlEnd = false; // install() parks it at its curl end
    // With startAtCurlEnd: a knee parked by a curl at this PWM (compliance) sits
    // at extendedEnd(parkPwm); 0 parks it at `high`.
    int parkPwm = 0;
    // Speed multipliers per direction (on top of speedScale).
    double extendSpeed = 1.0;
    double retractSpeed = 1.0;
    // A knee driven out of its curl end does not move for its first breakawayMs.
    uint32_t breakawayMs = 0;
    // A knee whose uncurl drive started at its curl end springs out this many
    // edges further once the drive stops.
    double springOutEdges = 0.0;
    // Any rod keeps going this many edges in its drive's direction once it stops.
    double coastEdges = 0.0;
    // IS ADC while driven: running, driven into the retracted end (its limit
    // switch opens: the joint's switch-open level, FL hips 7-8, FRHL 11), and
    // pressed into an extended end (knee leg stop 13-24, 2026-11-24).
    double isRun = 12.0;
    double isSwitchOpen = 8.0;
    double isStall = 20.0;
    // A hip whose knee is at its curl end cannot extend more than this many
    // edges from its curl end: the curled tibia meets the body (0: never).
    double bodyContactEdges = 0.0;

    // Motion state (Timeline): drive time since this drive started, whether
    // it started at the curl end, travel still to come after a stop (counts),
    // and the last drive's sign and peak |PWM|.
    uint32_t drivenMs = 0;
    bool driveFromCurlEnd = false;
    double pendingTravel = 0.0;
    int lastDriveSign = 0;
    int lastDriveMagnitude = 0;

    // Travel in edges from position 0.
    double edgePosition() const { return position * edgesPerCount; }
    long edgeIndex() const { return static_cast<long>(std::floor(edgePosition())); }

    int potReading() const
    {
        if (sensor == RodSensor::None) return kUnpluggedPotReading;
        if (sensor == RodSensor::Hall)
        {
            // B is a half edge out of phase with A, so it can flip without an A edge.
            if (deadB) return kHallBLow;
            return (static_cast<long>(std::floor(edgePosition() + 0.5)) & 1) != 0 ? kHallBHigh : kHallBLow;
        }
        const double reading = isReversedPot ? low + high - position : position;
        return static_cast<int>(reading + 0.5);
    }
    // Physical curl end: hips curl by retracting, knees by extending.
    double curlEnd(int joint) const { return curlSign(joint) < 0 ? low : high; }
    // How far short of the curl end a tapered knee can stop and still be there.
    double atEndTolerance() const
    {
        return kAtEndTolerance + (kneeTaper ? taperEdges / edgesPerCount : 0.0);
    }
    // At its curl end or pushed deeper into a compliant one.
    bool isAtCurlEnd(int joint) const
    {
        const double shortBy = curlSign(joint) * (curlEnd(joint) - position);
        return shortBy <= atEndTolerance();
    }
    // Extended end under a drive of |pwm|: a compliant knee goes deeper.
    double extendedEnd(int joint, int pwmMagnitude) const
    {
        if (!kneeCompliance || !isKnee(joint)) return high;
        return high + kKneeComplianceEdges * pwmMagnitude / kKneeCompliancePwm / edgesPerCount;
    }
    // Edges out of its curl end (negative: pushed deeper into it).
    double edgesOut(int joint) const { return curlSign(joint) * (curlEnd(joint) - position) * edgesPerCount; }
};

// FLKL as measured on 2026-11-23: a 776-edge stroke with its hip curled; at PWM
// 80 about 48 edges/s uncurling (its nudge's rate; 34 mid-stroke) and 60
// curling; stored by a PWM 200 curl, 17 edges deeper than PWM 80 reaches; its
// nudge breaks free 0.77 s after it starts and it springs out 9 edges once
// stopped (fact 4: a 12-edge nudge ends 18-21 out); it coasts 1 edge.
inline void makeRobotLikeKnee(Rod &rod)
{
    rod.sensor = RodSensor::Hall;
    rod.edgesPerCount = 1.0;
    rod.low = rod.high - 776.0;
    rod.retractSpeed = 48.0 / 320.0;
    rod.extendSpeed = 60.0 / 320.0;
    rod.kneeTaper = true;
    rod.kneeCompliance = true;
    rod.startAtCurlEnd = true;
    rod.parkPwm = 200;
    rod.breakawayMs = 750;  // counted from the first ramp step, 20 ms into the move
    rod.springOutEdges = 9.0;
    rod.coastEdges = 1.0;
}

// FLHL as measured on 2026-11-23: about 45 edges/s uncurling and 25 curling at
// PWM 80, back exactly to its curl end (limit switch); with its knee curled the
// tibia meets the body at 941 edges, short of the hip's own end; stored
// curled; its nudge breaks free in about 0.45 s (24 edges at about 1.3 s); it
// coasts 2 edges.
inline void makeRobotLikeHip(Rod &rod)
{
    rod.sensor = RodSensor::Hall;
    rod.edgesPerCount = 1.0;
    rod.high = rod.low + kHallHipStrokeEdges;
    rod.extendSpeed = 45.0 / 320.0;
    rod.retractSpeed = 25.0 / 320.0;
    rod.bodyContactEdges = kBodyContactEdges;
    rod.startAtCurlEnd = true;
    rod.breakawayMs = 450;
    rod.coastEdges = 2.0;
}

// A contiguous drive of one joint in one direction.
struct Move
{
    int board;
    int joint;
    int sign;
    uint32_t startMs;
    uint32_t endMs;  // 0 while still running
    int peakMagnitude;
    // Bit (6 * board + joint) set: that hip or knee was at its curl end as the move began.
    uint32_t curledAtStart;
    double startPosition;  // rod position (counts) as the move began
    double endPosition;    // and as it stopped
};

struct DriveChange
{
    uint32_t ms;
    int board;
    int joint;
    int pwm;
};

class Timeline
{
public:
    Rod rods[kBoards][kJoints];
    // Quadrature hall counting per board (hallHwHasQuadrature, Rev 3); set before install().
    bool quadrature[kBoards] = {false, false, false};
    // The run uses CAL_OPT_KNEE_NUDGE: the bounded knee nudge is allowed. On for
    // the shipped default (every C command run); set false for a strict run
    // (startCalibration with options 0).
    bool kneeNudgeOption = kSpecDefaultKneeNudge;
    // |PWM| limit a calibration drive may use; 0 disables the PWM checks.
    int requestedPwm = 0;
    std::vector<Move> moves;
    std::vector<DriveChange> changes;
    std::vector<std::string> violations;
    // EEPROM record changes: board and time, in order.
    std::vector<std::pair<int, uint32_t>> eepromChanges;

    Timeline() = default;
    Timeline(const Timeline &) = delete;
    Timeline &operator=(const Timeline &) = delete;
    ~Timeline() { krabby::sim::pinState().setWriteObserver(nullptr); }

    // Bit (6 * board + joint) for every hip and knee now at its curl end.
    uint32_t curledMask() const
    {
        uint32_t mask = 0;
        for (int board = 0; board < kBoards; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
                if (!isYaw(joint) && rods[board][joint].isAtCurlEnd(joint))
                    mask |= 1UL << (kJoints * board + joint);
        return mask;
    }

    // Every hip and knee on boards [0, boardCount).
    static uint32_t allHipsAndKnees(int boardCount)
    {
        uint32_t mask = 0;
        for (int board = 0; board < boardCount; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
                if (!isYaw(joint)) mask |= 1UL << (kJoints * board + joint);
        return mask;
    }

    // Sets every hip and knee of boards [firstBoard, lastBoard] to a hall rod.
    // Hips get a stroke longer than their reach with the knee curled, where the
    // curled tibia meets the body (call after setting edgesPerCount).
    void setHallRods(int firstBoard, int lastBoard)
    {
        for (int board = firstBoard; board <= lastBoard; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
            {
                if (isYaw(joint)) continue;
                Rod &rod = rods[board][joint];
                rod.sensor = RodSensor::Hall;
                if (!isHip(joint)) continue;
                rod.high = rod.low + kHallHipStrokeEdges / rod.edgesPerCount;
                rod.bodyContactEdges = kBodyContactEdges;
            }
    }

    // The furthest each joint has been out of its curl end, in edges, since install().
    double furthestOutEdges(int board, int joint) const { return furthestOut_[board][joint]; }

    // Installs the pin observer, the boards' quadrature flags and the sensors;
    // call after krabby_host_reset() and before the firmware boots.
    void install()
    {
        std::memset(digital_, 0, sizeof(digital_));
        std::memset(duty_, 0, sizeof(duty_));
        std::memset(drive_, 0, sizeof(drive_));
        std::memset(lastChangeMs_, 0, sizeof(lastChangeMs_));
        for (int board = 0; board < kBoards; ++board)
        {
            krabby::sim::hallSimSetQuadrature(board, quadrature[board]);
            for (int joint = 0; joint < kJoints; ++joint)
            {
                Rod &rod = rods[board][joint];
                if (rod.startAtCurlEnd && !isYaw(joint))
                    rod.position = rod.parkPwm > 0 ? rod.extendedEnd(joint, rod.parkPwm) : rod.curlEnd(joint);
                rod.drivenMs = 0;
                rod.driveFromCurlEnd = false;
                rod.pendingTravel = 0.0;
                rod.lastDriveSign = 0;
                rod.lastDriveMagnitude = 0;
                lastMoveSign_[board][joint] = 0;
                openMove_[board][joint] = -1;
                reachedMs_[board][joint] = 0;
                pressAtStart_[board][joint] = false;
                edgeIndex_[board][joint] = rod.edgeIndex();
                deadBPhase_[board][joint] = false;
                furthestOut_[board][joint] = 0.0;
                bodyContactFlagged_[board][joint] = false;
            }
        }
        nudge_ = PendingNudge();
        krabby::sim::pinState().setWriteObserver(
            [this](const krabby::sim::PinWriteEvent &event) { onWrite(event); });
        publishSensors();
        snapshotEeprom();
    }

    // Any drive starting after this point is a violation, except a joint
    // explicitly allowed (a jog the test sends on purpose).
    void forbidDrives(int allowedBoard = -1, int allowedJoint = -1)
    {
        isFrozen_ = true;
        allowedBoard_ = allowedBoard;
        allowedJoint_ = allowedJoint;
    }

    // Operator jogs (J, B) between runs: exempt from the partner, ramp and PWM
    // checks; one actuator at a time and never a yaw still apply.
    void setOperatorJogs(bool on) { isOperatorJog_ = on; }

    // Moves every rod by its board's present drive for dtMs, then republishes the sensors.
    void advancePlant(uint32_t dtMs)
    {
        for (int board = 0; board < kBoards; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
            {
                moveRod(board, joint, dtMs);
                countEdges(board, joint);
                const double out = rods[board][joint].edgesOut(joint);
                if (out > furthestOut_[board][joint]) furthestOut_[board][joint] = out;
                noteEndReached(board, joint);
            }
        publishSensors();
    }

    // End-detector checks made (drives that ran into a limit switch, into a
    // mechanical stop, presses of a joint already at its end, and drives that
    // reached their end within 1.2 s of their start).
    int switchEndsChecked() const { return endChecks_[0]; }
    int stopEndsChecked() const { return endChecks_[1]; }
    int pressesChecked() const { return endChecks_[2]; }
    int earlyEndsChecked() const { return endChecks_[3]; }

    // Forget EEPROM changes so far (boot and role election write EEPROM too).
    void resetEepromTracking()
    {
        snapshotEeprom();
        eepromChanges.clear();
    }

    // Call once per firmware loop: notes which board's calibration record changed.
    void sampleEeprom()
    {
        for (int board = 0; board < kBoards; ++board)
        {
            if (std::memcmp(EEPROM.boards[board].bytes, eepromSnapshot_[board], kCalDataBytes) == 0)
                continue;
            eepromChanges.push_back(std::make_pair(board, nowMs()));
            std::memcpy(eepromSnapshot_[board], EEPROM.boards[board].bytes, kCalDataBytes);
        }
    }

    int drive(int board, int joint) const { return drive_[board][joint]; }

    int drivenCount() const
    {
        int count = 0;
        for (int board = 0; board < kBoards; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
                count += drive_[board][joint] != 0;
        return count;
    }

    int movesOnBoard(int board) const
    {
        int count = 0;
        for (const Move &move : moves) count += move.board == board;
        return count;
    }

    // The board's hall count for a joint, as the firmware reads it.
    static int32_t hallCount(int board, int joint)
    {
        return krabby::sim::hallSimEdgeCountBoard(board, static_cast<uint8_t>(joint));
    }

    bool hasViolation(const std::string &text) const
    {
        for (const std::string &violation : violations)
            if (violation.find(text) != std::string::npos) return true;
        return false;
    }

    std::string report() const
    {
        std::ostringstream text;
        for (const std::string &violation : violations) text << violation << "\n";
        text << moves.size() << " moves";
        return text.str();
    }

    static uint32_t nowMs() { return static_cast<uint32_t>(krabby::sim::simClock().nowMs()); }

private:
    // A knee nudge under CAL_OPT_KNEE_NUDGE: while open it is bounded; once
    // closed, the next move anywhere must be that knee's curl.
    struct PendingNudge
    {
        bool isOpen = false;
        bool awaitsCurl = false;
        int board = -1;
        int joint = -1;
        uint32_t startMs = 0;
        double startEdges = 0.0;
    };

    // Where a hip's extension stops: at the body when its knee is at its curl
    // end and it has a body contact, else nowhere short of its own end.
    double bodyContactPosition(int board, int joint) const
    {
        const Rod &rod = rods[board][joint];
        if (!isHip(joint) || rod.bodyContactEdges <= 0.0 || !rods[board][joint + 1].isAtCurlEnd(joint + 1))
            return HUGE_VAL;
        return rod.low + rod.bodyContactEdges / rod.edgesPerCount;
    }

    // Whether the rod sits at the end its drive pushes it into: its retracted
    // end (a limit switch) or its extended end under that drive (a mechanical
    // stop or the body).
    bool isAtDriveEnd(int board, int joint, int drive) const
    {
        const Rod &rod = rods[board][joint];
        if (drive < 0) return rod.position <= rod.low + 1e-9;
        const double end = std::min(rod.extendedEnd(joint, drive), bodyContactPosition(board, joint));
        return rod.position >= end - 1e-9;
    }

    // The end-detector checks apply to calibration drives of a hall rod whose
    // count follows its motion, on a quadrature board.
    bool isEndChecked(int board, int joint) const
    {
        const Rod &rod = rods[board][joint];
        return requestedPwm > 0 && !isOperatorJog_ && quadrature[board] && rod.sensor == RodSensor::Hall &&
               !rod.deadA && !rod.deadB && !rod.isStuck && !rod.isDeadMotor && !rod.motorReversed &&
               rod.stallSign == 0;
    }

    // The first time a driven rod reaches the end of its drive (a compliant
    // knee's: the end under the calibration PWM, not under a ramp step).
    void noteEndReached(int board, int joint)
    {
        const int drive = drive_[board][joint];
        if (drive == 0 || openMove_[board][joint] < 0 || reachedMs_[board][joint] != 0) return;
        const int endDrive = requestedPwm > 0 ? (drive > 0 ? requestedPwm : -requestedPwm) : drive;
        if (isAtDriveEnd(board, joint, endDrive)) reachedMs_[board][joint] = nowMs();
    }

    // At a move's stop: a press of a joint already at its end lasts at most
    // kSpecPressMaxMs; a drive that ran (over 1.2 s before reaching its end, so
    // the end detector had armed) stops within kSpecSwitchStopMaxMs of a limit
    // switch and kSpecMechanicalStopMaxMs of a mechanical stop, and one that
    // reached its end sooner within kSpecEarlyEndStopMaxMs. A knee whose drive
    // stopped in its taper zone, short of the stop, stopped in time.
    void checkEndStop(int board, int joint, int previous, const Move &move, uint32_t now)
    {
        const uint32_t reached = reachedMs_[board][joint];
        const bool pressed = pressAtStart_[board][joint];
        reachedMs_[board][joint] = 0;
        pressAtStart_[board][joint] = false;
        if (!isEndChecked(board, joint)) return;
        if (pressed)
        {
            ++endChecks_[2];
            if (now - move.startMs > kSpecPressMaxMs) violation(board, joint, "pressed its end longer than 1100 ms");
            return;
        }
        if (reached == 0)
        {
            const Rod &rod = rods[board][joint];
            const double end = std::min(rod.extendedEnd(joint, move.peakMagnitude), bodyContactPosition(board, joint));
            if (previous > 0 && rod.kneeTaper && now - move.startMs >= 1200 &&
                (end - rod.position) * rod.edgesPerCount < rod.taperEdges)
                ++endChecks_[1];
            return;
        }
        if (reached - move.startMs < 1200)
        {
            ++endChecks_[3];
            if (now - reached > kSpecEarlyEndStopMaxMs)
                violation(board, joint, "drove on past an end it reached within 1.2 s");
            return;
        }
        if (previous < 0)
        {
            ++endChecks_[0];
            if (now - reached > kSpecSwitchStopMaxMs) violation(board, joint, "drove on past its limit switch");
        }
        else
        {
            ++endChecks_[1];
            if (now - reached > kSpecMechanicalStopMaxMs) violation(board, joint, "drove on past its mechanical stop");
        }
    }

    // A hip driven into the body with its knee curled: a violation, once per drive.
    void noteBodyContact(int board, int joint, double contact)
    {
        if (rods[board][joint].position < contact - 1e-9 || bodyContactFlagged_[board][joint]) return;
        bodyContactFlagged_[board][joint] = true;
        violation(board, joint, "hip drove its curled tibia into the body");
    }

    // Undriven: coast and spring-out still to come, at kAfterStopCountsPerMs,
    // stopping at the rod's end in that direction: `low` when retracting; when
    // extending, its extended end under the last drive (or the body), unless it
    // already sits deeper (a knee parked by a harder curl stays there).
    void settleRod(int board, int joint, uint32_t dtMs)
    {
        Rod &rod = rods[board][joint];
        if (rod.pendingTravel == 0.0) return;
        const double limit = kAfterStopCountsPerMs * dtMs;
        const double step = rod.pendingTravel > 0 ? std::min(limit, rod.pendingTravel)
                                                  : std::max(-limit, rod.pendingTravel);
        rod.pendingTravel -= step;
        double target = rod.position + step;
        const double end = step < 0
            ? rod.low
            : std::max(rod.position, std::min(rod.extendedEnd(joint, rod.lastDriveMagnitude),
                                              bodyContactPosition(board, joint)));
        if (step < 0 ? target <= end : target >= end)
        {
            target = end;
            rod.pendingTravel = 0.0;
        }
        rod.position = target;
    }

    void moveRod(int board, int joint, uint32_t dtMs)
    {
        Rod &rod = rods[board][joint];
        const int drive = drive_[board][joint];
        if (drive == 0)
        {
            settleRod(board, joint, dtMs);
            return;
        }
        rod.drivenMs += dtMs;
        if (rod.isStuck || rod.isDeadMotor) return;
        const int sign = drive > 0 ? 1 : -1;
        if (sign == rod.stallSign) return;
        // Pressed into its curl end, a rod needs breakawayMs of drive to leave it.
        if (rod.driveFromCurlEnd && sign != curlSign(joint) && rod.drivenMs <= rod.breakawayMs) return;
        const int magnitude = drive < 0 ? -drive : drive;
        double velocity = drive * kCountsPerPwmMs * rod.speedScale *
                          (sign > 0 ? rod.extendSpeed : rod.retractSpeed);  // counts per ms
        if (rod.motorReversed) velocity = -velocity;
        if (velocity < 0)
        {
            if (rod.position > rod.low) rod.position = std::max(rod.low, rod.position + velocity * dtMs);
            return;
        }
        const double contact = bodyContactPosition(board, joint);
        const double end = std::min(rod.extendedEnd(joint, magnitude), contact);
        if (rod.position >= end)  // pushed into its end: no motion, no pull-back
        {
            noteBodyContact(board, joint, contact);
            return;
        }
        if (rod.kneeTaper && isKnee(joint))
        {
            const double edgesLeft = (end - rod.position) * rod.edgesPerCount;
            if (edgesLeft < rod.taperEdges)
            {
                const double normal = velocity * rod.edgesPerCount * 1000.0;  // edges per s
                double rate = normal * edgesLeft / rod.taperEdges;
                if (rate < kKneeTaperFloorEdgesPerS) rate = kKneeTaperFloorEdgesPerS;
                velocity = rate / rod.edgesPerCount / 1000.0;
            }
        }
        rod.position = std::min(end, rod.position + velocity * dtMs);
        noteBodyContact(board, joint, contact);
    }

    // A hall rod's A edges since the last tick go to the board's signed count.
    void countEdges(int board, int joint)
    {
        const Rod &rod = rods[board][joint];
        const long index = rod.edgeIndex();
        const long steps = index - edgeIndex_[board][joint];
        edgeIndex_[board][joint] = index;
        if (steps == 0 || rod.sensor != RodSensor::Hall || rod.deadA) return;
        const uint8_t slot = static_cast<uint8_t>(joint);
        if (rod.deadB)
        {
            // x2 decoding with B constant: each A edge undoes the one before.
            const long count = steps < 0 ? -steps : steps;
            for (long i = 0; i < count; ++i)
            {
                deadBPhase_[board][joint] = !deadBPhase_[board][joint];
                krabby::sim::hallSimAddEdges(board, slot, deadBPhase_[board][joint] ? 1 : -1);
            }
            return;
        }
        const int sign = kSpecHallExtendCountSign * (rod.hallReversed ? -1 : 1);
        krabby::sim::hallSimAddEdges(board, slot, static_cast<int32_t>(sign * steps));
    }

    // The rod's IS reading under its board's present drive (see Rod::isRun).
    int currentReading(int board, int joint) const
    {
        const Rod &rod = rods[board][joint];
        const int drive = drive_[board][joint];
        if (drive == 0 || rod.isDeadMotor) return 0;
        int sign = drive > 0 ? 1 : -1;
        if (rod.motorReversed) sign = -sign;
        double level = rod.isRun;
        if (sign < 0)
        {
            if (rod.position <= rod.low + 1e-9) level = rod.isSwitchOpen;
        }
        else
        {
            const int magnitude = drive < 0 ? -drive : drive;
            const double end = std::min(rod.extendedEnd(joint, magnitude), bodyContactPosition(board, joint));
            const double edgesLeft = (end - rod.position) * rod.edgesPerCount;
            if (edgesLeft <= 1e-9)
                level = rod.isStall;
            else if (rod.kneeTaper && isKnee(joint) && edgesLeft < rod.taperEdges)
                level = rod.isRun + (rod.isStall - rod.isRun) * (1.0 - edgesLeft / rod.taperEdges);
        }
        return static_cast<int>(level + 0.5);
    }

    void publishSensors()
    {
        for (int board = 0; board < kBoards; ++board)
            for (int joint = 0; joint < kJoints; ++joint)
            {
                const Rod &rod = rods[board][joint];
                krabby::sim::pinState().setBoardAnalogOverride(
                    board, kJointPins[joint].pot, rod.potReading(), rod.isOpenPot);
                krabby::sim::pinState().setBoardAnalogOverride(
                    board, kJointPins[joint].is, currentReading(board, joint));
            }
    }

    void snapshotEeprom()
    {
        for (int board = 0; board < kBoards; ++board)
            std::memcpy(eepromSnapshot_[board], EEPROM.boards[board].bytes, kCalDataBytes);
    }

    void violation(int board, int joint, const std::string &what)
    {
        std::ostringstream text;
        text << "t=" << nowMs() << " " << kJointNames[board][joint] << " pwm="
             << drive_[board][joint] << ": " << what;
        if (violations.size() < 20) violations.push_back(text.str());
    }

    void onWrite(const krabby::sim::PinWriteEvent &event)
    {
        if (event.board < 0 || event.board >= kBoards || event.pin < 0 || event.pin >= kPins)
            return;
        if (event.isPwm)
            duty_[event.board][event.pin] = event.value;
        else
            digital_[event.board][event.pin] = event.value;
        for (int joint = 0; joint < kJoints; ++joint)
        {
            const JointPins &pins = kJointPins[joint];
            if (event.pin != pins.pwmR && event.pin != pins.pwmL && event.pin != pins.en) continue;
            const int now = digital_[event.board][pins.en]
                ? duty_[event.board][pins.pwmR] - duty_[event.board][pins.pwmL]
                : 0;
            if (now != drive_[event.board][joint]) onDriveChange(event.board, joint, now);
        }
    }

    bool isPartnerCurled(int board, int partner) const
    {
        return lastMoveSign_[board][partner] == curlSign(partner) &&
               rods[board][partner].isAtCurlEnd(partner);
    }

    // The rod's motion state at a drive's start and stop: breakaway from its
    // curl end, then coast and (a knee leaving its curl end) spring-out.
    void noteRodDrive(int board, int joint, int previous, int pwm)
    {
        Rod &rod = rods[board][joint];
        const int magnitude = pwm < 0 ? -pwm : pwm;
        if (pwm != 0 && (previous == 0 || (pwm > 0) != (previous > 0)))
        {
            rod.drivenMs = 0;
            rod.driveFromCurlEnd = rod.isAtCurlEnd(joint);
            rod.pendingTravel = 0.0;
            rod.lastDriveSign = pwm > 0 ? 1 : -1;
            rod.lastDriveMagnitude = 0;
            bodyContactFlagged_[board][joint] = false;
        }
        if (magnitude > rod.lastDriveMagnitude) rod.lastDriveMagnitude = magnitude;
        if (pwm != 0 || previous == 0) return;
        rod.pendingTravel = rod.lastDriveSign * rod.coastEdges / rod.edgesPerCount;
        if (isKnee(joint) && rod.lastDriveSign != curlSign(joint) && rod.driveFromCurlEnd)
            rod.pendingTravel += rod.lastDriveSign * rod.springOutEdges / rod.edgesPerCount;
    }

    void onDriveChange(int board, int joint, int pwm)
    {
        const int previous = drive_[board][joint];
        const uint32_t now = nowMs();
        drive_[board][joint] = pwm;
        changes.push_back(DriveChange{now, board, joint, pwm});
        noteRodDrive(board, joint, previous, pwm);

        if (previous != 0 && (pwm == 0 || (pwm > 0) != (previous > 0)))
        {
            closeMove(board, joint, previous, now);
            if (pwm != 0) violation(board, joint, "reversed without stopping");
        }
        if (pwm == 0)
        {
            lastChangeMs_[board][joint] = now;
            return;
        }

        const int magnitude = pwm < 0 ? -pwm : pwm;
        const int sign = pwm > 0 ? 1 : -1;
        const bool isAllowed = board == allowedBoard_ && joint == allowedJoint_;
        if (isFrozen_ && !isAllowed) violation(board, joint, "driven after the stop");

        if (drivenCount() > 1) violation(board, joint, "a second actuator is driven");
        if (isYaw(joint)) violation(board, joint, "a yaw is driven");
        if (requestedPwm > 0 && !isAllowed && !isOperatorJog_)
        {
            if (magnitude > requestedPwm) violation(board, joint, "PWM above the requested calibration PWM");
            const int previousMagnitude = previous < 0 ? -previous : previous;
            if (previous == 0 && magnitude > kSpecRampStep)
                violation(board, joint, "did not ramp up from 0");
            if (previous != 0)
            {
                if (magnitude <= previousMagnitude || magnitude - previousMagnitude > kSpecRampStep)
                    violation(board, joint, "ramp step is not +5");
                if (now - lastChangeMs_[board][joint] < kSpecRampIntervalMs)
                    violation(board, joint, "ramp step faster than 20 ms");
            }
        }
        lastChangeMs_[board][joint] = now;

        if (openMove_[board][joint] < 0)
        {
            if (!isOperatorJog_) checkMoveStart(board, joint, sign, now);
            // At its end under the calibration PWM: a compliant knee parked
            // by a softer curl is not yet at the end this drive reaches.
            pressAtStart_[board][joint] =
                isAtDriveEnd(board, joint, requestedPwm > 0 ? sign * requestedPwm : pwm);
            reachedMs_[board][joint] = 0;
            openMove_[board][joint] = static_cast<int>(moves.size());
            const double position = rods[board][joint].position;
            moves.push_back(Move{board, joint, sign, now, 0, magnitude, curledMask(), position, position});
        }
        else
        {
            Move &move = moves[static_cast<size_t>(openMove_[board][joint])];
            if (magnitude > move.peakMagnitude) move.peakMagnitude = magnitude;
        }
    }

    // A hip moves only with its knee curled (its last move was to the curl end
    // and it is there); a knee leaves its curl end only with its hip curled, or,
    // with CAL_OPT_KNEE_NUDGE, as a bounded nudge followed directly by its curl.
    void checkMoveStart(int board, int joint, int sign, uint32_t now)
    {
        if (nudge_.awaitsCurl)
        {
            nudge_.awaitsCurl = false;
            if (board != nudge_.board || joint != nudge_.joint || sign != curlSign(joint))
                violation(board, joint, "a knee nudge is not followed directly by its curl");
            const Rod &nudged = rods[nudge_.board][nudge_.joint];
            const double endedOut = std::fabs(nudged.edgePosition() - nudge_.startEdges);
            if (nudged.deadA)
            {
                if (endedOut > kSpecBlindKneeNudgeMaxEdges + kSpecKneeSpringMaxEdges)
                    violation(nudge_.board, nudge_.joint, "blind knee nudge ended more than 58 edges out");
            }
            else if (endedOut > kSpecKneeNudgeMaxEdges + kSpecKneeSpringMaxEdges)
            {
                violation(nudge_.board, nudge_.joint, "knee nudge ended more than 25 edges out");
            }
        }
        if (isHip(joint) && !isPartnerCurled(board, joint + 1))
            violation(board, joint, "hip driven while its knee is not curled");
        if (isKnee(joint) && sign != curlSign(joint) && !isPartnerCurled(board, joint - 1))
        {
            if (!kneeNudgeOption)
            {
                violation(board, joint, "knee uncurled while its hip is not curled");
                return;
            }
            nudge_.isOpen = true;
            nudge_.board = board;
            nudge_.joint = joint;
            nudge_.startMs = now;
            nudge_.startEdges = rods[board][joint].edgePosition();
        }
    }

    void closeMove(int board, int joint, int previous, uint32_t now)
    {
        lastMoveSign_[board][joint] = previous > 0 ? 1 : -1;
        const int index = openMove_[board][joint];
        openMove_[board][joint] = -1;
        if (nudge_.isOpen && board == nudge_.board && joint == nudge_.joint)
        {
            nudge_.isOpen = false;
            nudge_.awaitsCurl = true;
            const double moved = std::fabs(rods[board][joint].edgePosition() - nudge_.startEdges);
            if (now - nudge_.startMs > kSpecKneeNudgeMaxMs)
                violation(board, joint, "knee nudge longer than 1000 ms");
            if (rods[board][joint].deadA)
            {
                if (moved > kSpecBlindKneeNudgeMaxEdges)
                    violation(board, joint, "blind knee nudge moved more than 49 edges");
            }
            else if (moved > kSpecKneeNudgeMaxEdges)
            {
                violation(board, joint, "knee nudge moved more than 16 edges");
            }
        }
        if (index < 0) return;
        moves[static_cast<size_t>(index)].endMs = now;
        moves[static_cast<size_t>(index)].endPosition = rods[board][joint].position;
        checkEndStop(board, joint, previous, moves[static_cast<size_t>(index)], now);
    }

    int digital_[kBoards][kPins];
    int duty_[kBoards][kPins];
    int drive_[kBoards][kJoints];
    uint32_t lastChangeMs_[kBoards][kJoints];
    int lastMoveSign_[kBoards][kJoints];
    int openMove_[kBoards][kJoints];
    long edgeIndex_[kBoards][kJoints];
    bool deadBPhase_[kBoards][kJoints];
    double furthestOut_[kBoards][kJoints] = {};
    bool bodyContactFlagged_[kBoards][kJoints] = {};
    uint32_t reachedMs_[kBoards][kJoints] = {};
    bool pressAtStart_[kBoards][kJoints] = {};
    int endChecks_[4] = {};
    unsigned char eepromSnapshot_[kBoards][kCalDataBytes];
    PendingNudge nudge_;
    bool isFrozen_ = false;
    bool isOperatorJog_ = false;
    int allowedBoard_ = -1;
    int allowedJoint_ = -1;
};

// The (board, joint, direction) every move of a run must have, in order.
struct ExpectedMove
{
    int board;
    int joint;
    int sign;
};

inline void appendSeek(std::vector<ExpectedMove> &out, int board)
{
    for (int leg = 0; leg < 2; ++leg)
    {
        const int hip = 3 * leg + 1;
        const int knee = 3 * leg + 2;
        out.push_back({board, knee, curlSign(knee)});
        out.push_back({board, hip, curlSign(hip)});
    }
}

inline void appendSweep(std::vector<ExpectedMove> &out, int board)
{
    for (int leg = 0; leg < 2; ++leg)
    {
        const int hip = 3 * leg + 1;
        const int knee = 3 * leg + 2;
        out.push_back({board, knee, curlSign(knee)});
        out.push_back({board, hip, curlSign(hip)});
        out.push_back({board, hip, -curlSign(hip)});
        out.push_back({board, hip, curlSign(hip)});
        out.push_back({board, knee, -curlSign(knee)});
        out.push_back({board, knee, curlSign(knee)});
    }
}

// A joint's verification: a curl that shows no motion, the nudge out, the curl back.
inline void appendNudged(std::vector<ExpectedMove> &out, int board, int joint)
{
    out.push_back({board, joint, curlSign(joint)});
    out.push_back({board, joint, -curlSign(joint)});
    out.push_back({board, joint, curlSign(joint)});
}

// Empty when the recorded moves from `first` on match `expected` exactly, else what differs.
inline std::string compareMoves(
    const std::vector<Move> &moves,
    const std::vector<ExpectedMove> &expected,
    size_t first = 0)
{
    std::ostringstream text;
    const size_t recorded = moves.size() > first ? moves.size() - first : 0;
    if (recorded != expected.size())
        text << "expected " << expected.size() << " moves, got " << recorded << "\n";
    const size_t count = recorded < expected.size() ? recorded : expected.size();
    for (size_t i = 0; i < count; ++i)
    {
        const Move &move = moves[first + i];
        const ExpectedMove &want = expected[i];
        if (move.board == want.board && move.joint == want.joint && move.sign == want.sign) continue;
        text << "move " << i << ": got " << kJointNames[move.board][move.joint] << (move.sign > 0 ? " +" : " -")
             << ", want " << kJointNames[want.board][want.joint] << (want.sign > 0 ? " +" : " -") << "\n";
        break;
    }
    return text.str();
}

}  // namespace calibration_timeline

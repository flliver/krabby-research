#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "unity.h"

#include "src/actuator/leg_calibration.h"

#include "robot_log_replay.hpp"

namespace
{

struct FakeOutput final : public CalibrationTextSink
{
    std::string text;

    void printFlash(const char *value) override { text += value; }
    void printText(const char *value) override { text += value; }
    void printNumber(long value) override { text += std::to_string(value); }
    void printChar(char value) override { text += value; }
    void endLine() override { text += '\n'; }

    std::vector<std::string> lines() const
    {
        std::vector<std::string> result;
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) result.push_back(line);
        return result;
    }

    bool has(const std::string &line) const
    {
        for (const std::string &candidate : lines())
            if (candidate == line) return true;
        return false;
    }

    // Lines that start with prefix.
    int count(const std::string &prefix) const
    {
        int result = 0;
        for (const std::string &candidate : lines())
            result += candidate.compare(0, prefix.size(), prefix) == 0;
        return result;
    }

    // The first line that starts with prefix ("" if none).
    std::string find(const std::string &prefix) const
    {
        for (const std::string &candidate : lines())
            if (candidate.compare(0, prefix.size(), prefix) == 0) return candidate;
        return std::string();
    }

    // The number after the first "key=" in the first line that starts with prefix.
    long valueAfter(const std::string &prefix, const std::string &key) const
    {
        const std::string line = find(prefix);
        const size_t at = line.find(key + "=");
        TEST_ASSERT_TRUE_MESSAGE(at != std::string::npos, prefix.c_str());
        return std::strtol(line.c_str() + at + key.size() + 1, nullptr, 10);
    }

    std::string last() const
    {
        const std::vector<std::string> all = lines();
        return all.empty() ? std::string() : all.back();
    }

    // An END line exactly `line` plus the end detector's " by=<by> cur=a/b".
    bool hasEnd(const std::string &line, const char *by) const
    {
        const std::string prefix = line + " by=" + by + " cur=";
        for (const std::string &candidate : lines())
        {
            if (candidate.compare(0, prefix.size(), prefix) != 0) continue;
            const std::string rest = candidate.substr(prefix.size());
            const size_t slash = rest.find('/');
            if (slash == std::string::npos || slash == 0 || slash + 1 == rest.size()) return false;
            return rest.find_first_not_of("0123456789/") == std::string::npos;
        }
        return false;
    }
};

struct Drive
{
    uint8_t joint;
    int pwm;
    uint32_t ms;
};

// Six joints with rods in [low, high] extension; +PWM extends. A pot rod's pot
// reads the extension (or its mirror when reversed). A hall rod counts edges:
// the count falls as the rod extends (or rises when hallReversed), and its pot
// pin reads hall B at the rails. Movement: speed units every moveEvery ticks
// while driven, unless stuck (either way) or stalled in the direction stallSign
// (+1 / -1; 0 = never). A drive of sign breakawaySign does not move the rod for
// its first breakawayMs (a curled knee breaking free); when a drive of sign
// springSign stops, the rod moves springOut further (a curled knee springing
// out). The firmware's view of each joint (kind, home, limits) follows the
// plant interface: on a build without quadrature every kind is Pot.
//
// Motor current (tenths of an IS count, motorCurrentTenths), as measured on the
// robot (2026-11-24): 0 undriven; currentRun while driven (also while stuck,
// stalled by stallSign or breaking free: a stall with no signature); driven to
// retract at `low` (the actuator's limit switch opens): currentOpen; driven to
// extend at `high` (the leg's mechanical stop): currentStall. A rod with
// taperEdges > 0 has a compliant stop: over its last taperEdges before `high`
// it slows in proportion and its current ramps from currentRun to
// currentStall. currentNoise adds a seeded uniform +-noise to every driven read.
struct FakePlant final : public LegCalibrationPlant
{
    uint32_t now = 0;
    uint32_t ticks = 0;
    bool quadrature = false;
    int extension[6] = {500, 500, 500, 500, 500, 500};
    int low[6] = {100, 100, 100, 100, 100, 100};
    int high[6] = {900, 900, 900, 900, 900, 900};
    int pwm[6] = {};
    bool open[6] = {};
    bool reversed[6] = {};
    bool stuck[6] = {};
    int stallSign[6] = {};
    int moveEvery[6] = {1, 1, 1, 1, 1, 1};
    bool hallRod[6] = {};
    bool hallReversed[6] = {};
    bool hallDead[6] = {};
    int32_t countBias[6] = {};
    bool currentAbsent[6] = {};
    int breakawaySign[6] = {};
    uint32_t breakawayMs[6] = {};
    int springSign[6] = {};
    int springOut[6] = {};
    uint32_t driveStartMs[6] = {};
    int currentOpen[6] = {75, 75, 75, 75, 75, 75};
    int currentRun[6] = {110, 110, 110, 110, 110, 110};
    int currentStall[6] = {170, 170, 170, 170, 170, 170};
    int taperEdges[6] = {};
    int currentNoise = 0;
    uint32_t noiseSeed = 1;
    int speed = 1;
    std::function<int(uint8_t, uint32_t)> potScript;
    std::function<int(uint8_t, uint32_t)> snapScript;
    const char *names[6] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    FakeOutput output;
    std::vector<Drive> drives;
    int stopAllCount = 0;
    int saves = 0;
    // The firmware's view.
    JointSensor kind[6] = {JointSensor::Pot, JointSensor::Pot, JointSensor::Pot,
                           JointSensor::Pot, JointSensor::Pot, JointSensor::Pot};
    int32_t home[6] = {};
    bool homed[6] = {};
    bool homeLost[6] = {};
    bool limitsValid[6] = {};
    int minStop[6] = {0, 0, 0, 0, 0, 0};
    int maxStop[6] = {1023, 1023, 1023, 1023, 1023, 1023};
    std::vector<uint8_t> storedJoints;
    std::vector<uint8_t> setHomeJoints;
    std::vector<uint8_t> clearHomeJoints;
    std::vector<std::pair<uint8_t, JointSensor>> setSensorCalls;

    // Rev 3: hips and knees are hall rods of 1000 edges (hips curl to 0, knees
    // to 1000), every kind starts Unknown with no limits.
    void useHall()
    {
        quadrature = true;
        for (int joint = 0; joint < 6; ++joint)
        {
            kind[joint] = JointSensor::Unknown;
            minStop[joint] = 0;
            maxStop[joint] = 0;
            if (joint % 3 == 0) continue;
            hallRod[joint] = true;
            low[joint] = 0;
            high[joint] = 1000;
        }
    }

    void drive(uint8_t joint, int value) override
    {
        if (value != 0 && (pwm[joint] == 0 || (value > 0) != (pwm[joint] > 0))) driveStartMs[joint] = now;
        pwm[joint] = value;
        drives.push_back({joint, value, now});
    }
    void stopAll() override
    {
        ++stopAllCount;
        for (int joint = 0; joint < 6; ++joint)
        {
            if (pwm[joint] != 0 && springSign[joint] == (pwm[joint] > 0 ? 1 : -1))
                extension[joint] = clampExtension(joint, extension[joint] + springOut[joint]);
            pwm[joint] = 0;
        }
    }
    int rawPot(uint8_t joint) override
    {
        if (potScript) return potScript(joint, now);
        if (hallRod[joint]) return (hallCount(joint) % 2) != 0 ? 1016 : 12;
        return reversed[joint] ? low[joint] + high[joint] - extension[joint] : extension[joint];
    }
    int potSnapshot(uint8_t joint) override
    {
        if (snapScript) return snapScript(joint, now);
        return rawPot(joint);
    }
    bool isPotOpen(uint8_t joint) override { return open[joint]; }
    bool hasQuadratureHall() override { return quadrature; }
    int32_t hallCount(uint8_t joint) override
    {
        if (!hallRod[joint] || hallDead[joint]) return countBias[joint];
        return countBias[joint] + (hallReversed[joint] ? 1 : -1) * extension[joint];
    }
    int position(uint8_t joint) override
    {
        if (kind[joint] == JointSensor::Pot) return rawPot(joint);
        int32_t x = ACTUATOR_HALL_EXTEND_COUNT_SIGN * (hallCount(joint) - home[joint]);
        if (x > ACTUATOR_HALL_POSITION_LIMIT) x = ACTUATOR_HALL_POSITION_LIMIT;
        if (x < -ACTUATOR_HALL_POSITION_LIMIT) x = -ACTUATOR_HALL_POSITION_LIMIT;
        return static_cast<int>(x);
    }
    JointSensor sensor(uint8_t joint) override { return kind[joint]; }
    void setSensor(uint8_t joint, JointSensor value) override
    {
        setSensorCalls.push_back({joint, value});
        if (!quadrature) value = JointSensor::Pot;
        if (kind[joint] == value) return;
        kind[joint] = value;
        home[joint] = 0;
        homed[joint] = false;
        homeLost[joint] = false;
        limitsValid[joint] = false;
        minStop[joint] = 0;
        maxStop[joint] = value == JointSensor::Pot ? 1023 : 0;
    }
    bool isHomed(uint8_t joint) override { return homed[joint]; }
    void setHome(uint8_t joint) override
    {
        setHomeJoints.push_back(joint);
        home[joint] = hallCount(joint);
        homed[joint] = true;
        homeLost[joint] = false;
    }
    void clearHome(uint8_t joint) override
    {
        clearHomeJoints.push_back(joint);
        if (!homed[joint]) return;
        homed[joint] = false;
        homeLost[joint] = true;
    }
    bool isCurrentAbsent(uint8_t joint) override { return currentAbsent[joint]; }
    uint16_t motorCurrentTenths(uint8_t joint) override
    {
        if (pwm[joint] == 0) return 0;
        const int direction = pwm[joint] > 0 ? 1 : -1;
        int value = currentRun[joint];
        if (direction < 0 && extension[joint] <= low[joint])
            value = currentOpen[joint];
        else if (direction > 0 && extension[joint] >= high[joint])
            value = currentStall[joint];
        else if (direction > 0 && high[joint] - extension[joint] < taperEdges[joint])
            value = currentRun[joint] + (currentStall[joint] - currentRun[joint]) *
                (taperEdges[joint] - (high[joint] - extension[joint])) / taperEdges[joint];
        if (currentNoise > 0)
        {
            noiseSeed = noiseSeed * 1103515245UL + 12345UL;
            value += static_cast<int>((noiseSeed >> 16) % static_cast<uint32_t>(2 * currentNoise + 1)) - currentNoise;
        }
        return static_cast<uint16_t>(value < 0 ? 0 : value);
    }
    bool measuredLimit(uint8_t joint, bool extended, int &value) override
    {
        if (kind[joint] != JointSensor::Pot || !limitsValid[joint]) return false;
        value = extended ? maxStop[joint] : minStop[joint];
        return true;
    }
    int savedStroke(uint8_t joint) override
    {
        return kind[joint] == JointSensor::Hall && limitsValid[joint]
            ? maxStop[joint] - minStop[joint]
            : 0;
    }
    uint32_t nowMs() override { return now; }
    const char *jointName(uint8_t joint) override { return names[joint]; }
    const char *roleLabel() override { return "FRONT"; }
    CalibrationTextSink &out() override { return output; }
    void storeLimits(uint8_t joint, int minimum, int maximum) override
    {
        storedJoints.push_back(joint);
        minStop[joint] = minimum;
        maxStop[joint] = maximum;
        limitsValid[joint] = true;
    }
    void saveLimits() override { ++saves; }

    int drivenCount() const
    {
        int count = 0;
        for (int value : pwm) count += value != 0;
        return count;
    }

    int clampExtension(int joint, int value) const
    {
        if (value > high[joint]) return high[joint];
        return value < low[joint] ? low[joint] : value;
    }

    void advance(uint32_t milliseconds)
    {
        now += milliseconds;
        ++ticks;
        for (int joint = 0; joint < 6; ++joint)
        {
            if (pwm[joint] == 0 || stuck[joint]) continue;
            if (ticks % static_cast<uint32_t>(moveEvery[joint]) != 0) continue;
            const int direction = pwm[joint] > 0 ? 1 : -1;
            if (direction == stallSign[joint]) continue;
            if (direction == breakawaySign[joint] && now - driveStartMs[joint] < breakawayMs[joint]) continue;
            // Compliant stop: within taperEdges of `high` the rod moves on one
            // tick in (taperEdges / edges left).
            const int left = high[joint] - extension[joint];
            if (direction > 0 && left > 0 && left < taperEdges[joint] &&
                ticks % static_cast<uint32_t>((taperEdges[joint] + left - 1) / left) != 0)
                continue;
            extension[joint] = clampExtension(joint, extension[joint] + direction * speed);
        }
    }
};

// One 10 ms tick, checking the safety invariants: one joint at a time, never a
// yaw, never above the requested PWM; a hip moves only with its knee curled; a
// knee leaves its curl end only with its hip curled, or as a CAL_OPT_KNEE_NUDGE nudge.
void tick(LegCalibration &calibration, FakePlant &plant, int requestedPwm, uint32_t tickMs = 10)
{
    plant.advance(tickMs);
    calibration.update(plant);
    TEST_ASSERT_TRUE(plant.drivenCount() <= 1);
    TEST_ASSERT_EQUAL_INT(0, plant.pwm[0]);
    TEST_ASSERT_EQUAL_INT(0, plant.pwm[3]);
    for (uint8_t joint = 0; joint < 6; ++joint)
    {
        if (plant.pwm[joint] == 0) continue;
        TEST_ASSERT_TRUE(std::abs(plant.pwm[joint]) <= requestedPwm);
        if (isLegCalibrationHip(joint))
            TEST_ASSERT_TRUE(calibration.isJointCurled(joint + 1));
        else if (!calibration.isCurrentMoveCurl() &&
                 !(calibration.moveStage() == LegCalibrationMoveStage::NudgeOut &&
                   (calibration.options() & CAL_OPT_KNEE_NUDGE) != 0))
            TEST_ASSERT_TRUE(calibration.isJointCurled(joint - 1));
    }
}

// Runs until the sequencer stops.
int runToEnd(LegCalibration &calibration, FakePlant &plant, int requestedPwm, uint32_t tickMs = 10)
{
    int ticks = 0;
    while (calibration.isRunning() && ticks < 200000)
    {
        tick(calibration, plant, requestedPwm, tickMs);
        ++ticks;
    }
    return ticks;
}

// Runs until pred() holds or the sequencer stops.
void runUntil(LegCalibration &calibration, FakePlant &plant, int requestedPwm,
              const std::function<bool()> &pred)
{
    int ticks = 0;
    while (calibration.isRunning() && !pred() && ticks < 200000)
    {
        tick(calibration, plant, requestedPwm);
        ++ticks;
    }
}

// Runs the sequencer until it reaches `step` of the current pass (or stops).
void runUntilStep(LegCalibration &calibration, FakePlant &plant, uint8_t step, int requestedPwm = CAL_PWM_MAX)
{
    runUntil(calibration, plant, requestedPwm, [&]() { return calibration.currentStep() >= step; });
}

std::vector<std::string> moveLines(const FakeOutput &output)
{
    std::vector<std::string> moves;
    for (const std::string &line : output.lines())
        if (line.find(" MOVE ") != std::string::npos) moves.push_back(line);
    return moves;
}

std::string reasonOf(const LegCalibration &calibration)
{
    FakeOutput reason;
    calibration.printFailureReason(reason);
    return reason.text;
}

bool drovePartnerAfter(const FakePlant &plant, size_t drivesBefore, uint8_t joint)
{
    for (size_t i = drivesBefore; i < plant.drives.size(); ++i)
        if (plant.drives[i].joint == joint) return true;
    return false;
}

void assertFailure(const LegCalibration &calibration, LegCalibrationFailure failure)
{
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Failed),
        static_cast<int>(calibration.outcome()));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(failure), static_cast<int>(calibration.failure()));
}

void assertCompleted(const LegCalibration &calibration)
{
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Completed),
        static_cast<int>(calibration.outcome()));
}

// A hall plant whose leg-A knee and hip are proven curled and homed by a SEEK
// (knee home count -1000, hip home count 0).
void seekHall(LegCalibration &calibration, FakePlant &plant, uint8_t options = 0)
{
    plant.useHall();
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, options));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_TRUE(plant.homed[1] && plant.homed[2]);
}

}  // namespace

void setUp() {}
void tearDown() {}

// The tuning the spec and the user agreed on, written out: a change to any of
// these is a behaviour change on the robot and must be deliberate.
static void test_constants_match_the_agreed_tuning()
{
    TEST_ASSERT_EQUAL_INT(128, CAL_DEFAULT_PWM);
    TEST_ASSERT_EQUAL_INT(40, CAL_PWM_MIN);
    TEST_ASSERT_EQUAL_INT(150, CAL_PWM_MAX);
    TEST_ASSERT_EQUAL_INT(5, CAL_PWM_RAMP_STEP);
    TEST_ASSERT_EQUAL_UINT32(20, CAL_PWM_RAMP_INTERVAL_MS);
    TEST_ASSERT_EQUAL_UINT32(1500, CAL_START_GRACE_MS);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_STALL_WINDOW_MS);
    TEST_ASSERT_EQUAL_INT(3, CAL_STALL_MIN_PROGRESS);
    TEST_ASSERT_EQUAL_UINT32(300, CAL_SETTLE_MS);
    TEST_ASSERT_EQUAL_UINT32(120000, CAL_MOVE_TIMEOUT_MS);
    TEST_ASSERT_EQUAL_INT(30, CAL_MIN_TRAVEL_COUNTS);
    TEST_ASSERT_EQUAL_INT(30, CAL_CURL_END_TOLERANCE_COUNTS);
    TEST_ASSERT_EQUAL_UINT(78, CAL_ROBOT_REASON_MAX_CHARS);
    TEST_ASSERT_EQUAL_UINT32(135000, CAL_FOLLOWER_SILENCE_MS);
    // Confirmed on the robot: +PWM extends; hips curl retracting, knees extending.
    TEST_ASSERT_EQUAL_INT(-1, CAL_HIP_CURL_PWM_SIGN);
    TEST_ASSERT_EQUAL_INT(1, CAL_KNEE_CURL_PWM_SIGN);
    // Hall thresholds the tests below are written against.
    TEST_ASSERT_EQUAL_INT(12, CAL_HALL_DETECT_EDGES);
    TEST_ASSERT_EQUAL_INT(12, CAL_HALL_CURL_EVIDENCE_EDGES);
    TEST_ASSERT_EQUAL_INT(3, CAL_HALL_STALL_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(40, CAL_HALL_CURL_END_TOLERANCE_EDGES);
    // From the on-robot measurement of 2026-11-23 (one knee: stroke 776 edges).
    TEST_ASSERT_EQUAL_INT(450, CAL_HALL_MIN_STROKE_EDGES);
    TEST_ASSERT_EQUAL_INT(1500, CAL_HALL_MAX_STROKE_EDGES);
    // The owner's fixed 70% hip sweep (a curled tibia meets the body at 941-1003).
    TEST_ASSERT_EQUAL_INT(700, CAL_HIP_SWEEP_EDGES);
    TEST_ASSERT_EQUAL_INT(24, CAL_HIP_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_UINT32(2000, CAL_HIP_NUDGE_MAX_MS);
    TEST_ASSERT_EQUAL_INT(12, CAL_KNEE_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_KNEE_NUDGE_MAX_MS);
    TEST_ASSERT_EQUAL_INT(128, CAL_KNEE_NUDGE_MAX_PWM);
    TEST_ASSERT_EQUAL_INT(2, CAL_KNEE_RETURN_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(30, CAL_POT_NUDGE_COUNTS);
    TEST_ASSERT_EQUAL_INT(-1, ACTUATOR_HALL_EXTEND_COUNT_SIGN);
}

static void test_step_tables_follow_curl_first_order()
{
    TEST_ASSERT_EQUAL_UINT(4, legCalibrationStepCount(CAL_PASS_SEEK));
    TEST_ASSERT_EQUAL_UINT(12, legCalibrationStepCount(CAL_PASS_SWEEP));
    const uint8_t seekJoints[] = {2, 1, 5, 4};
    for (uint8_t step = 0; step < 4; ++step)
    {
        const LegCalibrationStep entry = legCalibrationStep(CAL_PASS_SEEK, step);
        TEST_ASSERT_EQUAL_UINT(seekJoints[step], entry.joint);
        TEST_ASSERT_TRUE(entry.curl);
        TEST_ASSERT_FALSE(entry.completesJoint);
    }
    const uint8_t sweepJoints[] = {2, 1, 1, 1, 2, 2, 5, 4, 4, 4, 5, 5};
    const bool sweepCurl[] = {true, true, false, true, false, true, true, true, false, true, false, true};
    const bool sweepCompletes[] = {false, false, false, true, false, true, false, false, false, true, false, true};
    for (uint8_t step = 0; step < 12; ++step)
    {
        const LegCalibrationStep entry = legCalibrationStep(CAL_PASS_SWEEP, step);
        TEST_ASSERT_EQUAL_UINT(sweepJoints[step], entry.joint);
        TEST_ASSERT_EQUAL(sweepCurl[step], entry.curl);
        TEST_ASSERT_EQUAL(sweepCompletes[step], entry.completesJoint);
    }
    TEST_ASSERT_EQUAL_INT(-1, legCalibrationPwmSign(1, true));
    TEST_ASSERT_EQUAL_INT(1, legCalibrationPwmSign(1, false));
    TEST_ASSERT_EQUAL_INT(1, legCalibrationPwmSign(5, true));
    TEST_ASSERT_EQUAL_INT(-1, legCalibrationPwmSign(5, false));
}

// Every joint, curl and uncurl, nudge or not, option on and off, every mask.
static void test_partner_ready_table()
{
    for (uint8_t joint = 0; joint < 6; ++joint)
    {
        const uint8_t hipBit = static_cast<uint8_t>(1U << ((joint / 3) * 3 + 1));
        const uint8_t kneeBit = static_cast<uint8_t>(1U << ((joint / 3) * 3 + 2));
        for (unsigned mask = 0; mask < 64; ++mask)
            for (int curl = 0; curl < 2; ++curl)
                for (int nudge = 0; nudge < 2; ++nudge)
                    for (int option = 0; option < 2; ++option)
                    {
                        bool expected;
                        if (joint % 3 == 1)
                            expected = (mask & kneeBit) != 0;
                        else if (curl)
                            expected = true;
                        else if (nudge && option)
                            expected = true;
                        else
                            expected = (mask & hipBit) != 0;
                        TEST_ASSERT_EQUAL(expected, legCalibrationPartnerReady(
                            static_cast<uint8_t>(mask), joint, curl != 0, nudge != 0,
                            option ? CAL_OPT_KNEE_NUDGE : 0));
                    }
    }
    // Spelled out: a hip never moves, even as a nudge, without its curled knee.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x02, 1, true, false, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 4, false, true, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x20, 4, false, true, 0));
    // A knee leaves its curl end before its hip only as an opted-in nudge.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 2, false, true, 0));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 2, false, false, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 2, false, true, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x02, 2, false, false, 0));
}

// Every reason fits CAL_ROBOT_REASON_MAX_CHARS with a 4-character joint name,
// PWM 150 and 5-digit signed numbers, in every unit and direction.
static void test_every_failure_reason_fits_the_leader_echo()
{
    const LegCalibrationFailure failures[] = {
        LegCalibrationFailure::PotNotConnected, LegCalibrationFailure::NoEndOfTravel,
        LegCalibrationFailure::BarelyMoved, LegCalibrationFailure::NotAtCurlEnd,
        LegCalibrationFailure::WrongDirection, LegCalibrationFailure::PotWrongWay,
        LegCalibrationFailure::NoFeedback, LegCalibrationFailure::NoMotorCurrent,
        LegCalibrationFailure::NoMotion, LegCalibrationFailure::ShortOfCurlEnd,
        LegCalibrationFailure::NoCurlBack, LegCalibrationFailure::StrokeOutOfRange, LegCalibrationFailure::StrokeShort,
        LegCalibrationFailure::SweepShort, LegCalibrationFailure::StuckMidStroke,
        LegCalibrationFailure::PartnerNotCurled, LegCalibrationFailure::Aborted,
    };
    const JointSensor kinds[] = {JointSensor::Unknown, JointSensor::Pot, JointSensor::Hall};
    size_t longest = 0;
    for (LegCalibrationFailure failure : failures)
        for (JointSensor kind : kinds)
            for (int curl = 0; curl < 2; ++curl)
                for (int32_t value : {-99999, 99999})
                {
                    FakeOutput out;
                    printLegCalibrationFailureReason(out, failure, "FRKL", kind, curl != 0,
                        failure == LegCalibrationFailure::Aborted ? 'H' : value, -32768, CAL_PWM_MAX);
                    TEST_ASSERT_TRUE(out.text.size() > 4);
                    TEST_ASSERT_TRUE(out.text.size() <= CAL_ROBOT_REASON_MAX_CHARS);
                    if (out.text.size() > longest) longest = out.text.size();
                }
    TEST_ASSERT_EQUAL_UINT(77, longest);

    FakeOutput none;
    printLegCalibrationFailureReason(none, LegCalibrationFailure::None, "FRKL", JointSensor::Hall,
        true, 1, 2, 80);
    TEST_ASSERT_EQUAL_STRING("", none.text.c_str());

    struct Expected
    {
        LegCalibrationFailure failure;
        JointSensor kind;
        int32_t value;
        int value2;
        const char *text;
    };
    const Expected expected[] = {
        {LegCalibrationFailure::BarelyMoved, JointSensor::Hall, 299, 0,
         "FRKL barely moved (299 edges) at PWM 150: check hall wiring or raise speed"},
        {LegCalibrationFailure::NotAtCurlEnd, JointSensor::Hall, -7, 0,
         "FRKL did not reach its curl end (hall=-7)"},
        {LegCalibrationFailure::WrongDirection, JointSensor::Unknown, -12, 0,
         "FRKL hall counts the wrong way: check motor or hall A/B wiring"},
        {LegCalibrationFailure::PotWrongWay, JointSensor::Pot, 0, 0,
         "FRKL pot moves the wrong way: check motor or pot end wires"},
        {LegCalibrationFailure::NoFeedback, JointSensor::Unknown, 0, 0,
         "FRKL no feedback: no hall edges, pot not moving (check sensor cable)"},
        {LegCalibrationFailure::NoFeedback, JointSensor::Unknown, 50, 0,
         "FRKL no feedback at PWM 50: check sensor cable, or calibrate at PWM 128"},
        {LegCalibrationFailure::NoCurlBack, JointSensor::Hall, 11, 0,
         "FRKL curled back only 11 edges after its nudge"},
        {LegCalibrationFailure::NoCurlBack, JointSensor::Pot, 29, 0,
         "FRKL curled back only 29 counts after its nudge"},
        {LegCalibrationFailure::NoMotorCurrent, JointSensor::Hall, 0, 0,
         "FRKL no motor current at PWM 150: check motor wiring"},
        {LegCalibrationFailure::NoMotion, JointSensor::Unknown, 0, 0,
         "FRKL no motion seen: jog it off its curl end, then retry"},
        {LegCalibrationFailure::ShortOfCurlEnd, JointSensor::Pot, 31, 0,
         "FRKL stopped 31 counts short of its curl end"},
        {LegCalibrationFailure::StrokeOutOfRange, JointSensor::Hall, 1501, 0,
         "FRKL stroke 1501 edges, outside 450-1500"},
        {LegCalibrationFailure::StrokeShort, JointSensor::Hall, 849, 1000,
         "FRKL stroke 849 edges, saved 1000: obstacle?"},
        {LegCalibrationFailure::SweepShort, JointSensor::Hall, 280, 0,
         "FRKL stopped 280 edges short of its 700-edge sweep: obstacle?"},
        {LegCalibrationFailure::StuckMidStroke, JointSensor::Hall, 771, 0,
         "FRKL stuck mid-stroke after 771 edges at PWM 150: raise PWM or check load"},
        {LegCalibrationFailure::PartnerNotCurled, JointSensor::Hall, 0, 0,
         "FRKL partner not proven curled"},
    };
    for (const Expected &entry : expected)
    {
        FakeOutput out;
        printLegCalibrationFailureReason(out, entry.failure, "FRKL", entry.kind, true,
            entry.value, entry.value2, CAL_PWM_MAX);
        TEST_ASSERT_EQUAL_STRING(entry.text, out.text.c_str());
    }
}

// The CAL lines a hall run prints, nudges included, use only the event words
// the leader and the host know; none starts a reserved word by accident.
static void test_hall_run_lines_use_only_known_event_words()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;  // hip parked at its curl end: nudge and return
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
    const char *known[] = {"PASS ", "PASS_DONE ", "MOVE ", "END ", "SENSOR ", "HOME ", "WARN ", "JOINT "};
    for (const std::string &line : plant.output.lines())
    {
        TEST_ASSERT_EQUAL_INT(0, line.compare(0, 10, "CAL FRONT "));
        const std::string event = line.substr(10);
        bool matched = false;
        for (const char *word : known)
            matched = matched || event.compare(0, std::string(word).size(), word) == 0;
        TEST_ASSERT_TRUE_MESSAGE(matched, line.c_str());
        for (const char *reserved : {"FAIL", "ABORTED", "BUSY", "ERROR"})
            TEST_ASSERT_TRUE(event.compare(0, std::string(reserved).size(), reserved) != 0);
        if (event.compare(0, 5, "MOVE ") == 0)
        {
            const std::string direction = event.substr(event.rfind(' ') + 1);
            TEST_ASSERT_TRUE(direction == "curl" || direction == "uncurl" || direction == "nudge");
        }
    }
}

static void test_full_run_curls_then_sweeps_and_saves_hips_and_knees_only()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Idle),
        static_cast<int>(calibration.outcome()));
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_TRUE(calibration.isRunning());
    TEST_ASSERT_EQUAL_INT(CAL_DEFAULT_PWM, calibration.pwm());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_ALL, calibration.passes());
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);

    assertCompleted(calibration);
    const std::vector<std::string> expectedMoves = {
        "CAL FRONT MOVE FLKL curl", "CAL FRONT MOVE FLHL curl",
        "CAL FRONT MOVE FRKL curl", "CAL FRONT MOVE FRHL curl",
        "CAL FRONT MOVE FLKL curl", "CAL FRONT MOVE FLHL curl",
        "CAL FRONT MOVE FLHL uncurl", "CAL FRONT MOVE FLHL curl",
        "CAL FRONT MOVE FLKL uncurl", "CAL FRONT MOVE FLKL curl",
        "CAL FRONT MOVE FRKL curl", "CAL FRONT MOVE FRHL curl",
        "CAL FRONT MOVE FRHL uncurl", "CAL FRONT MOVE FRHL curl",
        "CAL FRONT MOVE FRKL uncurl", "CAL FRONT MOVE FRKL curl",
    };
    const std::vector<std::string> moves = moveLines(plant.output);
    TEST_ASSERT_EQUAL_UINT(expectedMoves.size(), moves.size());
    for (size_t i = 0; i < moves.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expectedMoves[i].c_str(), moves[i].c_str());

    const std::vector<std::string> lines = plant.output.lines();
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS SEEK pwm=128", lines.front().c_str());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS_DONE SWEEP saved", lines.back().c_str());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT PASS_DONE SEEK"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT PASS SWEEP pwm=128"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FLHL min=100 max=900"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FRKL min=100 max=900"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=900 travel=800"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL uncurl pot=100 travel=-800"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl pot=900 travel=400"));
    // A pot build prints no hall lines and never touches homes.
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT SENSOR"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT HOME"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT WARN"));
    TEST_ASSERT_TRUE(plant.setHomeJoints.empty());
    TEST_ASSERT_TRUE(plant.setSensorCalls.empty());

    TEST_ASSERT_EQUAL_INT(1, plant.saves);
    TEST_ASSERT_EQUAL_UINT(4, plant.storedJoints.size());
    for (uint8_t joint : {1, 2, 4, 5})
    {
        TEST_ASSERT_EQUAL_INT(100, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(900, plant.maxStop[joint]);
        TEST_ASSERT_EQUAL_INT(100, calibration.stagedMin(joint));
        TEST_ASSERT_EQUAL_INT(900, calibration.stagedMax(joint));
    }
    for (uint8_t joint : {0, 3})
    {
        TEST_ASSERT_EQUAL_INT(0, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(1023, plant.maxStop[joint]);
    }
    TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
    TEST_ASSERT_EQUAL_UINT(0x36, calibration.curledMask());
    TEST_ASSERT_EQUAL_STRING("", reasonOf(calibration).c_str());
}

static void test_ramp_climbs_from_zero_and_never_exceeds_requested_pwm()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, 83, 0));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Ramp),
        static_cast<int>(calibration.movePhase()));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMoveStage::Normal),
        static_cast<int>(calibration.moveStage()));
    TEST_ASSERT_EQUAL_UINT(2, calibration.currentJoint());
    TEST_ASSERT_TRUE(calibration.isCurrentMoveCurl());
    plant.advance(CAL_PWM_RAMP_INTERVAL_MS - 1);
    calibration.update(plant);
    TEST_ASSERT_TRUE(plant.drives.empty());
    plant.advance(1);
    calibration.update(plant);
    TEST_ASSERT_EQUAL_INT(CAL_PWM_RAMP_STEP, plant.drives.back().pwm);
    while (calibration.movePhase() == LegCalibrationMovePhase::Ramp)
    {
        plant.advance(CAL_PWM_RAMP_INTERVAL_MS);
        calibration.update(plant);
    }
    TEST_ASSERT_EQUAL_UINT(17, plant.drives.size());
    for (size_t i = 0; i + 1 < plant.drives.size(); ++i)
        TEST_ASSERT_EQUAL_INT(CAL_PWM_RAMP_STEP * static_cast<int>(i + 1), plant.drives[i].pwm);
    TEST_ASSERT_EQUAL_INT(83, plant.drives.back().pwm);
    TEST_ASSERT_EQUAL_INT(83, calibration.rampPwm());
    runToEnd(calibration, plant, 83);
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
}

static void test_hip_ramps_with_negative_pwm_to_curl()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MIN, 0));
    // Knee FLKL then hip FLHL: the hip's first drive is retract (-PWM).
    runToEnd(calibration, plant, CAL_PWM_MIN);
    bool sawHip = false;
    for (const Drive &drive : plant.drives)
    {
        if (drive.joint != 1) continue;
        TEST_ASSERT_TRUE(drive.pwm < 0);
        sawHip = true;
    }
    TEST_ASSERT_TRUE(sawHip);
    TEST_ASSERT_EQUAL_INT(100, plant.extension[1]);
    TEST_ASSERT_EQUAL_INT(900, plant.extension[2]);
    TEST_ASSERT_EQUAL_UINT(0x36, calibration.curledMask());
}

static void test_seek_only_never_saves_and_sweep_only_does()
{
    FakePlant seekPlant;
    LegCalibration seek;
    TEST_ASSERT_TRUE(seek.start(seekPlant, CAL_PASS_SEEK, CAL_PWM_MAX, 0));
    runToEnd(seek, seekPlant, CAL_PWM_MAX);
    assertCompleted(seek);
    TEST_ASSERT_EQUAL_INT(0, seekPlant.saves);
    TEST_ASSERT_TRUE(seekPlant.storedJoints.empty());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS_DONE SEEK", seekPlant.output.last().c_str());
    TEST_ASSERT_EQUAL_UINT(4, moveLines(seekPlant.output).size());

    FakePlant sweepPlant;
    LegCalibration sweep;
    TEST_ASSERT_TRUE(sweep.start(sweepPlant, CAL_PASS_SWEEP, CAL_PWM_MAX, 0));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS SWEEP pwm=150", sweepPlant.output.lines().front().c_str());
    runToEnd(sweep, sweepPlant, CAL_PWM_MAX);
    TEST_ASSERT_EQUAL_INT(1, sweepPlant.saves);
    TEST_ASSERT_EQUAL_UINT(12, moveLines(sweepPlant.output).size());
    TEST_ASSERT_FALSE(sweepPlant.output.has("CAL FRONT PASS_DONE SEEK"));
}

static void test_invalid_requests_and_options_do_not_start()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_ALL, CAL_PWM_MIN - 1, 0));
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_ALL, CAL_PWM_MAX + 1, 0));
    TEST_ASSERT_FALSE(calibration.start(plant, 0, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_FALSE(calibration.start(plant, 0x04 | CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0x02));
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0x80 | CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(plant.output.text.empty());
    TEST_ASSERT_EQUAL_INT(0, plant.stopAllCount);
    TEST_ASSERT_FALSE(calibration.isRunning());
    TEST_ASSERT_TRUE(isValidCalibrationRequest(CAL_PASS_SWEEP, CAL_PWM_MAX));
}

// The options reach the run; start() without options uses the shipped default.
static void test_options_pass_through_and_default()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_EQUAL_UINT(CAL_OPT_KNEE_NUDGE, calibration.options());
    TEST_ASSERT_TRUE(calibration.abort(plant, 'H'));
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_EQUAL_UINT(0, calibration.options());
    TEST_ASSERT_TRUE(calibration.abort(plant, 'H'));
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_UINT(CAL_DEFAULT_OPTIONS, calibration.options());
}

static void test_open_pot_fails_before_driving_only_when_proven_pot()
{
    FakePlant plant;
    plant.open[2] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_FALSE(calibration.isRunning());
    assertFailure(calibration, LegCalibrationFailure::PotNotConnected);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT FAIL FLKL pot not connected"));
    TEST_ASSERT_TRUE(plant.drives.empty());
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
    calibration.update(plant);
    TEST_ASSERT_TRUE(plant.drives.empty());
    TEST_ASSERT_EQUAL_STRING("FLKL pot not connected", reasonOf(calibration).c_str());

    // A hall-B line reads "open" to the probe: it is never consulted for a joint
    // not proven Pot (the run-2 false "pot not connected").
    FakePlant hall;
    hall.useHall();
    for (bool &value : hall.open) value = true;
    LegCalibration seek;
    TEST_ASSERT_TRUE(seek.start(hall, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(seek, hall, CAL_DEFAULT_PWM);
    assertCompleted(seek);
    TEST_ASSERT_EQUAL_INT(0, hall.output.count("CAL FRONT FAIL"));
}

static void test_move_without_end_times_out_and_stops()
{
    FakePlant plant;
    plant.low[2] = -100000;
    plant.high[2] = 100000;  // knee never reaches an end
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    uint32_t lastDriveMs = 0;
    while (calibration.isRunning())
    {
        plant.advance(10);
        calibration.update(plant);
        if (plant.pwm[2] != 0) lastDriveMs = plant.now;
    }
    assertFailure(calibration, LegCalibrationFailure::NoEndOfTravel);
    TEST_ASSERT_TRUE(lastDriveMs < CAL_MOVE_TIMEOUT_MS);
    TEST_ASSERT_EQUAL_UINT32(CAL_MOVE_TIMEOUT_MS, plant.now);
    TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT FAIL FLKL curl: no end of travel in 120 s"));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
}

static void test_uncurl_that_barely_moves_fails()
{
    FakePlant plant;
    plant.stuck[1] = true;  // FLHL never moves
    plant.extension[1] = 100;
    plant.minStop[1] = 100;  // measured retracted limit: the hip rests at its curl end
    plant.limitsValid[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, 90, 0));
    runToEnd(calibration, plant, 90);
    assertFailure(calibration, LegCalibrationFailure::BarelyMoved);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=100 travel=0"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=100 travel=0"));
    TEST_ASSERT_TRUE(plant.output.has(
        "CAL FRONT FAIL FLHL barely moved (0 counts) at PWM 90: check pot wiring or raise speed"));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
    TEST_ASSERT_TRUE(plant.storedJoints.empty());
    // A pot failure leaves homes alone.
    TEST_ASSERT_TRUE(plant.clearHomeJoints.empty());
}

// The pot direction is fixed by the PWM sign: a reversed pot (or motor) fails
// instead of being learned and saved reversed.
static void test_reversed_pot_fails_pot_wrong_way()
{
    FakePlant plant;
    plant.reversed[2] = true;
    plant.speed = 4;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, 0));
    uint32_t settleMs = 0;
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Settle;
    });
    settleMs = plant.now;
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    // Ramp 520 ms (128 / 5 per 20 ms) + grace 1500 ms: the first window already
    // moves backwards.
    TEST_ASSERT_EQUAL_UINT32(3020, settleMs);
    assertFailure(calibration, LegCalibrationFailure::PotWrongWay);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL pot moves the wrong way: check motor or pot end wires",
        plant.output.last().c_str());
    TEST_ASSERT_FALSE(drovePartnerAfter(plant, 0, 1));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT WARN"));
}

static void test_pot_window_follows_the_drive_direction()
{
    FakePlant plant;
    // Knee pot: rises 10 counts per second until 4 s after the start, then falls.
    plant.potScript = [](uint8_t joint, uint32_t now) -> int {
        if (joint != 2) return 500;
        return now <= 4000 ? 500 + static_cast<int>(now / 100) : 540 - static_cast<int>((now - 4000) / 100);
    };
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    uint32_t endMs = 0;
    while (calibration.isRunning() && calibration.currentJoint() == 2)
    {
        plant.advance(10);
        calibration.update(plant);
        if (endMs == 0 && calibration.movePhase() == LegCalibrationMovePhase::Settle)
            endMs = plant.now;
    }
    // Ramp 520 ms + grace 1500 ms, windows end at 3020 and 4020 (rising), 5020 falls.
    TEST_ASSERT_EQUAL_UINT32(5020, endMs);
}

static void test_abort_stops_everything_and_saves_nothing()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_FALSE(calibration.abort(plant, 'H'));
    TEST_ASSERT_TRUE(plant.output.text.empty());
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    for (int i = 0; i < 300; ++i)
    {
        plant.advance(10);
        calibration.update(plant);
    }
    TEST_ASSERT_EQUAL_INT(1, plant.drivenCount());
    TEST_ASSERT_TRUE(calibration.abort(plant, 'J'));
    TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
    TEST_ASSERT_FALSE(calibration.isRunning());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Aborted),
        static_cast<int>(calibration.outcome()));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT ABORTED J", plant.output.last().c_str());
    const size_t drivesAtAbort = plant.drives.size();
    for (int i = 0; i < 300; ++i)
    {
        plant.advance(10);
        calibration.update(plant);
    }
    TEST_ASSERT_EQUAL_UINT(drivesAtAbort, plant.drives.size());
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
    TEST_ASSERT_EQUAL_STRING("ABORTED J", reasonOf(calibration).c_str());
    TEST_ASSERT_FALSE(calibration.abort(plant, 'H'));

    calibration.reset();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationOutcome::Idle),
        static_cast<int>(calibration.outcome()));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationFailure::None),
        static_cast<int>(calibration.failure()));
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_EQUAL_UINT(0, calibration.currentStep());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SEEK, calibration.currentPass());
}

// An abort keeps hall homes: the counts keep tracking while the board is up.
static void test_abort_keeps_hall_homes()
{
    FakePlant plant;
    LegCalibration calibration;
    seekHall(calibration, plant);
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, 0));
    runUntilStep(calibration, plant, 2);  // hip uncurling
    for (int i = 0; i < 50; ++i) tick(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(plant.pwm[1] > 0);
    TEST_ASSERT_TRUE(calibration.abort(plant, 'H'));
    TEST_ASSERT_TRUE(plant.homed[1]);
    TEST_ASSERT_TRUE(plant.homed[2]);
    TEST_ASSERT_TRUE(plant.clearHomeJoints.empty());
}

// Review finding: a hip that cannot lift itself back (PWM too low against
// gravity, or held at its extend-end switch) never leaves the uncurl end.
static void test_sweep_curl_back_that_never_leaves_the_uncurl_end_fails()
{
    FakePlant plant;
    plant.extension[1] = 100;  // hip curled (retracted)
    plant.extension[2] = 900;  // knee curled (extended)
    plant.minStop[1] = 100;    // measured limits from an earlier calibration
    plant.maxStop[2] = 900;
    plant.limitsValid[1] = plant.limitsValid[2] = true;
    plant.stallSign[1] = -1;   // the hip cannot retract (curl) at this PWM
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_PWM_MIN, 0));
    runToEnd(calibration, plant, CAL_PWM_MIN);

    assertFailure(calibration, LegCalibrationFailure::NotAtCurlEnd);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=900 travel=800"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=900 travel=0"));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL did not reach its curl end (pot=900)",
        plant.output.last().c_str());
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    for (const Drive &drive : plant.drives)
        TEST_ASSERT_FALSE(drive.joint == 2 && drive.pwm < 0);  // knee never uncurled
    TEST_ASSERT_FALSE(calibration.isJointCurled(1));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
    TEST_ASSERT_TRUE(plant.storedJoints.empty());
    TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
    TEST_ASSERT_EQUAL_STRING("FLHL did not reach its curl end (pot=900)", reasonOf(calibration).c_str());
}

// A curl back that moves but stops short of where the curl end read before the
// uncurl (stalled part way under load) fails too, before the knee moves.
static void test_sweep_curl_back_that_stops_part_way_fails()
{
    FakePlant plant;
    plant.speed = 4;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, 0));
    runUntilStep(calibration, plant, 3);  // hip uncurled to 900, now curling back
    TEST_ASSERT_TRUE(calibration.isRunning());
    TEST_ASSERT_EQUAL_UINT(1, calibration.currentJoint());
    TEST_ASSERT_TRUE(calibration.isCurrentMoveCurl());
    plant.low[1] = 131;  // the rod now stops 31 counts short of its curl end (100)
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);

    assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=131 travel=-769"));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL stopped 31 counts short of its curl end",
        plant.output.last().c_str());
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);

    // 30 short is within the tolerance: the pot pass completes and saves.
    FakePlant within;
    within.speed = 4;
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(within, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, 0));
    runUntilStep(other, within, 3);
    within.low[1] = 130;
    runToEnd(other, within, CAL_DEFAULT_PWM);
    assertCompleted(other);
    TEST_ASSERT_TRUE(within.output.has("CAL FRONT JOINT FLHL min=130 max=900"));
}

// Same for the knee's curl back (step 5): the knee stays uncurled and the run
// stops before leg B or any save.
static void test_sweep_knee_curl_back_that_stalls_fails()
{
    FakePlant plant;
    plant.speed = 4;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, 0));
    runUntilStep(calibration, plant, 5);  // knee uncurled to 100, now curling back
    TEST_ASSERT_EQUAL_UINT(2, calibration.currentJoint());
    plant.stallSign[2] = 1;  // cannot extend (curl) any more
    const size_t drivesAtStall = plant.drives.size();
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);

    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT FAIL FLKL did not reach its curl end (pot=100)"));
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT JOINT FLKL min=100 max=100"));
    TEST_ASSERT_FALSE(drovePartnerAfter(plant, drivesAtStall, 4));
    TEST_ASSERT_FALSE(drovePartnerAfter(plant, drivesAtStall, 5));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
}

// A mid-stroke knee that does not move at all (motor dead zone under load) is
// not taken as curled: a strict run (options 0) fails it, and the hip never moves.
static void test_knee_curl_that_never_moves_fails_no_motion()
{
    FakePlant plant;
    plant.stuck[2] = true;  // knee mid-stroke at 500, no measured limits
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_PWM_MIN, 0));
    runToEnd(calibration, plant, CAL_PWM_MIN);

    assertFailure(calibration, LegCalibrationFailure::NoMotion);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry",
        plant.output.last().c_str());
    for (const Drive &drive : plant.drives)
        TEST_ASSERT_EQUAL_UINT(2, drive.joint);  // only the knee was ever driven
    TEST_ASSERT_FALSE(calibration.isJointCurled(2));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);

    // With no motor current evidence the reason says so.
    FakePlant dead;
    dead.stuck[2] = true;
    dead.currentAbsent[2] = true;
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(dead, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(other, dead, CAL_DEFAULT_PWM);
    assertFailure(other, LegCalibrationFailure::NoMotorCurrent);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL no motor current at PWM 128: check motor wiring",
        dead.output.last().c_str());
}

// A pot joint resting at its curl end (no motion) is accepted at a measured
// limit or at a curl end read on an earlier accepted curl, even in a later
// start(); never at the bare ADC rails, and never far past the known end.
static void test_pot_curl_without_motion_needs_a_proven_curl_end()
{
    {
        FakePlant plant;  // uncalibrated: joints rest near the rails, no limits
        for (bool &value : plant.stuck) value = true;
        plant.extension[1] = 5;
        plant.extension[2] = 1018;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::NoMotion);
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl pot=1018 travel=0"));
        TEST_ASSERT_FALSE(drovePartnerAfter(plant, 0, 1));
    }
    {
        FakePlant plant;  // measured limits: accepted within the tolerance
        for (bool &value : plant.stuck) value = true;
        plant.limitsValid[1] = plant.limitsValid[2] = true;
        plant.minStop[1] = 100;
        plant.maxStop[2] = 900;
        plant.extension[1] = 100 + CAL_CURL_END_TOLERANCE_COUNTS;
        plant.extension[2] = 900 + CAL_CURL_END_TOLERANCE_COUNTS;  // deeper, within tolerance
        plant.high[2] = 1000;
        plant.extension[5] = 900 - CAL_CURL_END_TOLERANCE_COUNTS - 1;  // FRKL: too far
        plant.limitsValid[5] = true;
        plant.maxStop[5] = 900;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl pot=930 travel=0"));
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=130 travel=0"));
        TEST_ASSERT_TRUE(calibration.isJointCurled(1));
        TEST_ASSERT_TRUE(calibration.isJointCurled(2));
        assertFailure(calibration, LegCalibrationFailure::NoMotion);
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FRKL curl pot=869 travel=0"));
    }
    {
        FakePlant plant;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);  // curls read 900 (knees), 100 (hips)
        assertCompleted(calibration);
        // Next run: the joints rest there (or nearby) and do not move; the curl
        // ends read on the accepted curls vouch for them (no measured limits).
        for (bool &value : plant.stuck) value = true;
        plant.extension[2] = 900 - CAL_CURL_END_TOLERANCE_COUNTS;
        plant.low[1] = 0;
        plant.extension[1] = 100 - CAL_CURL_END_TOLERANCE_COUNTS;  // deeper, within tolerance
        plant.high[5] = 1000;
        plant.extension[5] = 931;  // past its end by more than the tolerance
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl pot=870 travel=0"));
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=70 travel=0"));
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FRKL curl pot=931 travel=0"));
        assertFailure(calibration, LegCalibrationFailure::NoMotion);
        // reset() forgets the remembered curl ends.
        calibration.reset();
        plant.extension[2] = 900;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry",
            plant.output.last().c_str());
    }
}

// A pot curl that moves after a proven curl end: short by 31 fails, deeper by
// more than the tolerance warns and moves the end.
static void test_pot_curl_with_motion_against_its_proven_end()
{
    FakePlant plant;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);  // knee ends 900, hip 100
    plant.extension[2] = 500;
    plant.high[2] = 869;  // 31 short
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 31 counts short of its curl end",
        plant.output.last().c_str());

    plant.extension[2] = 500;
    plant.high[2] = 931;  // deeper by 31
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT WARN FLKL home moved 31 counts"));
    TEST_ASSERT_TRUE(plant.setHomeJoints.empty());
    // The deeper end is the reference now: 901 is 30 short of 931, accepted.
    plant.extension[2] = 500;
    plant.high[2] = 901;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT WARN"));
}

// Every saved hip/knee spans at least CAL_MIN_TRAVEL_COUNTS: a curl back that
// ends at the curl end but less than that from the uncurl end fails.
static void test_sweep_never_saves_a_span_below_the_minimum_travel()
{
    FakePlant plant;
    plant.high[1] = 100 + CAL_MIN_TRAVEL_COUNTS;  // the hip uncurls exactly the minimum
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    while (calibration.isRunning() &&
           !(calibration.currentPass() == CAL_PASS_SWEEP && calibration.currentStep() >= 3))
    {
        plant.advance(10);
        calibration.update(plant);
    }
    TEST_ASSERT_TRUE(calibration.isRunning());  // SWEEP, hip curling back from 130
    plant.low[1] = 101;  // back within tolerance of 100, but only 29 from 130
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=130 travel=30"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=101 travel=-29"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT FAIL FLHL did not reach its curl end (pot=101)"));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
}

// ---------------------------------------------------------------- hall joints

// All-hall board: Unknown joints become Hall on their first 12 edges, curls
// home them, and SWEEP saves canonical ranges: a hip's fixed sweep
// [0, CAL_HIP_SWEEP_EDGES] (its rod reaches 1000), a knee's stroke [-S, 0].
static void test_hall_full_run_homes_and_saves_canonical_strokes()
{
    FakePlant plant;
    plant.useHall();
    plant.speed = 4;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    int furthestHip[6] = {};
    while (calibration.isRunning())
    {
        tick(calibration, plant, CAL_DEFAULT_PWM);
        for (uint8_t hip : {1, 4})
            if (plant.extension[hip] > furthestHip[hip]) furthestHip[hip] = plant.extension[hip];
    }
    assertCompleted(calibration);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, furthestHip[1]);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, furthestHip[4]);

    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT SENSOR FLKL hall"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLKL hall"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FRHL hall"));
    TEST_ASSERT_EQUAL_INT(4, plant.output.count("CAL FRONT SENSOR"));
    // Knee curl ends are the leg's mechanical stop, hip curl ends and knee
    // uncurl ends the actuator's limit switch (the end detector's END tokens).
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLKL curl hall=1000 travel=500", "stop"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-1000"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLHL curl hall=0 travel=-500", "switch"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    // SWEEP after SEEK: steps 0 and 1 rest at their homes (a press: no
    // running phase, a plain stall, no token), no new HOME.
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=0 travel=0"));
    TEST_ASSERT_EQUAL_INT(4, plant.output.count("CAL FRONT HOME"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl hall=700 travel=700"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLHL curl hall=0 travel=-700", "switch"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FLHL min=0 max=700 edges (fixed sweep)"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FRHL min=0 max=700 edges (fixed sweep)"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLKL uncurl hall=-1000 travel=-1000", "switch"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLKL curl hall=0 travel=1000", "stop"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FLKL min=-1000 max=0 edges"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FRKL min=-1000 max=0 edges"));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS_DONE SWEEP saved", plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(1, plant.saves);
    for (uint8_t joint : {1, 4})
    {
        TEST_ASSERT_EQUAL_INT(0, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, plant.maxStop[joint]);
    }
    for (uint8_t joint : {2, 5})
    {
        TEST_ASSERT_EQUAL_INT(-1000, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(0, plant.maxStop[joint]);
    }
    for (uint8_t joint : {0, 3})
    {
        TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(plant.kind[joint]));
        TEST_ASSERT_EQUAL_INT(0, plant.maxStop[joint]);
    }
    for (uint8_t joint : {1, 2, 4, 5})
        TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Hall), static_cast<int>(plant.kind[joint]));
    TEST_ASSERT_TRUE(plant.clearHomeJoints.empty());
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT WARN"));
}

// Hall evidence counts in every driven phase (a quadrature build's hall drive
// runs Ramp, then Detect).
static void test_hall_evidence_in_ramp_and_detect()
{
    const LegCalibrationMovePhase phases[] = {LegCalibrationMovePhase::Ramp, LegCalibrationMovePhase::Detect};
    for (LegCalibrationMovePhase phase : phases)
    {
        FakePlant plant;
        plant.useHall();
        for (bool &value : plant.stuck) value = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.movePhase() == phase; });
        plant.countBias[2] = -11;  // knee extends: counts down
        tick(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT SENSOR"));
        plant.countBias[2] = -12;
        tick(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT SENSOR FLKL hall", plant.output.last().c_str());
        TEST_ASSERT_EQUAL_INT(static_cast<int>(phase), static_cast<int>(calibration.movePhase()));
        TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Hall), static_cast<int>(plant.kind[2]));
        TEST_ASSERT_TRUE(calibration.isRunning());
        // Confirmed again later in the run: printed once.
        plant.countBias[2] = -40;
        tick(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT SENSOR"));
    }
}

static void test_hall_wrong_way_fails_in_ramp_and_detect()
{
    const LegCalibrationMovePhase phases[] = {LegCalibrationMovePhase::Ramp, LegCalibrationMovePhase::Detect};
    for (LegCalibrationMovePhase phase : phases)
    {
        FakePlant plant;
        plant.useHall();
        for (bool &value : plant.stuck) value = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.movePhase() == phase; });
        plant.countBias[2] = 11;
        tick(calibration, plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(calibration.isRunning());
        plant.countBias[2] = 12;
        tick(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::WrongDirection);
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLKL hall counts the wrong way: check motor or hall A/B wiring",
            plant.output.last().c_str());
        TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
        TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(plant.kind[2]));
    }
    // A proven hall joint that fails loses its home.
    FakePlant plant;
    LegCalibration calibration;
    seekHall(calibration, plant);
    plant.extension[2] = 500;
    plant.hallReversed[2] = true;  // the hall now counts the wrong way
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::WrongDirection);
    TEST_ASSERT_FALSE(plant.homed[2]);
    TEST_ASSERT_TRUE(plant.homeLost[2]);
    TEST_ASSERT_EQUAL_UINT(1, plant.clearHomeJoints.size());
}

// Unknown and Hall share the hall frame: classifying during Detect does not
// restart the detector. A stuck Unknown knee that jumps 12 edges at 1.0 s is a
// hall; its unarmed curl, having made 12 edges, ends Still once 1000 ms pass
// with no further edge (a slow joint's pause is not its end).
static void test_unknown_to_hall_in_detect_keeps_the_detector()
{
    FakePlant plant;
    plant.useHall();
    for (bool &value : plant.stuck) value = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Detect;
    });
    TEST_ASSERT_EQUAL_UINT32(520, plant.now);  // ramp: 26 steps of 5 to 128
    TEST_ASSERT_TRUE(calibration.isDetectingEnd());
    while (plant.now < 1000) tick(calibration, plant, CAL_DEFAULT_PWM);
    plant.countBias[2] = -12;
    tick(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT SENSOR FLKL hall", plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Detect), static_cast<int>(calibration.movePhase()));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Settle;
    });
    // The jump lands in the sample closing at 1050; 20 samples later: 2050.
    TEST_ASSERT_EQUAL_UINT32(2050, plant.now);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(CalibrationEnd::Still), static_cast<int>(calibration.lastEnd()));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() != 2; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=512 travel=12"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-512"));
}

// A proven Pot joint that counts edges becomes Hall mid-window ("was pot"),
// and its window restarts in the hall frame.
static void test_pot_to_hall_in_windows_restarts_the_window()
{
    FakePlant plant;
    plant.useHall();
    for (bool &value : plant.stuck) value = true;
    plant.kind[2] = JointSensor::Pot;
    plant.hallRod[2] = false;
    plant.potScript = [](uint8_t joint, uint32_t now) -> int {
        return joint == 2 ? 300 + static_cast<int>(now / 100) : 500;  // a pot moving 10/s
    };
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Windows;
    });
    for (int i = 0; i < 150; ++i) tick(calibration, plant, CAL_DEFAULT_PWM);  // 2nd window, +500 ms
    plant.countBias[2] = -12;
    tick(calibration, plant, CAL_DEFAULT_PWM);
    const uint32_t changeMs = plant.now;
    TEST_ASSERT_EQUAL_STRING("CAL FRONT SENSOR FLKL hall (was pot)", plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Hall), static_cast<int>(plant.kind[2]));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Windows),
        static_cast<int>(calibration.movePhase()));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Settle;
    });
    TEST_ASSERT_EQUAL_UINT32(changeMs + CAL_STALL_WINDOW_MS, plant.now);
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() != 2; });
    // Accepted as a fresh hall: 12 edges, then stalled; homed.
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-12"));
}

// An unarmed hall curl (too slow to arm: under 20 edges/s) ends Still once it
// makes fewer than CAL_HALL_STALL_MIN_EDGES (3) edges in CAL_END_STILL_WINDOW_MS,
// not before CAL_END_STILL_MIN_DRIVE_MS of drive.
static void test_hall_still_threshold_three_edges()
{
    for (int perWindow : {2, 3})
    {
        FakePlant plant;
        plant.useHall();
        for (bool &value : plant.stuck) value = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        uint32_t settleMs = 0;
        for (int i = 0; i < 400 && settleMs == 0; ++i)
        {
            if (i % 50 == 25) plant.countBias[2] -= perWindow;  // knee extends: counts down
            tick(calibration, plant, CAL_DEFAULT_PWM);
            if (calibration.movePhase() == LegCalibrationMovePhase::Settle) settleMs = plant.now;
        }
        TEST_ASSERT_FALSE(calibration.endDetector().isArmed());
        if (perWindow < CAL_HALL_STALL_MIN_EDGES)
            TEST_ASSERT_EQUAL_UINT32(CAL_END_STILL_MIN_DRIVE_MS + 50, settleMs);  // sample 20 closes at 1050
        else
            TEST_ASSERT_EQUAL_UINT32(0, settleMs);  // 4 s, still moving
    }
}

// Curl against a home proven this boot: 40 edges short is accepted, 41 fails;
// deeper by more than 40 warns and re-homes; deeper within 40 re-homes quietly.
static void test_hall_curl_against_its_home()
{
    const struct
    {
        int kneeEnd;
        bool fails;
        bool warns;
        bool rehomes;
    } cases[] = {
        {959, true, false, false},
        {960, false, false, false},
        {1040, false, false, true},
        {1041, false, true, true},
    };
    for (const auto &entry : cases)
    {
        FakePlant plant;
        LegCalibration calibration;
        seekHall(calibration, plant);
        const size_t homesBefore = plant.setHomeJoints.size();
        plant.extension[2] = 900;
        plant.high[2] = entry.kneeEnd;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() != 2; });
        if (entry.fails)
        {
            assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 41 edges short of its curl end",
                plant.output.last().c_str());
            TEST_ASSERT_TRUE(plant.homeLost[2]);
            continue;
        }
        TEST_ASSERT_TRUE(calibration.isRunning());
        TEST_ASSERT_TRUE(calibration.isJointCurled(2));
        TEST_ASSERT_EQUAL(entry.warns, plant.output.has("CAL FRONT WARN FLKL home moved 41 edges"));
        TEST_ASSERT_EQUAL_UINT(homesBefore + (entry.rehomes ? 1 : 0), plant.setHomeJoints.size());
        TEST_ASSERT_EQUAL_INT(entry.rehomes ? -entry.kneeEnd : -1000, plant.home[2]);
    }
}

// No motion within the tolerance short of a home proven this boot is accepted
// (no nudge, no re-home); more than the tolerance short fails ShortOfCurlEnd
// like a curl that moved; deeper re-homes (with a WARN beyond the tolerance).
static void test_hall_curl_without_motion_at_its_home()
{
    FakePlant plant;
    LegCalibration calibration;
    seekHall(calibration, plant);
    plant.extension[2] = 1000 - CAL_HALL_CURL_END_TOLERANCE_EDGES;  // within tolerance, stuck
    plant.stuck[2] = true;
    plant.extension[1] = 0;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 2; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=-40 travel=0"));
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_EQUAL_UINT(4, plant.setHomeJoints.size());  // only the first SEEK's homes

    // 41 short and stuck: not at the end.
    FakePlant far;
    LegCalibration other;
    seekHall(other, far);
    far.extension[2] = 959;
    far.stuck[2] = true;
    TEST_ASSERT_TRUE(other.start(far, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(other, far, CAL_DEFAULT_PWM);
    assertFailure(other, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 41 edges short of its curl end",
        far.output.last().c_str());
    TEST_ASSERT_TRUE(far.homeLost[2]);
    // A hall joint pushed deeper than its home without moving is accepted and
    // re-homed there; beyond the tolerance it warns.
    for (int deeper : {CAL_HALL_CURL_END_TOLERANCE_EDGES, 100})
    {
        FakePlant deep;
        LegCalibration third;
        seekHall(third, deep);
        deep.countBias[2] = -deeper;  // reads deeper
        deep.stuck[2] = true;
        TEST_ASSERT_TRUE(third.start(deep, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(third, deep, CAL_DEFAULT_PWM, [&]() { return third.currentStep() >= 1; });
        TEST_ASSERT_TRUE(third.isJointCurled(2));
        TEST_ASSERT_EQUAL_UINT(5, deep.setHomeJoints.size());
        TEST_ASSERT_EQUAL_INT(-1000 - deeper, deep.home[2]);
        TEST_ASSERT_EQUAL(deeper > CAL_HALL_CURL_END_TOLERANCE_EDGES,
            deep.output.has("CAL FRONT WARN FLKL home moved 100 edges"));
        TEST_ASSERT_FALSE(deep.output.has("CAL FRONT MOVE FLKL nudge"));
    }
}

// The default path (knee nudge on): a hip or knee homed this boot and now held
// more than the tolerance short of its home by something on its curl side shows
// no motion. It fails ShortOfCurlEnd before any nudge, so no return re-homes it
// at the obstacle and its partner never moves next to it.
static void test_curl_without_motion_short_of_a_home_proven_this_boot_fails()
{
    for (uint8_t joint : {static_cast<uint8_t>(1), static_cast<uint8_t>(2)})
    {
        FakePlant plant;
        LegCalibration calibration;
        seekHall(calibration, plant);
        const bool hip = joint == 1;
        plant.extension[joint] = hip ? 100 : 900;  // 100 edges uncurled
        plant.stallSign[joint] = hip ? -1 : +1;    // blocked toward its curl end only
        const size_t drivesBefore = plant.drives.size();
        const size_t homesBefore = plant.setHomeJoints.size();
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
        const std::string expected = std::string("CAL FRONT FAIL ") + plant.names[joint] +
            " stopped 100 edges short of its curl end";
        TEST_ASSERT_EQUAL_STRING(expected.c_str(), plant.output.last().c_str());
        TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT MOVE FLHL nudge"));
        TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT MOVE FLKL nudge"));
        TEST_ASSERT_EQUAL_UINT(homesBefore, plant.setHomeJoints.size());
        TEST_ASSERT_TRUE(plant.homeLost[joint]);
        TEST_ASSERT_FALSE(drovePartnerAfter(plant, drivesBefore, hip ? 5 : 1));
    }
}

// A hip parked at its curl end gets the verification nudge: out until 24 edges,
// then a full curl back that proves the end and homes it.
static void test_hip_nudge_by_edges_then_return_homes_it()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;  // parked curled (limit switch: no edges)
    plant.moveEvery[1] = 3;  // 33 edges/s: the nudge outlasts the 520 ms ramp
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    uint32_t nudgeStart = 0;
    uint32_t nudgeStop = 0;
    LegCalibrationMovePhase phaseAtStop = LegCalibrationMovePhase::Ramp;
    LegCalibrationMovePhase lastPhase = LegCalibrationMovePhase::Ramp;
    while (calibration.isRunning() && calibration.currentStep() < 2)
    {
        lastPhase = calibration.movePhase();
        tick(calibration, plant, CAL_DEFAULT_PWM);
        if (calibration.moveStage() == LegCalibrationMoveStage::NudgeOut)
        {
            if (nudgeStart == 0) nudgeStart = plant.now;
            if (nudgeStop == 0 && calibration.movePhase() == LegCalibrationMovePhase::Settle)
            {
                nudgeStop = plant.now;
                phaseAtStop = lastPhase;
            }
        }
    }
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl hall=0 travel=0 pot=12"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Nudge), static_cast<int>(phaseAtStop));
    TEST_ASSERT_TRUE(nudgeStop - nudgeStart < CAL_HIP_NUDGE_MAX_MS);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLHL hall"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL nudge hall=24 travel=24"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl hall=0 travel=-24"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));
    TEST_ASSERT_EQUAL_INT(0, plant.extension[1]);
}

// Slow hip (PWM 40, 10 edges/s): the nudge ends on its 2 s cap with 12 or more
// edges, which is evidence; the return proves the end.
static void test_hip_nudge_by_timeout_with_evidence()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.moveEvery[1] = 10;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MIN, 0));
    runUntil(calibration, plant, CAL_PWM_MIN, [&]() {
        return calibration.moveStage() == LegCalibrationMoveStage::Return;
    });
    const long out = plant.output.valueAfter("CAL FRONT END FLHL nudge", "travel");
    TEST_ASSERT_TRUE(out >= CAL_HALL_DETECT_EDGES && out < CAL_HIP_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT SENSOR FLHL hall"));
    runUntil(calibration, plant, CAL_PWM_MIN, [&]() { return calibration.currentStep() >= 2; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    TEST_ASSERT_TRUE(calibration.isRunning());
}

// No edges at all (hall A dead): the nudge ends on its cap without evidence,
// the return still drives the hip back blind, then the pass fails.
static void test_hip_nudge_without_evidence_returns_blind_then_fails()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.hallDead[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    // Out until the 2 s cap (ramp included), then driven back before the FAIL.
    uint32_t firstOut = 0, lastOut = 0, firstBack = 0, lastBack = 0;
    while (calibration.isRunning())
    {
        tick(calibration, plant, CAL_DEFAULT_PWM);
        if (plant.pwm[1] > 0)
        {
            if (firstOut == 0) firstOut = plant.now;
            lastOut = plant.now;
        }
        if (plant.pwm[1] < 0 && lastOut != 0)
        {
            if (firstBack == 0) firstBack = plant.now;
            lastBack = plant.now;
        }
    }
    assertFailure(calibration, LegCalibrationFailure::NoFeedback);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLHL no feedback: no hall edges, pot not moving (check sensor cable)",
        plant.output.last().c_str());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL nudge hall=0 travel=0 pot=12"));
    TEST_ASSERT_TRUE(firstOut > 0 && firstBack > lastOut);
    // Driven out from the first ramp step until the 2 s cap from nudge start.
    TEST_ASSERT_EQUAL_UINT32(CAL_HIP_NUDGE_MAX_MS - CAL_PWM_RAMP_INTERVAL_MS - 10, lastOut - firstOut);
    // The blind return drives CAL_STALL_WINDOW_MS longer than the nudge did
    // (that is the unarmed return's minimum: hips curl back slower than they
    // uncurl), and not much longer.
    TEST_ASSERT_TRUE_MESSAGE(lastBack - firstBack >= lastOut - firstOut + CAL_STALL_WINDOW_MS,
                             std::to_string(lastBack - firstBack).c_str());
    TEST_ASSERT_TRUE_MESSAGE(lastBack - firstBack <= lastOut - firstOut + CAL_STALL_WINDOW_MS + 100,
                             std::to_string(lastBack - firstBack).c_str());
    TEST_ASSERT_EQUAL_INT(0, plant.extension[1]);  // driven back to its curl end
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT MOVE FRKL curl"));
}

// A hip loaded as Hall (not homed this boot) whose hall shows no edges: the
// nudge has no evidence even though the kind is Hall, and the return fails.
static void test_stored_hall_hip_nudge_without_edges_fails()
{
    FakePlant plant;
    plant.useHall();
    plant.kind[1] = JointSensor::Hall;
    plant.extension[1] = 0;
    plant.hallDead[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL nudge hall=0 travel=0"));
    assertFailure(calibration, LegCalibrationFailure::NoFeedback);
    TEST_ASSERT_EQUAL_INT(0, plant.extension[1]);
}

// Without quadrature (Rev 1 counts carry no direction) the counts never decide:
// a joint the plant still reports Unknown nudges to its time cap, and with no
// evidence the return fails.
static void test_nudge_ignores_counts_without_quadrature()
{
    FakePlant plant;
    plant.kind[1] = JointSensor::Unknown;
    plant.hallRod[1] = true;
    plant.low[1] = 0;
    plant.extension[1] = 0;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    uint32_t firstOut = 0;
    uint32_t lastOut = 0;
    while (calibration.isRunning())
    {
        tick(calibration, plant, CAL_DEFAULT_PWM);
        if (plant.pwm[1] > 0)
        {
            if (firstOut == 0) firstOut = plant.now;
            lastOut = plant.now;
        }
    }
    TEST_ASSERT_EQUAL_UINT32(CAL_HIP_NUDGE_MAX_MS - CAL_PWM_RAMP_INTERVAL_MS - 10, lastOut - firstOut);
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT SENSOR"));
    assertFailure(calibration, LegCalibrationFailure::NoFeedback);
}

// A pot hip on a Rev 3 slot nudged while not yet known, its pot wired
// backwards: PotWrongWay at the nudge end, and no return is driven.
static void test_pot_wrong_way_at_the_nudge_end()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[1] = false;
    plant.reversed[1] = true;
    plant.low[1] = 100;
    plant.high[1] = 900;
    plant.extension[1] = 100;
    plant.moveEvery[1] = 2;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::PotWrongWay);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL pot moves the wrong way: check motor or pot end wires",
        plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT MOVE FLHL curl"));
    bool out = false;
    for (const Drive &drive : plant.drives)
    {
        if (drive.joint != 1) continue;
        if (drive.pwm > 0) out = true;
        else TEST_ASSERT_FALSE(out);  // never driven back after the nudge
    }
    TEST_ASSERT_TRUE(out);
    TEST_ASSERT_TRUE(plant.extension[1] > 100);
}

// The current evidence of a dead motor is taken at the end of the nudge drive.
static void test_hip_nudge_no_motor_current_from_the_nudge()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.stuck[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.moveStage() == LegCalibrationMoveStage::NudgeOut;
    });
    plant.currentAbsent[1] = true;
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.moveStage() == LegCalibrationMoveStage::Return;
    });
    plant.currentAbsent[1] = false;  // the return drive shows nothing new
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::NoMotorCurrent);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL no motor current at PWM 128: check motor wiring",
        plant.output.last().c_str());
}

// The return must move (12 edges) and end no more than 40 edges short of where
// the nudge started.
static void test_hip_nudge_return_short_or_without_motion_fails()
{
    for (int returnEnd : {40, 41})
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;
        plant.moveEvery[1] = 2;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return calibration.moveStage() == LegCalibrationMoveStage::NudgeOut &&
                   calibration.movePhase() == LegCalibrationMovePhase::Settle;
        });
        plant.extension[1] += 60;  // pushed out further while settling
        plant.low[1] = returnEnd;
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 2; });
        if (returnEnd == 41)
        {
            assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL stopped 41 edges short of its curl end",
                plant.output.last().c_str());
            TEST_ASSERT_FALSE(plant.homed[1]);
        }
        else
        {
            TEST_ASSERT_TRUE(calibration.isRunning());
            TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=-40"));
        }
    }
    // Stuck on the way back: no motion.
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.moveStage() == LegCalibrationMoveStage::Return;
    });
    plant.stuck[1] = true;
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::NoCurlBack);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL curled back only 0 edges after its nudge",
        plant.output.last().c_str());
    TEST_ASSERT_TRUE(plant.homeLost[1] == false);  // never homed: nothing to lose
    TEST_ASSERT_FALSE(plant.homed[1]);
}

// Hall evidence and wrong way during the nudge and the return.
static void test_hall_wrong_way_in_nudge_and_return()
{
    for (int stage = 0; stage < 2; ++stage)
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;
        plant.stuck[1] = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        const LegCalibrationMoveStage target = stage == 0
            ? LegCalibrationMoveStage::NudgeOut
            : LegCalibrationMoveStage::Return;
        const LegCalibrationMovePhase phase = stage == 0
            ? LegCalibrationMovePhase::Nudge
            : LegCalibrationMovePhase::Detect;
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return calibration.moveStage() == target && calibration.movePhase() == phase;
        });
        TEST_ASSERT_TRUE(calibration.isRunning());
        // Nudge drives +PWM (counts should fall), return -PWM (should rise).
        plant.countBias[1] = stage == 0 ? 12 : -12;
        tick(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::WrongDirection);
        TEST_ASSERT_EQUAL_INT(0, plant.drivenCount());
    }
    // The settled count is evidence too: 12 edges against the drive by the end
    // of the nudge's settle fail it, with the joint still out.
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;
        plant.stuck[1] = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return calibration.moveStage() == LegCalibrationMoveStage::NudgeOut &&
                   calibration.movePhase() == LegCalibrationMovePhase::Settle;
        });
        plant.countBias[1] = 12;
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::WrongDirection);
        TEST_ASSERT_TRUE(plant.output.has(
            "CAL FRONT WARN FLHL nudged out, not driven back: jog it to its curl end"));
        TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT END FLHL nudge"));
    }
    // Evidence crossing 12 in the Nudge phase (slow hip: 12 edges after the ramp).
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.stuck[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return calibration.movePhase() == LegCalibrationMovePhase::Nudge;
    });
    plant.countBias[1] = -12;
    tick(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT SENSOR FLHL hall", plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Nudge),
        static_cast<int>(calibration.movePhase()));
}

// A pot hip at rest with no proven end: the nudge stops after 30 pot counts,
// and the return back to its start proves the end.
static void test_pot_hip_nudge_by_counts()
{
    FakePlant plant;
    plant.extension[1] = 100;
    plant.low[1] = 100;  // parked curled, no measured limits
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 2; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL nudge pot=130 travel=30"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=100 travel=-30"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));
    TEST_ASSERT_TRUE(plant.setHomeJoints.empty());  // a pot is never homed
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT HOME"));

    // A pot that does not come back far enough: short of its start.
    FakePlant shortPlant;
    shortPlant.extension[1] = 100;
    shortPlant.low[1] = 100;
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(shortPlant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(other, shortPlant, CAL_DEFAULT_PWM, [&]() {
        return other.moveStage() == LegCalibrationMoveStage::NudgeOut &&
               other.movePhase() == LegCalibrationMovePhase::Settle;
    });
    shortPlant.extension[1] = 200;
    shortPlant.low[1] = 131;
    runToEnd(other, shortPlant, CAL_DEFAULT_PWM);
    assertFailure(other, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL stopped 31 counts short of its curl end",
        shortPlant.output.last().c_str());
    // A pot nudge that moves the wrong way fails at once (no blind return).
    FakePlant wrong;
    wrong.extension[1] = 100;
    wrong.low[1] = 100;
    wrong.potScript = [&wrong](uint8_t joint, uint32_t) -> int {
        return joint == 1 ? 400 - wrong.extension[1] : wrong.extension[joint];
    };
    LegCalibration third;
    TEST_ASSERT_TRUE(third.start(wrong, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(third, wrong, CAL_DEFAULT_PWM);
    assertFailure(third, LegCalibrationFailure::PotWrongWay);
    TEST_ASSERT_TRUE(wrong.output.has("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_EQUAL_INT(0, wrong.output.count("CAL FRONT END FLHL nudge"));
}

// Knee resting curled: a strict run (options 0) fails with the jog hint; with the
// option (the shipped default, CAL_DEFAULT_OPTIONS) the knee nudges out at most
// 12 edges and 1 s, then straight back.
static void test_knee_nudge_only_with_the_option()
{
    FakePlant strict;
    strict.useHall();
    strict.extension[2] = 1000;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(strict, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, strict, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::NoMotion);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry",
        strict.output.last().c_str());
    TEST_ASSERT_FALSE(strict.output.has("CAL FRONT MOVE FLKL nudge"));
    for (const Drive &drive : strict.drives) TEST_ASSERT_TRUE(drive.joint == 2 && drive.pwm > 0);

    FakePlant plant;
    plant.useHall();
    plant.extension[2] = 1000;
    LegCalibration opted;
    TEST_ASSERT_TRUE(opted.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_OPT_KNEE_NUDGE));
    uint32_t nudgeFirst = 0;
    uint32_t nudgeLast = 0;
    int deepest = 1000;
    while (opted.isRunning() && opted.currentStep() < 1)
    {
        tick(opted, plant, CAL_DEFAULT_PWM);
        if (plant.pwm[2] < 0)
        {
            if (nudgeFirst == 0) nudgeFirst = plant.now;
            nudgeLast = plant.now;
        }
        if (plant.extension[2] < deepest) deepest = plant.extension[2];
    }
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLKL nudge"));
    TEST_ASSERT_TRUE(nudgeLast - nudgeFirst <= CAL_KNEE_NUDGE_MAX_MS);
    TEST_ASSERT_TRUE(1000 - deepest <= CAL_KNEE_NUDGE_EDGES);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL nudge hall=988 travel=-12"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-1000"));
    TEST_ASSERT_TRUE(opted.isJointCurled(2));

    // A knee nudge that counts no edge (here a dead motor) runs to its 1 s cap
    // (a stored knee needs up to 0.77 s to break free, so no earlier no-edge
    // stop), and a dead motor says so.
    FakePlant dead;
    dead.useHall();
    dead.extension[2] = 1000;
    dead.stuck[2] = true;
    dead.currentAbsent[2] = true;
    LegCalibration third;
    TEST_ASSERT_TRUE(third.start(dead, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_OPT_KNEE_NUDGE));
    uint32_t first = 0;
    uint32_t last = 0;
    while (third.isRunning())
    {
        tick(third, dead, CAL_DEFAULT_PWM);
        if (dead.pwm[2] < 0)
        {
            if (first == 0) first = dead.now;
            last = dead.now;
        }
    }
    TEST_ASSERT_TRUE(last - first < CAL_KNEE_NUDGE_MAX_MS);
    TEST_ASSERT_TRUE(last - first + 2 * CAL_PWM_RAMP_INTERVAL_MS >= CAL_KNEE_NUDGE_MAX_MS);
    assertFailure(third, LegCalibrationFailure::NoMotorCurrent);
}

// Hall A dead on a stored knee (the default path). As on the robot it needs
// 0.36-0.77 s of drive to break free of its curl end, then moves about 50
// counts/s, and springs out 9 more when the drive stops. No edge comes, so the
// nudge runs blind to its 1 s cap (the round-1 stop at 350 ms with no edge would
// also stop every healthy stored knee before it breaks free); the knee is driven
// straight back and the pass fails before its hip moves. The blind worst case is
// the fastest break-free: about 32 counts of drive plus the 9-count spring, about
// 41 counts out (calibration_constants.h), not the 18-21 of a healthy nudge.
static void test_blind_stored_knee_nudge_runs_to_its_cap_then_fails()
{
    for (uint32_t breakaway : {360U, 770U})
    {
        const std::string label = "break-free " + std::to_string(breakaway) + " ms";
        FakePlant plant;
        plant.useHall();
        plant.extension[2] = 1000;
        plant.hallDead[2] = true;
        plant.moveEvery[2] = 2;
        plant.breakawaySign[2] = -1;
        plant.breakawayMs[2] = breakaway;
        plant.springSign[2] = -1;
        plant.springOut[2] = -9;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
        uint32_t firstOut = 0, lastOut = 0, firstBack = 0, lastBack = 0;
        int deepest = 1000;
        while (calibration.isRunning())
        {
            tick(calibration, plant, CAL_DEFAULT_PWM);
            if (plant.pwm[2] < 0)
            {
                if (firstOut == 0) firstOut = plant.now;
                lastOut = plant.now;
            }
            if (plant.pwm[2] > 0 && lastOut != 0 && firstBack == 0) firstBack = plant.now;
            if (plant.pwm[2] > 0 && lastOut != 0) lastBack = plant.now;
            if (plant.extension[2] < deepest) deepest = plant.extension[2];
        }
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLKL nudge"));
        TEST_ASSERT_TRUE(lastOut - firstOut < CAL_KNEE_NUDGE_MAX_MS);
        TEST_ASSERT_TRUE(lastOut - firstOut + 2 * CAL_PWM_RAMP_INTERVAL_MS >= CAL_KNEE_NUDGE_MAX_MS);
        TEST_ASSERT_TRUE(firstBack > lastOut);
        // The blind return drives at least as long as the nudge did.
        TEST_ASSERT_TRUE(lastBack - firstBack >= lastOut - firstOut);
        TEST_ASSERT_EQUAL_INT(1000, plant.extension[2]);  // driven straight back
        // 50 counts/s from the break-free to the cap, then the 9-count spring.
        const int expectedOut = static_cast<int>((CAL_KNEE_NUDGE_MAX_MS - breakaway) / 20) + 9;
        TEST_ASSERT_INT_WITHIN_MESSAGE(3, expectedOut, 1000 - deepest,
            (label + ": " + std::to_string(1000 - deepest)).c_str());
        TEST_ASSERT_TRUE_MESSAGE(1000 - deepest <= 45, label.c_str());
        assertFailure(calibration, LegCalibrationFailure::NoFeedback);
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLKL no feedback: no hall edges, pot not moving (check sensor cable)",
            plant.output.last().c_str());
        for (const Drive &drive : plant.drives) TEST_ASSERT_EQUAL_UINT(2, drive.joint);
    }
}

// Below the default PWM 128 (FRKL could not break free at 80 or 100,
// 2026-11-24) a knee that shows no edge in its nudge fails naming the PWM and
// the PWM to use, rather than only the sensor cable.
static void test_knee_nudge_without_edges_below_the_default_pwm_names_the_speed()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[2] = 1000;
    plant.stuck[2] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, 50, CAL_DEFAULT_OPTIONS));
    runToEnd(calibration, plant, 50);
    assertFailure(calibration, LegCalibrationFailure::NoFeedback);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLKL no feedback at PWM 50: check sensor cable, or calibrate at PWM 128",
        plant.output.last().c_str());
    for (const Drive &drive : plant.drives) TEST_ASSERT_EQUAL_UINT(2, drive.joint);
}

// A slow working knee (10 edges/s) resting at its curl end, at C 150: the nudge
// ramps to no more than CAL_KNEE_NUDGE_MAX_PWM and runs to the 1 s cap, and with
// fewer than 12 edges the pass fails "barely moved" (a slow joint), not "no hall
// edges" (a dead cable).
static void test_slow_knee_nudge_is_capped_and_reported_as_barely_moved()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[2] = 1000;
    plant.moveEvery[2] = 10;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MAX, CAL_DEFAULT_OPTIONS));
    uint32_t firstOut = 0, lastOut = 0;
    int peakOut = 0;
    int peakCurl = 0;
    while (calibration.isRunning())
    {
        tick(calibration, plant, CAL_PWM_MAX);
        if (plant.pwm[2] < 0)
        {
            if (firstOut == 0) firstOut = plant.now;
            lastOut = plant.now;
            if (-plant.pwm[2] > peakOut) peakOut = -plant.pwm[2];
        }
        if (plant.pwm[2] > peakCurl) peakCurl = plant.pwm[2];
    }
    TEST_ASSERT_EQUAL_INT(CAL_KNEE_NUDGE_MAX_PWM, peakOut);
    TEST_ASSERT_EQUAL_INT(CAL_PWM_MAX, peakCurl);
    TEST_ASSERT_TRUE(lastOut - firstOut < CAL_KNEE_NUDGE_MAX_MS);
    TEST_ASSERT_TRUE(lastOut - firstOut + 2 * CAL_PWM_RAMP_INTERVAL_MS >= CAL_KNEE_NUDGE_MAX_MS);
    const long out = plant.output.valueAfter("CAL FRONT END FLKL nudge", "travel");
    TEST_ASSERT_TRUE(out < -CAL_POT_DETECT_MAX_EDGES && out > -CAL_HALL_DETECT_EDGES);
    assertFailure(calibration, LegCalibrationFailure::BarelyMoved);
    const std::string expected = "CAL FRONT FAIL FLKL barely moved (" + std::to_string(-out) +
        " edges) at PWM 150: check hall wiring or raise speed";
    TEST_ASSERT_EQUAL_STRING(expected.c_str(), plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(1000, plant.extension[2]);
}

// A healthy stored knee as measured on the robot (FLKL, 2026-11-23), on the
// default path. Parked by a PWM 200 curl, its PWM 80 curl shows no motion. It
// breaks free 0.36-0.77 s into its nudge and makes about 50 counts/s, so the
// drive stops at 12 edges, or 11 at the 1 s cap; it springs out 5-9 more (the
// measured range). The PWM 80 curl back stalls at the PWM 80 curl end, 17
// counts shallower than where it rested. With 2 or more counts of motion back
// it is accepted, homed there, and its hip moves next. With less (a spring at
// the low end) it fails "curled back only" before its hip moves, at its PWM 80
// curl end, so a second Curl only passes.
static void test_healthy_stored_knee_nudge_and_short_return_are_accepted()
{
    int accepted = 0;
    int retried = 0;
    for (uint32_t breakaway : {360U, 770U})
        for (int spring = 5; spring <= 9; ++spring)
        {
            const std::string label = "break-free " + std::to_string(breakaway) +
                " ms, spring " + std::to_string(spring);
            FakePlant plant;
            plant.useHall();
            plant.extension[2] = 1000;
            plant.moveEvery[2] = 2;
            plant.breakawaySign[2] = -1;
            plant.breakawayMs[2] = breakaway;
            plant.springSign[2] = -1;
            plant.springOut[2] = -spring;
            LegCalibration calibration;
            TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
            uint32_t firstOut = 0;
            uint32_t lastOut = 0;
            uint32_t firstEdge = 0;
            while (calibration.isRunning() && calibration.moveStage() != LegCalibrationMoveStage::Return)
            {
                tick(calibration, plant, CAL_DEFAULT_PWM);
                if (plant.pwm[2] < 0)
                {
                    if (firstOut == 0) firstOut = plant.now;
                    lastOut = plant.now;
                }
                if (firstEdge == 0 && plant.extension[2] < 1000) firstEdge = plant.now;
            }
            TEST_ASSERT_TRUE_MESSAGE(calibration.isRunning(), label.c_str());
            TEST_ASSERT_TRUE_MESSAGE(firstEdge - firstOut + 20 >= breakaway &&
                                         firstEdge - firstOut <= breakaway + 20,
                label.c_str());
            TEST_ASSERT_TRUE_MESSAGE(lastOut - firstOut < CAL_KNEE_NUDGE_MAX_MS, label.c_str());
            const long out = plant.output.valueAfter("CAL FRONT END FLKL nudge", "travel");
            // 12 drive edges (11 at the cap after a late break-free), then the spring.
            TEST_ASSERT_TRUE_MESSAGE(out <= -(CAL_KNEE_NUDGE_EDGES - 1 + spring) &&
                                         out >= -(CAL_KNEE_NUDGE_EDGES + spring + 1),
                (label + ": " + std::to_string(out)).c_str());
            plant.high[2] = 983;  // the PWM 80 curl end: 17 counts shallower than the PWM 200 park
            const size_t drivesBefore = plant.drives.size();
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() != 2; });
            const long back = -out - 17;  // where it stalls: the PWM 80 curl end
            if (back >= CAL_KNEE_RETURN_MIN_EDGES)
            {
                ++accepted;
                TEST_ASSERT_TRUE_MESSAGE(calibration.isRunning(), (label + ": " + plant.output.last()).c_str());
                TEST_ASSERT_EQUAL_INT_MESSAGE(back,
                    plant.output.valueAfter("CAL FRONT END FLKL curl hall=983", "travel"), label.c_str());
                TEST_ASSERT_TRUE_MESSAGE(plant.output.has("CAL FRONT HOME FLKL count=-983"), label.c_str());
                TEST_ASSERT_TRUE(calibration.isJointCurled(2));
                TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL curl"));
                TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL"));
                continue;
            }
            // The low-end spring: it fails, names the lack of motion (not "short":
            // it is well inside the 40-edge tolerance), and the hip never moved.
            ++retried;
            assertFailure(calibration, LegCalibrationFailure::NoCurlBack);
            const std::string expected = "CAL FRONT FAIL FLKL curled back only " + std::to_string(back) +
                " edges after its nudge";
            TEST_ASSERT_EQUAL_STRING_MESSAGE(expected.c_str(), plant.output.last().c_str(), label.c_str());
            TEST_ASSERT_FALSE(drovePartnerAfter(plant, drivesBefore, 1));
            TEST_ASSERT_EQUAL_INT(983, plant.extension[2]);
            // The knee now rests at its PWM 80 curl end: a second Curl only passes.
            LegCalibration again;
            TEST_ASSERT_TRUE(again.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
            runUntil(again, plant, CAL_DEFAULT_PWM, [&]() { return again.currentJoint() != 2; });
            TEST_ASSERT_TRUE_MESSAGE(again.isRunning(), (label + " retry: " + plant.output.last()).c_str());
            TEST_ASSERT_TRUE(again.isJointCurled(2));
        }
    // A 12-edge stop needs a 7-edge spring to come back 2; an 11-edge stop at
    // the 1 s cap needs 8: the low end of the measured 5-9 needs the retry.
    TEST_ASSERT_EQUAL_INT(5, accepted);
    TEST_ASSERT_EQUAL_INT(5, retried);
}

// A hall knee's curl back after its nudge is accepted when it stalls no more
// than 40 edges short of where the knee rested and moved at least
// CAL_KNEE_RETURN_MIN_EDGES, far fewer than the 12 of the nudge evidence. With
// less motion (a curl drive that cannot move it: it may be stuck mid-stroke) it
// fails "curled back only", not "short" (it is inside the tolerance); more than
// 40 short fails "short". Either way before its hip moves. A hip's return still
// needs CAL_HALL_DETECT_EDGES of motion.
static void test_nudge_return_bounds_for_knee_and_hip()
{
    struct Case
    {
        uint8_t joint;
        int springOut;   // after the nudge stops (knee nudge 12, hip nudge 24 out)
        int returnEnd;   // where the curl back stalls
        const char *failure;
        LegCalibrationFailure kind;
    };
    const LegCalibrationFailure none = LegCalibrationFailure::None;
    const Case cases[] = {
        {2, -9, 981, nullptr, none},  // 2 back, 19 short
        {2, -9, 980, "CAL FRONT FAIL FLKL curled back only 1 edges after its nudge",
         LegCalibrationFailure::NoCurlBack},  // 20 short
        {2, -9, 979, "CAL FRONT FAIL FLKL curled back only 0 edges after its nudge",
         LegCalibrationFailure::NoCurlBack},  // 21 short
        {2, -5, 983, "CAL FRONT FAIL FLKL curled back only 0 edges after its nudge",
         LegCalibrationFailure::NoCurlBack},  // a 5-edge spring after a PWM 200 park
        {2, -50, 960, nullptr, none},  // 22 back, 40 short
        {2, -50, 959, "CAL FRONT FAIL FLKL stopped 41 edges short of its curl end",
         LegCalibrationFailure::ShortOfCurlEnd},
        {2, -29, 959, "CAL FRONT FAIL FLKL stopped 41 edges short of its curl end",
         LegCalibrationFailure::ShortOfCurlEnd},  // no motion, 41 short
        {1, 0, 13, "CAL FRONT FAIL FLHL curled back only 11 edges after its nudge",
         LegCalibrationFailure::NoCurlBack},  // 13 short: within the tolerance
        {1, 0, 12, nullptr, none},  // 12 back
    };
    for (const Case &entry : cases)
    {
        FakePlant plant;
        plant.useHall();
        const bool hip = entry.joint == 1;
        plant.extension[2] = 1000;
        plant.extension[1] = 0;
        plant.springSign[entry.joint] = hip ? 1 : -1;
        plant.springOut[entry.joint] = entry.springOut;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
        if (hip)  // the knee first: its own nudge and return, accepted
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() == 1; });
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return calibration.moveStage() == LegCalibrationMoveStage::Return;
        });
        TEST_ASSERT_EQUAL_UINT(entry.joint, calibration.currentJoint());
        if (hip)
            plant.low[1] = entry.returnEnd;
        else
            plant.high[2] = entry.returnEnd;
        const size_t drivesBefore = plant.drives.size();
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() != entry.joint; });
        if (entry.failure != nullptr)
        {
            assertFailure(calibration, entry.kind);
            TEST_ASSERT_EQUAL_STRING(entry.failure, plant.output.last().c_str());
            TEST_ASSERT_FALSE(drovePartnerAfter(plant, drivesBefore, hip ? 5 : 1));
            continue;
        }
        TEST_ASSERT_TRUE(calibration.isRunning());
        TEST_ASSERT_TRUE(calibration.isJointCurled(entry.joint));
    }
}

// A knee nudge that fails during the nudge (hall A/B swapped: the count runs
// against the drive) is not driven back: a WARN says so before the FAIL. An
// abort during a nudge says the same; an abort elsewhere does not.
static void test_failure_or_abort_during_a_nudge_warns_the_joint_is_out()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[2] = 1000;
    plant.hallReversed[2] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::WrongDirection);
    const std::vector<std::string> lines = plant.output.lines();
    TEST_ASSERT_TRUE(lines.size() >= 2);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT WARN FLKL nudged out, not driven back: jog it to its curl end",
        lines[lines.size() - 2].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL hall counts the wrong way: check motor or hall A/B wiring",
        lines.back().c_str());
    TEST_ASSERT_TRUE(plant.extension[2] < 1000);

    FakePlant hip;
    hip.useHall();
    hip.extension[1] = 0;
    hip.moveEvery[1] = 3;
    LegCalibration aborted;
    TEST_ASSERT_TRUE(aborted.start(hip, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(aborted, hip, CAL_DEFAULT_PWM, [&]() {
        return aborted.moveStage() == LegCalibrationMoveStage::NudgeOut &&
               aborted.movePhase() == LegCalibrationMovePhase::Nudge;
    });
    TEST_ASSERT_TRUE(aborted.abort(hip, 'H'));
    const std::vector<std::string> abortLines = hip.output.lines();
    TEST_ASSERT_EQUAL_STRING("CAL FRONT WARN FLHL nudged out, not driven back: jog it to its curl end",
        abortLines[abortLines.size() - 2].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT ABORTED H", abortLines.back().c_str());

    FakePlant normal;
    normal.useHall();
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(normal, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    tick(other, normal, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(other.abort(normal, 'H'));
    TEST_ASSERT_EQUAL_INT(0, normal.output.count("CAL FRONT WARN"));
}

// ------------------------------------------------- pot on a quadrature build

// A pot on a Rev 3 slot: the first attempt is windowed on edges and stops after
// the first window; at rest both snapshots are in band with no edges, so it is
// a pot; the step repeats once with pot windows.
static void test_late_pot_classification_repeats_the_step()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[2] = false;  // the knee is a pot rod: no edges, pot in band
    plant.low[2] = 100;
    plant.high[2] = 900;
    plant.extension[2] = 300;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 1; });
    TEST_ASSERT_TRUE(calibration.isRunning());
    // The edge-windowed attempt ends after one window (2.82 s): no END line, a
    // SENSOR line, then the repeat runs to the pot's end.
    TEST_ASSERT_EQUAL_INT(900, plant.output.valueAfter("CAL FRONT END FLKL curl", "pot"));
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT SENSOR FLKL pot"));
    TEST_ASSERT_EQUAL_INT(2, plant.output.count("CAL FRONT MOVE FLKL curl"));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Pot), static_cast<int>(plant.kind[2]));
    TEST_ASSERT_TRUE(calibration.isJointCurled(2));
    TEST_ASSERT_TRUE(plant.setHomeJoints.empty());
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT END FLKL"));
}

// Stored Hall but now a pot (swapped): classified at rest "(was hall)".
static void test_stored_hall_that_is_now_a_pot()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[2] = false;
    plant.kind[2] = JointSensor::Hall;
    plant.extension[2] = 300;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 1; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLKL pot (was hall)"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(2));
}

// The edge pattern that reaches the repeat check a second time: the repeated
// move counts 12 edges (Hall), then they unwind, and it reads as a pot at rest
// again. No second repeat: it is judged as it stands.
static void test_second_late_classification_does_not_repeat()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[2] = false;
    plant.extension[2] = 300;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
        return plant.output.count("CAL FRONT MOVE FLKL curl") == 2;
    });
    plant.countBias[2] = -12;  // 12 edges in the drive direction: hall
    tick(calibration, plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLKL hall (was pot)"));
    plant.countBias[2] = -1;   // back within the pot edge tolerance
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 1; });
    TEST_ASSERT_EQUAL_INT(2, plant.output.count("CAL FRONT MOVE FLKL curl"));
    TEST_ASSERT_EQUAL_INT(2, plant.output.count("CAL FRONT SENSOR FLKL pot"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(2));
}

// A pot on a Rev 3 slot wired backwards: PotWrongWay at rest.
static void test_late_pot_classification_wrong_way_fails()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[2] = false;
    plant.reversed[2] = true;
    plant.extension[2] = 300;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::PotWrongWay);
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT MOVE FLKL curl"));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(JointSensor::Unknown), static_cast<int>(plant.kind[2]));
}

// A hall-B rail flip between the snapshots (1015 -> 6) with no A edges is never
// pot evidence: the knee has no evidence and fails.
static void test_rail_flip_is_not_pot_evidence()
{
    FakePlant plant;
    plant.useHall();
    plant.hallDead[2] = true;
    plant.snapScript = [](uint8_t joint, uint32_t now) -> int {
        return joint == 2 && now > 0 ? 6 : 1015;
    };
    plant.potScript = [](uint8_t, uint32_t now) -> int { return static_cast<int>(now % 1000); };
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::NoMotion);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=0 travel=0 pot=6"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT SENSOR"));
    TEST_ASSERT_TRUE(plant.setSensorCalls.empty());
}

// Pot evidence comes from the at-rest snapshots only, never from the smoothed
// reading (EMA, which lags and averages hall-B motion noise into the band). A
// no-edge move whose EMA is in the pot band at both ends while a snapshot sits
// on a hall-B rail is not a pot; with the snapshots in band and the EMA on the
// rails it is. Run for a dead-A knee and a dead-A hip.
static void test_pot_evidence_is_the_snapshots_not_the_smoothed_reading()
{
    for (uint8_t joint : {static_cast<uint8_t>(2), static_cast<uint8_t>(1)})
    {
        for (bool snapshotsInBand : {false, true})
        {
            FakePlant plant;
            plant.useHall();
            plant.hallDead[joint] = true;
            const bool rising = legCalibrationPwmSign(joint, true) > 0;  // curl raises a pot
            const int bandStart = rising ? 400 : 700;
            const int bandEnd = rising ? 700 : 400;
            auto driven = [&plant, joint]() {
                for (const Drive &drive : plant.drives)
                    if (drive.joint == joint) return true;
                return false;
            };
            auto rest = [&plant](uint8_t other) { return (plant.hallCount(other) % 2) != 0 ? 1016 : 12; };
            // In band: start -> end; rails: the snapshot starts on a rail, the EMA ends on one.
            plant.snapScript = [=](uint8_t j, uint32_t) -> int {
                if (j != joint) return rest(j);
                if (snapshotsInBand) return driven() ? bandEnd : bandStart;
                return driven() ? bandEnd : 1016;
            };
            plant.potScript = [=](uint8_t j, uint32_t) -> int {
                if (j != joint) return rest(j);
                if (snapshotsInBand) return driven() ? 19 : 1016;
                return driven() ? bandEnd : bandStart;
            };
            LegCalibration calibration;
            TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
            runToEnd(calibration, plant, CAL_DEFAULT_PWM);
            const std::string sensorLine = std::string("CAL FRONT SENSOR ") + plant.names[joint] + " pot";
            bool setPot = false;
            for (const auto &call : plant.setSensorCalls)
                setPot = setPot || (call.first == joint && call.second == JointSensor::Pot);
            TEST_ASSERT_EQUAL_MESSAGE(snapshotsInBand, plant.output.has(sensorLine), plant.output.text.c_str());
            TEST_ASSERT_EQUAL(snapshotsInBand, setPot);
        }
    }
}

// The nudge itself can classify a pot; the return then runs in pot units with
// no repeat. If the return classifies differently from the nudge, it fails.
static void test_classification_at_the_nudge_and_the_return()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[1] = false;  // the hip is a pot rod parked curled
    plant.low[1] = 100;
    plant.extension[1] = 100;
    plant.moveEvery[1] = 2;    // 50 counts/s, no edges: the nudge ends on its cap
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentStep() >= 2; });
    TEST_ASSERT_TRUE(calibration.isRunning());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLHL pot"));
    const long nudged = plant.output.valueAfter("CAL FRONT END FLHL nudge", "travel");
    TEST_ASSERT_TRUE(nudged >= CAL_MIN_TRAVEL_COUNTS);
    TEST_ASSERT_FALSE(plant.output.find("CAL FRONT END FLHL nudge pot=").empty());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=100 travel=" + std::to_string(-nudged)));
    TEST_ASSERT_EQUAL_INT(2, plant.output.count("CAL FRONT MOVE FLHL curl"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));

    // Hall during the nudge, pot at the end of the return: the start reading is
    // in other units, so the end is not proven.
    FakePlant mixed;
    mixed.useHall();
    mixed.hallRod[1] = false;
    mixed.low[1] = 100;
    mixed.extension[1] = 100;
    mixed.moveEvery[1] = 2;
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(mixed, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(other, mixed, CAL_DEFAULT_PWM, [&]() {
        return other.moveStage() == LegCalibrationMoveStage::NudgeOut &&
               other.movePhase() == LegCalibrationMovePhase::Nudge;
    });
    mixed.countBias[1] = -12;  // 12 edges out: hall
    tick(other, mixed, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(mixed.output.has("CAL FRONT SENSOR FLHL hall"));
    runUntil(other, mixed, CAL_DEFAULT_PWM, [&]() {
        return other.moveStage() == LegCalibrationMoveStage::Return;
    });
    mixed.countBias[1] = -11;  // the return counts back to within 1 edge of its start
    runToEnd(other, mixed, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE(mixed.output.has("CAL FRONT SENSOR FLHL pot (was hall)"));
    assertFailure(other, LegCalibrationFailure::NoFeedback);
}

// -------------------------------------------------------------- hall SWEEP

// A SWEEP after a SEEK with configurable hip travel.
struct HallSweep
{
    FakePlant plant;
    LegCalibration calibration;

    explicit HallSweep(uint8_t options = 0)
    {
        seekHall(calibration, plant, options);
        plant.speed = 4;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_DEFAULT_PWM, options));
    }

    void runToStep(uint8_t step) { runUntilStep(calibration, plant, step, CAL_DEFAULT_PWM); }
    void finish() { runToEnd(calibration, plant, CAL_DEFAULT_PWM); }
};

// A hall hip's SWEEP uncurl is a fixed sweep: it stops once it has made
// CAL_HIP_SWEEP_EDGES drive edges from its curl end, long before its rod's end
// (1000 here: with its knee curled a real hip would press the tibia into the
// body first) and without a stall window. Its saved range is [0, 700] whatever
// an earlier SWEEP saved: no stroke check or stroke WARN applies to a hall hip.
// The knee still sweeps its full stroke.
static void test_hall_hip_sweep_stops_at_its_fixed_target()
{
    HallSweep sweep;
    sweep.plant.speed = 1;                // 100 edges/s
    sweep.plant.limitsValid[1] = true;    // an older record's full stroke
    sweep.plant.maxStop[1] = 1000;
    int furthest = 0;
    uint32_t firstOut = 0;
    uint32_t lastOut = 0;
    while (sweep.calibration.isRunning() && sweep.calibration.currentStep() < 4)
    {
        tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
        if (sweep.plant.pwm[1] > 0)
        {
            if (firstOut == 0) firstOut = sweep.plant.now;
            lastOut = sweep.plant.now;
        }
        if (sweep.plant.extension[1] > furthest) furthest = sweep.plant.extension[1];
    }
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, furthest);
    // Driven for the 700 edges only (7 s at 100 edges/s), not into a stall.
    TEST_ASSERT_TRUE_MESSAGE(lastOut - firstOut <= 7000, std::to_string(lastOut - firstOut).c_str());
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLHL uncurl hall=700 travel=700"));
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL curl hall=0 travel=-700", "switch"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLHL min=0 max=700 edges (fixed sweep)"));
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT WARN"));
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.minStop[1]);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, sweep.plant.maxStop[1]);
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1000 max=0 edges"));
    TEST_ASSERT_EQUAL_INT(-1000, sweep.plant.minStop[2]);
}

// The settled reading of a hip stopped at its sweep target may relax an edge
// or two back toward its curl end: the sweep still counts as reached.
static void test_hall_hip_sweep_target_survives_relaxation()
{
    HallSweep sweep;
    runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
        return sweep.calibration.currentStep() == 2 &&
               sweep.calibration.movePhase() == LegCalibrationMovePhase::Settle;
    });
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, sweep.plant.extension[1]);
    sweep.plant.countBias[1] += 2;  // the count rises: 2 edges back
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLHL uncurl hall=698 travel=698"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLHL min=0 max=700 edges (fixed sweep)"));
}

// An obstacle stops the hip before its sweep target: the pass fails naming the
// shortfall, the hip's home is no longer trusted, its knee never uncurls and
// nothing is saved. A hip that does not move at all is the whole sweep short.
static void test_hall_hip_sweep_short_of_its_target_fails()
{
    for (int reach : {420, 0})
    {
        HallSweep sweep;
        sweep.plant.high[1] = reach;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::SweepShort);
        const std::string expected = "CAL FRONT FAIL FLHL stopped " + std::to_string(CAL_HIP_SWEEP_EDGES - reach) +
            " edges short of its 700-edge sweep: obstacle?";
        TEST_ASSERT_EQUAL_STRING(expected.c_str(), sweep.plant.output.last().c_str());
        const std::string end = "CAL FRONT END FLHL uncurl hall=" + std::to_string(reach) +
            " travel=" + std::to_string(reach);
        // An obstacle after a running phase is a stop signature; a hip that
        // never ran ends as a plain stall.
        TEST_ASSERT_TRUE_MESSAGE(reach > 0 ? sweep.plant.output.hasEnd(end, "stop") : sweep.plant.output.has(end),
                                 sweep.plant.output.text.c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_FALSE(sweep.plant.output.has("CAL FRONT MOVE FLKL uncurl"));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
}

// A pot hip (here a proven pot on a quadrature board) keeps the full sweep to
// its uncurl end and saves its pot range.
static void test_pot_hip_keeps_the_full_sweep()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[1] = false;
    plant.kind[1] = JointSensor::Pot;
    plant.low[1] = 100;
    plant.high[1] = 900;
    plant.speed = 4;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=900 travel=800"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FLHL min=100 max=900"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FRHL min=0 max=700 edges (fixed sweep)"));
    TEST_ASSERT_EQUAL_INT(900, plant.maxStop[1]);
}

// A knee uncurl must travel the minimum stroke.
static void test_sweep_uncurl_must_travel_the_minimum_stroke()
{
    for (int travel : {CAL_HALL_MIN_STROKE_EDGES - 1, CAL_HALL_MIN_STROKE_EDGES})
    {
        HallSweep sweep;
        sweep.plant.speed = 1;
        sweep.plant.low[2] = 1000 - travel;
        sweep.runToStep(5);
        if (travel < CAL_HALL_MIN_STROKE_EDGES)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::BarelyMoved);
            TEST_ASSERT_EQUAL_STRING(
                "CAL FRONT FAIL FLKL barely moved (449 edges) at PWM 128: check hall wiring or raise speed",
                sweep.plant.output.last().c_str());
            TEST_ASSERT_FALSE(sweep.plant.homed[2]);
            continue;
        }
        TEST_ASSERT_TRUE(sweep.calibration.isRunning());
        TEST_ASSERT_EQUAL_INT(-450, sweep.calibration.stagedMin(2));
        sweep.finish();
        assertCompleted(sweep.calibration);
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-450 max=0 edges"));
    }
}

static void test_sweep_curl_back_must_leave_the_uncurl_end_with_its_home()
{
    {
        HallSweep sweep;
        sweep.runToStep(3);
        sweep.plant.stuck[1] = true;  // cannot come back
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL did not reach its curl end (hall=700)",
            sweep.plant.output.last().c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_FALSE(sweep.plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    }
    {
        HallSweep sweep;
        sweep.runToStep(3);
        sweep.plant.clearHome(1);  // a guard cleared the home meanwhile
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL did not reach its curl end (hall=0)",
            sweep.plant.output.last().c_str());
    }
    {
        // The hip's curl back must leave its sweep end by the minimum stroke.
        HallSweep sweep;
        sweep.plant.speed = 1;
        sweep.runToStep(3);
        sweep.plant.low[1] = CAL_HIP_SWEEP_EDGES - CAL_HALL_MIN_STROKE_EDGES + 1;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL did not reach its curl end (hall=251)",
            sweep.plant.output.last().c_str());
    }
}

static void test_sweep_curl_back_short_of_its_home_fails()
{
    for (int end : {40, 41})
    {
        HallSweep sweep;
        sweep.runToStep(3);
        sweep.plant.low[1] = end;
        sweep.finish();
        if (end == 41)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::ShortOfCurlEnd);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL stopped 41 edges short of its curl end",
                sweep.plant.output.last().c_str());
            TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        }
        else
        {
            assertCompleted(sweep.calibration);
            // Short within tolerance: the home stays; a hip's range is its fixed sweep.
            TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLHL min=0 max=700 edges (fixed sweep)"));
        }
    }
}

// The knee's stroke must fall within the stroke bounds.
static void test_sweep_stroke_out_of_range()
{
    {
        // Low: the stroke is never less than the distance the curl back left the
        // uncurl end, so a stroke under the minimum fails that check first. Here
        // the knee rested 20 edges deeper than its home (accepted without motion
        // and re-homed there), went 460 edges out and came back 440 of them.
        HallSweep sweep;
        sweep.plant.speed = 1;
        sweep.plant.home[2] = -980;
        sweep.plant.low[2] = 1000 - 460;
        sweep.runToStep(5);
        TEST_ASSERT_EQUAL_INT(-1000, sweep.plant.home[2]);
        TEST_ASSERT_EQUAL_INT(-460, sweep.calibration.stagedMin(2));
        sweep.plant.high[2] = 980;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL did not reach its curl end (hall=-20)",
            sweep.plant.output.last().c_str());
    }
    for (int stroke : {CAL_HALL_MAX_STROKE_EDGES, CAL_HALL_MAX_STROKE_EDGES + 1})
    {
        HallSweep sweep;
        sweep.plant.low[2] = 1000 - stroke;
        sweep.runToStep(6);
        if (stroke > CAL_HALL_MAX_STROKE_EDGES)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::StrokeOutOfRange);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stroke 1501 edges, outside 450-1500",
                sweep.plant.output.last().c_str());
        }
        else
        {
            TEST_ASSERT_TRUE(sweep.calibration.isRunning());
            TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1500 max=0 edges"));
        }
    }
}

// Against a knee's saved stroke S0: under 85% fails, a change over
// max(30, S0/10) warns.
static void test_sweep_stroke_against_the_saved_stroke()
{
    const struct
    {
        int saved;
        int stroke;
        bool fails;
        bool warns;
    } cases[] = {
        {1000, 849, true, false},
        {1000, 850, false, true},
        {1000, 900, false, false},
        {1000, 1100, false, false},
        {1000, 1101, false, true},
        {460, 506, false, false},
        {460, 507, false, true},
    };
    for (const auto &entry : cases)
    {
        HallSweep sweep;
        sweep.plant.speed = 1;
        sweep.plant.limitsValid[2] = true;  // a stroke saved by an earlier SWEEP
        sweep.plant.minStop[2] = -entry.saved;
        sweep.plant.maxStop[2] = 0;
        sweep.plant.low[2] = 1000 - entry.stroke;
        sweep.runToStep(6);
        if (entry.fails)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::StrokeShort);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stroke 849 edges, saved 1000: obstacle?",
                sweep.plant.output.last().c_str());
            TEST_ASSERT_EQUAL_INT(1, sweep.plant.output.count("CAL FRONT MOVE FRKL curl"));  // SEEK's only
            continue;
        }
        TEST_ASSERT_TRUE(sweep.calibration.isRunning());
        const std::string warn = "CAL FRONT WARN FLKL stroke " + std::to_string(entry.saved) +
                                 " -> " + std::to_string(entry.stroke) + " edges";
        TEST_ASSERT_EQUAL(entry.warns, sweep.plant.output.has(warn));
        TEST_ASSERT_EQUAL_INT(entry.warns ? 1 : 0, sweep.plant.output.count("CAL FRONT WARN"));
    }
}

// A curl back that ends deeper than the home re-homes there. A hip keeps its
// fixed range; a knee's stroke is measured to its new home.
static void test_sweep_deeper_curl_back_rehomes_and_recomputes_the_stroke()
{
    HallSweep sweep;
    sweep.runToStep(3);
    sweep.plant.low[1] = -50;
    sweep.runToStep(4);
    TEST_ASSERT_TRUE(sweep.calibration.isRunning());
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL curl hall=-50 travel=-750", "switch"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLHL min=0 max=700 edges (fixed sweep)"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT HOME FLHL count=50"));
    TEST_ASSERT_EQUAL_INT(50, sweep.plant.home[1]);
    sweep.runToStep(5);
    sweep.plant.high[2] = 1050;  // the knee curls back 50 deeper than its home
    sweep.runToStep(6);
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLKL curl hall=50 travel=1050", "stop"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1050 max=0 edges"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT HOME FLKL count=-1050"));
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_EQUAL_INT(-1050, sweep.calibration.stagedMin(2));
    TEST_ASSERT_EQUAL_INT(0, sweep.calibration.stagedMax(2));
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, sweep.plant.maxStop[1]);
    TEST_ASSERT_EQUAL_INT(-1050, sweep.plant.minStop[2]);
}

// A failed hall joint loses its home; a joint that did not fail keeps it.
static void test_fail_clears_only_the_failed_hall_home()
{
    HallSweep sweep;
    sweep.runToStep(5);  // knee curling back
    sweep.plant.stallSign[2] = 1;
    sweep.finish();
    assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
    TEST_ASSERT_FALSE(sweep.plant.homed[2]);
    TEST_ASSERT_TRUE(sweep.plant.homeLost[2]);
    TEST_ASSERT_TRUE(sweep.plant.homed[1]);
    TEST_ASSERT_EQUAL_UINT(1, sweep.plant.clearHomeJoints.size());
    TEST_ASSERT_EQUAL_UINT(2, sweep.plant.clearHomeJoints[0]);
}

// --------------------------------------------------------- robot-log replays

// Open-loop replay of recorded FRONT frames: each joint with a recording reads
// it from the time of its move's first drive. The probe reads every pin as open,
// as in run 2.
struct ReplayPlant final : public LegCalibrationPlant
{
    uint32_t now = 0;
    const RobotLogFrame *frames[6] = {};
    size_t frameCount[6] = {};
    uint32_t originMs[6] = {};
    bool started[6] = {};
    JointSensor kind[6] = {JointSensor::Unknown, JointSensor::Unknown, JointSensor::Unknown,
                           JointSensor::Unknown, JointSensor::Unknown, JointSensor::Unknown};
    int32_t home[6] = {};
    bool homed[6] = {};
    int pwm[6] = {};
    const char *names[6] = {"FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
    FakeOutput output;

    void play(uint8_t joint, const RobotLogFrame *data, size_t count)
    {
        frames[joint] = data;
        frameCount[joint] = count;
    }
    const RobotLogFrame *frame(uint8_t joint) const
    {
        if (frames[joint] == nullptr) return nullptr;
        size_t index = started[joint] ? (now - originMs[joint]) / ROBOT_LOG_FRAME_MS : 0;
        if (index >= frameCount[joint]) index = frameCount[joint] - 1;
        return &frames[joint][index];
    }
    // Recorded time of joint's move (ms from its first drive).
    uint32_t moveMs(uint8_t joint) const { return now - originMs[joint]; }

    void drive(uint8_t joint, int value) override
    {
        pwm[joint] = value;
        if (value != 0 && !started[joint])
        {
            started[joint] = true;
            originMs[joint] = now;
        }
    }
    void stopAll() override { for (int &value : pwm) value = 0; }
    int rawPot(uint8_t joint) override
    {
        const RobotLogFrame *recorded = frame(joint);
        return recorded != nullptr ? recorded->pot : 1016;
    }
    int potSnapshot(uint8_t joint) override { return rawPot(joint); }
    bool isPotOpen(uint8_t) override { return true; }
    bool hasQuadratureHall() override { return true; }
    int32_t hallCount(uint8_t joint) override
    {
        const RobotLogFrame *recorded = frame(joint);
        return recorded != nullptr ? recorded->count : 0;
    }
    int position(uint8_t joint) override
    {
        if (kind[joint] == JointSensor::Pot) return rawPot(joint);
        return static_cast<int>(ACTUATOR_HALL_EXTEND_COUNT_SIGN * (hallCount(joint) - home[joint]));
    }
    JointSensor sensor(uint8_t joint) override { return kind[joint]; }
    void setSensor(uint8_t joint, JointSensor value) override { kind[joint] = value; }
    bool isHomed(uint8_t joint) override { return homed[joint]; }
    void setHome(uint8_t joint) override
    {
        home[joint] = hallCount(joint);
        homed[joint] = true;
    }
    void clearHome(uint8_t joint) override { homed[joint] = false; }
    bool isCurrentAbsent(uint8_t) override { return false; }
    // These 2026-11-23 logs carry no current: a model of what 2026-11-24
    // measured. Running 110; extending, it rises as the recorded count slows
    // (to 170 with no edge in 200 ms: the knee's compliant stop); retracting,
    // it drops to 75 once the count has not moved for 150 ms (the hip's switch).
    uint16_t motorCurrentTenths(uint8_t joint) override
    {
        if (pwm[joint] == 0 || frames[joint] == nullptr) return 0;
        const size_t index = (now - originMs[joint]) / ROBOT_LOG_FRAME_MS;
        const RobotLogFrame *at = frame(joint);
        const size_t clamped = static_cast<size_t>(at - frames[joint]);
        const int back3 = frames[joint][clamped >= 3 ? clamped - 3 : 0].count;
        const int back4 = frames[joint][clamped >= 4 ? clamped - 4 : 0].count;
        if (index < 4) return 110;
        if (pwm[joint] < 0) return at->count == back3 ? 75 : 110;
        int speed = std::abs(at->count - back4);
        if (speed > 8) speed = 8;
        return static_cast<uint16_t>(110 + 60 * (8 - speed) / 8);
    }
    bool measuredLimit(uint8_t, bool, int &) override { return false; }
    int savedStroke(uint8_t) override { return 0; }
    uint32_t nowMs() override { return now; }
    const char *jointName(uint8_t joint) override { return names[joint]; }
    const char *roleLabel() override { return "FRONT"; }
    CalibrationTextSink &out() override { return output; }
    void storeLimits(uint8_t, int, int) override {}
    void saveLimits() override {}
};

// Time (ms from the first drive) of the recorded EN drop.
uint32_t recordedEnDropMs(const RobotLogFrame *frames, size_t count)
{
    for (size_t i = 1; i < count; ++i)
        if (frames[i].en == 0 && frames[i - 1].en != 0) return static_cast<uint32_t>(i) * ROBOT_LOG_FRAME_MS;
    return static_cast<uint32_t>(count) * ROBOT_LOG_FRAME_MS;
}

// Recorded count at ms from the first drive.
int recordedCount(const RobotLogFrame *frames, size_t count, uint32_t ms)
{
    size_t index = ms / ROBOT_LOG_FRAME_MS;
    if (index >= count) index = count - 1;
    return frames[index].count;
}

struct ReplayResult
{
    uint32_t hallMs = 0;           // SENSOR hall line, from the joint's first drive
    std::vector<uint32_t> stalls;  // window ends (Windows -> Settle), same clock
    bool failedWhileRecording = false;
};

// Replays one joint's move until the sequencer leaves it.
ReplayResult replayMove(LegCalibration &calibration, ReplayPlant &plant, uint8_t joint)
{
    ReplayResult result;
    const std::string sensorLine = std::string("CAL FRONT SENSOR ") + plant.names[joint] + " hall";
    const uint32_t recordedMs = static_cast<uint32_t>(plant.frameCount[joint]) * ROBOT_LOG_FRAME_MS;
    while (calibration.isRunning() && calibration.currentJoint() != joint)
    {
        plant.now += 10;
        calibration.update(plant);
    }
    while (calibration.isRunning() && calibration.currentJoint() == joint)
    {
        const LegCalibrationMovePhase before = calibration.movePhase();
        plant.now += 10;
        calibration.update(plant);
        if (result.hallMs == 0 && plant.output.has(sensorLine)) result.hallMs = plant.moveMs(joint);
        if ((before == LegCalibrationMovePhase::Windows || before == LegCalibrationMovePhase::Detect) &&
            calibration.movePhase() == LegCalibrationMovePhase::Settle)
            result.stalls.push_back(plant.moveMs(joint));
        if (!calibration.isRunning() && plant.moveMs(joint) <= recordedMs)
            result.failedWhileRecording = true;
    }
    return result;
}

// Run 1, FLKL curl at PWM 80 (log 14513-14865): count 160 -> 1 with the taper
// 8, 6, 5, 4, 3, 2, 2, 1. Hall within 0.5 s; the end detector calls the
// compliant stop once, during the final creep (this PWM-80 knee crept its last
// 15 edges over about 1.5 s: that creep is the compliant stop itself, and the
// 25-edge T inset at the stop side absorbs it), never while the count still
// fell at running speed; no FAIL; accepted and homed at the count it settled to.
static void test_replay_run1_flkl_curl()
{
    ReplayPlant plant;
    plant.play(2, kRun1FlklCurl, kRun1FlklCurlFrames);
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    const ReplayResult result = replayMove(calibration, plant, 2);
    TEST_ASSERT_TRUE(result.hallMs > 0 && result.hallMs <= 500);
    TEST_ASSERT_EQUAL_UINT(1, result.stalls.size());
    const uint32_t stall = result.stalls.front();
    const int finalCount = kRun1FlklCurl[kRun1FlklCurlFrames - 1].count;
    const int atStop = recordedCount(kRun1FlklCurl, kRun1FlklCurlFrames, stall);
    const std::string detail = "stall=" + std::to_string(stall) + " atStop=" + std::to_string(atStop) +
                               " final=" + std::to_string(finalCount) + "\n" + plant.output.text;
    TEST_ASSERT_TRUE_MESSAGE(atStop - finalCount >= 0 && atStop - finalCount <= 15, detail.c_str());
    // At the decision the last 200 ms fell at most a quarter of running (8 per 200 ms).
    const int fell = recordedCount(kRun1FlklCurl, kRun1FlklCurlFrames, stall - 200) - atStop;
    TEST_ASSERT_TRUE_MESSAGE(fell <= 3, detail.c_str());
    TEST_ASSERT_FALSE(result.failedWhileRecording);
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL"));
    TEST_ASSERT_TRUE(plant.output.find("CAL FRONT END FLKL curl").find(" by=stop cur=") != std::string::npos);
    const std::string home = "CAL FRONT HOME FLKL count=" +
        std::to_string(recordedCount(kRun1FlklCurl, kRun1FlklCurlFrames, stall + CAL_SETTLE_MS));
    TEST_ASSERT_TRUE(plant.output.has(home));
    TEST_ASSERT_TRUE(calibration.isJointCurled(2));
}

// Run 1, FLHL curl (log 14884-15135), after the FLKL curl above: hall within
// 0.6 s; the switch end is called after the recorded count stopped (it moved
// up to the recorded EN drop) and within 0.4 s of it; the probe's
// "open" is not consulted; no NotAtCurlEnd and no FAIL.
static void test_replay_run1_flhl_curl()
{
    ReplayPlant plant;
    plant.play(2, kRun1FlklCurl, kRun1FlklCurlFrames);
    plant.play(1, kRun1FlhlCurl, kRun1FlhlCurlFrames);
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    replayMove(calibration, plant, 2);
    const ReplayResult result = replayMove(calibration, plant, 1);
    TEST_ASSERT_TRUE(result.hallMs > 0 && result.hallMs <= 600);
    const uint32_t enDrop = recordedEnDropMs(kRun1FlhlCurl, kRun1FlhlCurlFrames);
    TEST_ASSERT_EQUAL_UINT32(3850, enDrop);
    uint32_t stoppedMs = 0;
    for (size_t i = 1; i < kRun1FlhlCurlFrames; ++i)
        if (kRun1FlhlCurl[i].count != kRun1FlhlCurl[i - 1].count) stoppedMs = static_cast<uint32_t>(i) * ROBOT_LOG_FRAME_MS;
    TEST_ASSERT_EQUAL_UINT(1, result.stalls.size());
    const std::string detail = "stall=" + std::to_string(result.stalls.front()) + " stopped=" +
                               std::to_string(stoppedMs) + "\n" + plant.output.text;
    TEST_ASSERT_TRUE_MESSAGE(result.stalls.front() > stoppedMs && result.stalls.front() <= stoppedMs + 400,
                             detail.c_str());
    TEST_ASSERT_TRUE(plant.output.find("CAL FRONT END FLHL curl").find(" by=switch cur=") != std::string::npos);
    TEST_ASSERT_FALSE(result.failedWhileRecording);
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL FLHL"));
    TEST_ASSERT_FALSE(plant.output.text.find("pot not connected") != std::string::npos);
    TEST_ASSERT_FALSE(plant.output.text.find("did not reach") != std::string::npos);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=-17"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));
}

// Run 2, FLKL curl (log 28784-29034): the old firmware stopped it at count 124
// while it still moved about 60 edges/s. No window ends before that stop.
static void test_replay_run2_flkl_curl()
{
    ReplayPlant plant;
    plant.play(2, kRun2FlklCurl, kRun2FlklCurlFrames);
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    const ReplayResult result = replayMove(calibration, plant, 2);
    uint32_t stopMs = 0;
    for (size_t i = 0; i < kRun2FlklCurlFrames && stopMs == 0; ++i)
        if (kRun2FlklCurl[i].count == 124) stopMs = static_cast<uint32_t>(i) * ROBOT_LOG_FRAME_MS;
    TEST_ASSERT_EQUAL_UINT32(3850, stopMs);
    TEST_ASSERT_FALSE(result.stalls.empty());
    for (uint32_t stall : result.stalls) TEST_ASSERT_TRUE(stall >= stopMs);
    TEST_ASSERT_TRUE(result.hallMs > 0 && result.hallMs <= 500);
    TEST_ASSERT_FALSE(result.failedWhileRecording);
}

// ------------------------------------------------------------- end detector

// Which drives run the end detector: hall-capable joints on quadrature builds
// (Unknown included) except the nudge out; a proven Pot and every joint on a
// build without quadrature keep the windows.
static void test_end_detector_runs_on_hall_drives_only()
{
    FakePlant hall;
    hall.useHall();
    LegCalibration a;
    TEST_ASSERT_TRUE(a.start(hall, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_TRUE(a.isDetectingEnd());
    runUntil(a, hall, CAL_DEFAULT_PWM, [&]() { return a.movePhase() != LegCalibrationMovePhase::Ramp; });
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Detect), static_cast<int>(a.movePhase()));

    FakePlant pot;
    LegCalibration b;
    TEST_ASSERT_TRUE(b.start(pot, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_FALSE(b.isDetectingEnd());

    FakePlant provenPot;
    provenPot.useHall();
    provenPot.kind[2] = JointSensor::Pot;
    provenPot.hallRod[2] = false;
    LegCalibration c;
    TEST_ASSERT_TRUE(c.start(provenPot, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    TEST_ASSERT_FALSE(c.isDetectingEnd());

    // Nudge out: its own bounds; the Return: the detector.
    FakePlant parked;
    parked.useHall();
    parked.extension[1] = 0;
    parked.moveEvery[1] = 3;
    LegCalibration d;
    TEST_ASSERT_TRUE(d.start(parked, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(d, parked, CAL_DEFAULT_PWM, [&]() { return d.moveStage() == LegCalibrationMoveStage::NudgeOut; });
    TEST_ASSERT_FALSE(d.isDetectingEnd());
    runUntil(d, parked, CAL_DEFAULT_PWM, [&]() { return d.moveStage() == LegCalibrationMoveStage::Return; });
    TEST_ASSERT_TRUE(d.isDetectingEnd());
}

// A full hall SEEK + SWEEP at 100 edges/s: every drive that runs into an
// actuator limit switch (hip curl, knee uncurl) stops at most 200 ms after the
// rod reaches it, every knee curl at most 350 ms after the leg stop; a press of
// a joint already at its end drives at most 1.05 s (was about 3 s).
static void test_hall_ends_stop_the_drive_promptly()
{
    FakePlant plant;
    plant.useHall();
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    uint32_t driveStart[6] = {};
    uint32_t reached[6] = {};
    bool atEndAtStart[6] = {};
    int switches = 0, stops = 0, presses = 0;
    uint32_t worstSwitch = 0, worstStop = 0, worstPress = 0;
    while (calibration.isRunning())
    {
        tick(calibration, plant, CAL_DEFAULT_PWM);
        for (int j = 0; j < 6; ++j)
        {
            const bool atLow = plant.extension[j] <= plant.low[j];
            const bool atHigh = plant.extension[j] >= plant.high[j];
            if (plant.pwm[j] != 0)
            {
                const bool atEnd = plant.pwm[j] < 0 ? atLow : atHigh;
                if (driveStart[j] == 0)
                {
                    driveStart[j] = plant.now;
                    atEndAtStart[j] = atEnd;
                    reached[j] = 0;
                }
                if (atEnd && reached[j] == 0) reached[j] = plant.now;
                continue;
            }
            if (driveStart[j] == 0) continue;
            const uint32_t stopped = plant.now;
            if (atEndAtStart[j])
            {
                ++presses;
                if (stopped - driveStart[j] > worstPress) worstPress = stopped - driveStart[j];
            }
            else if (reached[j] != 0 && atLow)
            {
                ++switches;
                if (stopped - reached[j] > worstSwitch) worstSwitch = stopped - reached[j];
            }
            else if (reached[j] != 0 && atHigh)
            {
                ++stops;
                if (stopped - reached[j] > worstStop) worstStop = stopped - reached[j];
            }
            driveStart[j] = 0;
        }
    }
    assertCompleted(calibration);
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL"));
    // SEEK: 2 hip curls; SWEEP: 2 hip curls back, 2 knee uncurls.
    TEST_ASSERT_EQUAL_INT(6, switches);
    // SEEK: 2 knee curls; SWEEP: 2 knee curls back.
    TEST_ASSERT_EQUAL_INT(4, stops);
    // SWEEP steps 0 and 1 of each leg.
    TEST_ASSERT_EQUAL_INT(4, presses);
    TEST_ASSERT_TRUE_MESSAGE(worstSwitch <= 200, std::to_string(worstSwitch).c_str());
    TEST_ASSERT_TRUE_MESSAGE(worstStop <= 350, std::to_string(worstStop).c_str());
    TEST_ASSERT_TRUE_MESSAGE(worstPress <= 1050, std::to_string(worstPress).c_str());
}

// Stuck mid-stroke: a stall with no end signature after a running phase fails
// the pass before the partner joint moves, the joint's hall home is cleared and
// nothing is saved. A knee uncurl (FRKL at 771 edges, 18 short of its switch as
// on 2026-11-24), a hip curl and a hip sweep.
static void test_stuck_mid_stroke_fails_the_pass()
{
    {
        HallSweep sweep;
        sweep.plant.speed = 1;
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.plant.pwm[5] < 0 && sweep.plant.extension[5] <= 1000 - 771;
        });
        sweep.plant.stuck[5] = true;
        const size_t before = sweep.plant.drives.size();
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::StuckMidStroke);
        const std::string expected =
            "CAL FRONT FAIL FRKL stuck mid-stroke after 771 edges at PWM 128: raise PWM or check load";
        TEST_ASSERT_EQUAL_STRING(expected.c_str(), sweep.plant.output.last().c_str());
        TEST_ASSERT_TRUE(expected.size() - std::string("CAL FRONT FAIL ").size() <= CAL_ROBOT_REASON_MAX_CHARS);
        TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FRKL uncurl hall=-771 travel=-771", "stuck"));
        TEST_ASSERT_TRUE(sweep.plant.homeLost[5]);
        TEST_ASSERT_FALSE(drovePartnerAfter(sweep.plant, before, 4));
        TEST_ASSERT_EQUAL_INT(1, sweep.plant.output.count("CAL FRONT MOVE FRKL uncurl"));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
        TEST_ASSERT_EQUAL_INT(static_cast<int>(CalibrationEnd::Stuck), static_cast<int>(sweep.calibration.lastEnd()));
    }
    {
        // Hip curl blocked 200 edges in (an obstacle: no switch drop).
        FakePlant plant;
        plant.useHall();
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return plant.pwm[1] < 0 && plant.extension[1] <= 300;
        });
        plant.stuck[1] = true;
        const size_t before = plant.drives.size();
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        assertFailure(calibration, LegCalibrationFailure::StuckMidStroke);
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLHL stuck mid-stroke after 200 edges at PWM 128: raise PWM or check load",
            plant.output.last().c_str());
        TEST_ASSERT_FALSE(drovePartnerAfter(plant, before, 5));
        TEST_ASSERT_FALSE(calibration.isJointCurled(1));
        TEST_ASSERT_FALSE(plant.homed[1]);
    }
    {
        // Hip sweep frozen 300 edges out: stuck, not "short of its sweep".
        HallSweep sweep;
        sweep.plant.speed = 1;
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.plant.pwm[1] > 0 && sweep.plant.extension[1] >= 300;
        });
        sweep.plant.stuck[1] = true;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::StuckMidStroke);
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLHL stuck mid-stroke after 300 edges at PWM 128: raise PWM or check load",
            sweep.plant.output.last().c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_FALSE(sweep.plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    }
}

// The FR knee's curl end at PWM 80 (2026-11-24): the current rose only about
// 10% at the leg stop. That is no stop signature: stuck mid-stroke, the hip
// never moves.
static void test_weak_mechanical_end_is_stuck_mid_stroke()
{
    FakePlant plant;
    plant.useHall();
    plant.currentStall[2] = 121;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::StuckMidStroke);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLKL stuck mid-stroke after 500 edges at PWM 128: raise PWM or check load",
        plant.output.last().c_str());
    for (const Drive &drive : plant.drives) TEST_ASSERT_EQUAL_UINT(2, drive.joint);
}

// Compliant knee stops (the last 20 edges slow down while the current rises)
// with noisy current on every joint (+-20 tenths, ten seeds): the run completes
// with every end found by its signature; each knee is homed at most 10 edges
// short of the deepest its creep would reach.
static void test_compliant_stops_and_noisy_current_complete_the_run()
{
    for (uint32_t seed = 1; seed <= 10; ++seed)
    {
        FakePlant plant;
        plant.useHall();
        plant.taperEdges[2] = 20;
        plant.taperEdges[5] = 20;
        plant.currentNoise = 20;
        plant.noiseSeed = seed;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        const std::string label = "seed " + std::to_string(seed) + "\n" + plant.output.text;
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(LegCalibrationOutcome::Completed),
                                      static_cast<int>(calibration.outcome()), label.c_str());
        // Per knee: the SEEK curl, the SWEEP press (step 0) and the curl back.
        TEST_ASSERT_EQUAL_INT_MESSAGE(6, plant.output.count("CAL FRONT END FLKL curl") +
                                             plant.output.count("CAL FRONT END FRKL curl"), label.c_str());
        for (uint8_t knee : {2, 5})
        {
            const long home = -plant.home[knee];
            TEST_ASSERT_TRUE_MESSAGE(home <= 1000 && home >= 990, label.c_str());
            TEST_ASSERT_TRUE_MESSAGE(plant.minStop[knee] <= -980 && plant.maxStop[knee] == 0, label.c_str());
        }
        // Signature ends: 4 hip curls and 2 knee uncurls by switch, 4 knee curls by stop.
        int switches = 0, stops = 0;
        for (const std::string &line : plant.output.lines())
        {
            switches += line.find(" by=switch cur=") != std::string::npos;
            stops += line.find(" by=stop cur=") != std::string::npos;
        }
        TEST_ASSERT_EQUAL_INT_MESSAGE(6, switches, label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(4, stops, label.c_str());
    }
}

// A joint that keeps creeping (an edge every 400 ms after a running phase) is
// never stuck: the 120 s move timeout ends it.
static void test_steady_creep_runs_to_the_move_timeout()
{
    FakePlant plant;
    plant.useHall();
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return plant.extension[2] >= 650; });
    TEST_ASSERT_TRUE(calibration.endDetector().isArmed());
    plant.moveEvery[2] = 40;
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::NoEndOfTravel);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL curl: no end of travel in 120 s", plant.output.last().c_str());
    TEST_ASSERT_TRUE(plant.extension[2] < 1000);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_constants_match_the_agreed_tuning);
    RUN_TEST(test_step_tables_follow_curl_first_order);
    RUN_TEST(test_partner_ready_table);
    RUN_TEST(test_every_failure_reason_fits_the_leader_echo);
    RUN_TEST(test_hall_run_lines_use_only_known_event_words);
    RUN_TEST(test_full_run_curls_then_sweeps_and_saves_hips_and_knees_only);
    RUN_TEST(test_ramp_climbs_from_zero_and_never_exceeds_requested_pwm);
    RUN_TEST(test_hip_ramps_with_negative_pwm_to_curl);
    RUN_TEST(test_seek_only_never_saves_and_sweep_only_does);
    RUN_TEST(test_invalid_requests_and_options_do_not_start);
    RUN_TEST(test_options_pass_through_and_default);
    RUN_TEST(test_open_pot_fails_before_driving_only_when_proven_pot);
    RUN_TEST(test_move_without_end_times_out_and_stops);
    RUN_TEST(test_uncurl_that_barely_moves_fails);
    RUN_TEST(test_reversed_pot_fails_pot_wrong_way);
    RUN_TEST(test_pot_window_follows_the_drive_direction);
    RUN_TEST(test_abort_stops_everything_and_saves_nothing);
    RUN_TEST(test_abort_keeps_hall_homes);
    RUN_TEST(test_sweep_curl_back_that_never_leaves_the_uncurl_end_fails);
    RUN_TEST(test_sweep_curl_back_that_stops_part_way_fails);
    RUN_TEST(test_sweep_knee_curl_back_that_stalls_fails);
    RUN_TEST(test_knee_curl_that_never_moves_fails_no_motion);
    RUN_TEST(test_pot_curl_without_motion_needs_a_proven_curl_end);
    RUN_TEST(test_pot_curl_with_motion_against_its_proven_end);
    RUN_TEST(test_sweep_never_saves_a_span_below_the_minimum_travel);
    RUN_TEST(test_hall_full_run_homes_and_saves_canonical_strokes);
    RUN_TEST(test_hall_evidence_in_ramp_and_detect);
    RUN_TEST(test_hall_wrong_way_fails_in_ramp_and_detect);
    RUN_TEST(test_unknown_to_hall_in_detect_keeps_the_detector);
    RUN_TEST(test_pot_to_hall_in_windows_restarts_the_window);
    RUN_TEST(test_hall_still_threshold_three_edges);
    RUN_TEST(test_hall_curl_against_its_home);
    RUN_TEST(test_hall_curl_without_motion_at_its_home);
    RUN_TEST(test_curl_without_motion_short_of_a_home_proven_this_boot_fails);
    RUN_TEST(test_hip_nudge_by_edges_then_return_homes_it);
    RUN_TEST(test_hip_nudge_by_timeout_with_evidence);
    RUN_TEST(test_hip_nudge_without_evidence_returns_blind_then_fails);
    RUN_TEST(test_stored_hall_hip_nudge_without_edges_fails);
    RUN_TEST(test_nudge_ignores_counts_without_quadrature);
    RUN_TEST(test_pot_wrong_way_at_the_nudge_end);
    RUN_TEST(test_hip_nudge_no_motor_current_from_the_nudge);
    RUN_TEST(test_hip_nudge_return_short_or_without_motion_fails);
    RUN_TEST(test_hall_wrong_way_in_nudge_and_return);
    RUN_TEST(test_pot_hip_nudge_by_counts);
    RUN_TEST(test_knee_nudge_only_with_the_option);
    RUN_TEST(test_blind_stored_knee_nudge_runs_to_its_cap_then_fails);
    RUN_TEST(test_knee_nudge_without_edges_below_the_default_pwm_names_the_speed);
    RUN_TEST(test_slow_knee_nudge_is_capped_and_reported_as_barely_moved);
    RUN_TEST(test_healthy_stored_knee_nudge_and_short_return_are_accepted);
    RUN_TEST(test_nudge_return_bounds_for_knee_and_hip);
    RUN_TEST(test_failure_or_abort_during_a_nudge_warns_the_joint_is_out);
    RUN_TEST(test_late_pot_classification_repeats_the_step);
    RUN_TEST(test_stored_hall_that_is_now_a_pot);
    RUN_TEST(test_second_late_classification_does_not_repeat);
    RUN_TEST(test_late_pot_classification_wrong_way_fails);
    RUN_TEST(test_rail_flip_is_not_pot_evidence);
    RUN_TEST(test_pot_evidence_is_the_snapshots_not_the_smoothed_reading);
    RUN_TEST(test_classification_at_the_nudge_and_the_return);
    RUN_TEST(test_hall_hip_sweep_stops_at_its_fixed_target);
    RUN_TEST(test_hall_hip_sweep_target_survives_relaxation);
    RUN_TEST(test_hall_hip_sweep_short_of_its_target_fails);
    RUN_TEST(test_pot_hip_keeps_the_full_sweep);
    RUN_TEST(test_sweep_uncurl_must_travel_the_minimum_stroke);
    RUN_TEST(test_sweep_curl_back_must_leave_the_uncurl_end_with_its_home);
    RUN_TEST(test_sweep_curl_back_short_of_its_home_fails);
    RUN_TEST(test_sweep_stroke_out_of_range);
    RUN_TEST(test_sweep_stroke_against_the_saved_stroke);
    RUN_TEST(test_sweep_deeper_curl_back_rehomes_and_recomputes_the_stroke);
    RUN_TEST(test_fail_clears_only_the_failed_hall_home);
    RUN_TEST(test_replay_run1_flkl_curl);
    RUN_TEST(test_replay_run1_flhl_curl);
    RUN_TEST(test_replay_run2_flkl_curl);
    RUN_TEST(test_end_detector_runs_on_hall_drives_only);
    RUN_TEST(test_hall_ends_stop_the_drive_promptly);
    RUN_TEST(test_stuck_mid_stroke_fails_the_pass);
    RUN_TEST(test_weak_mechanical_end_is_stuck_mid_stroke);
    RUN_TEST(test_compliant_stops_and_noisy_current_complete_the_run);
    RUN_TEST(test_steady_creep_runs_to_the_move_timeout);
    return UNITY_END();
}

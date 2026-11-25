#include <cstdint>
#include <cstdio>
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

    // An END line exactly `line` plus the end detector's " by=<by> cur=a/b",
    // then on a drive into a limit switch (a retract, a hip extend) " floor=f".
    bool hasEnd(const std::string &line, const char *by) const
    {
        const std::string prefix = line + " by=" + by + " cur=";
        for (const std::string &candidate : lines())
        {
            if (candidate.compare(0, prefix.size(), prefix) != 0) continue;
            std::string rest = candidate.substr(prefix.size());
            const size_t floorAt = rest.find(" floor=");
            if (floorAt != std::string::npos)
            {
                const std::string floor = rest.substr(floorAt + 7);
                if (floor.empty() || floor.find_first_not_of("0123456789") != std::string::npos) return false;
                rest = rest.substr(0, floorAt);
            }
            const size_t slash = rest.find('/');
            if (slash == std::string::npos || slash == 0 || slash + 1 == rest.size()) return false;
            return rest.find_first_not_of("0123456789/") == std::string::npos;
        }
        return false;
    }

    // An END line exactly `line` plus a switch-end drive's plain stall at the
    // switch-open level: " open=<current>/<floor>".
    bool hasOpenEnd(const std::string &line) const
    {
        const std::string prefix = line + " open=";
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
// robot (2026-11-24, fix round). Undriven: 0. At a limit switch the motor no
// longer conducts and the IS reads its offset, currentOpen, flat: a rod driven
// to retract at `low`, and one driven to extend at `high` when extendSwitch is
// set (a hall hip's extend switch). Otherwise the motor conducts and the level
// grows with |pwm| from that offset (so the early ramp reads near it, which is
// the end detector's floor): running, currentOpen + (currentRun - currentOpen)
// x |pwm| / CAL_DEFAULT_PWM. Stalled (stuck, blocked by stallSign, breaking
// free, or pressed at `high` without a switch: a knee's stop, an obstacle) the
// same scaling to currentStall, plus a wobble of +-currentWobble in 100 ms
// halves (a stalled motor's current is never flat). A rod with taperEdges > 0
// has a compliant stop: over its last taperEdges before `high` it slows in
// proportion and its current ramps from running to stalled. currentNoise adds
// a seeded uniform +-noise to every read while the motor conducts.
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
    int currentWobble[6] = {10, 10, 10, 10, 10, 10};
    bool extendSwitch[6] = {};
    int taperEdges[6] = {};
    // The table-safe envelope check in tick() (off for a test that drives the
    // knee out of its band on purpose).
    bool checkEnvelope = true;
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

    // Rev 3: hips and knees are hall rods (hips curl to 0, knees to 1000). A
    // knee's stroke is 1000 edges, its uncurl end a limit switch, its curl end
    // the leg's stop. A hip's stroke is FL's CAL_HIP_STROKE_EDGES, a limit
    // switch at each end. Every kind starts Unknown with no limits.
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
            high[joint] = joint % 3 == 1 ? CAL_HIP_STROKE_EDGES : 1000;
            extendSwitch[joint] = joint % 3 == 1;
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
        if ((direction < 0 && extension[joint] <= low[joint]) ||
            (direction > 0 && extendSwitch[joint] && extension[joint] >= high[joint]))
            return static_cast<uint16_t>(currentOpen[joint]);  // switch open: flat offset
        const int magnitude = std::abs(pwm[joint]);
        const int running = currentOpen[joint] + (currentRun[joint] - currentOpen[joint]) * magnitude / CAL_DEFAULT_PWM;
        const int stalled = currentOpen[joint] + (currentStall[joint] - currentOpen[joint]) * magnitude / CAL_DEFAULT_PWM;
        const int wobble = (now / 100) % 2 == 0 ? currentWobble[joint] : -currentWobble[joint];
        const bool blocked = stuck[joint] || direction == stallSign[joint] ||
            (direction == breakawaySign[joint] && now - driveStartMs[joint] < breakawayMs[joint]);
        int value = running;
        if (blocked || (direction > 0 && extension[joint] >= high[joint]))
            value = stalled + wobble;
        else if (direction > 0 && high[joint] - extension[joint] < taperEdges[joint])
            value = running + (stalled - running) *
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
// yaw, never above the requested PWM; a hip moves only with its knee curled or
// parked, and sweeps only with it parked; a knee moves without its hip curled
// only on a curl of a SEEK or of SWEEP step 0 (not a park), or as a
// CAL_OPT_KNEE_NUDGE nudge. On a table-safe SWEEP the table-safe envelope
// (FW2 spec, FL model) holds: a hall hip above 560 edges has its knee inside
// its park band, and it never goes beyond its 840-edge sweep plus 27 edges of
// coast (unless the hollow-stand option is set).
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
        {
            const uint8_t kneeBit = static_cast<uint8_t>(1U << (joint + 1));
            TEST_ASSERT_TRUE(calibration.isJointCurled(joint + 1) || (calibration.parkedMask() & kneeBit) != 0);
            if (calibration.moveStage() == LegCalibrationMoveStage::Normal &&
                calibration.currentAction() == LegCalibrationAction::Sweep)
                TEST_ASSERT_TRUE((calibration.parkedMask() & kneeBit) != 0);
        }
        else if (!calibration.isJointCurled(joint - 1))
        {
            const bool nudge = calibration.moveStage() == LegCalibrationMoveStage::NudgeOut &&
                               (calibration.options() & CAL_OPT_KNEE_NUDGE) != 0;
            const bool freeCurl = calibration.isCurrentMoveCurl() &&
                                  calibration.currentAction() == LegCalibrationAction::Seek &&
                                  (calibration.currentPass() == CAL_PASS_SEEK || calibration.currentLegStep() == 0);
            TEST_ASSERT_TRUE(nudge || freeCurl);
        }
    }
    if (!plant.checkEnvelope || !plant.quadrature || calibration.currentPass() != CAL_PASS_SWEEP)
        return;
    const bool hollow = (calibration.options() & CAL_OPT_HOLLOW_STAND) != 0;
    for (uint8_t hip : {1, 4})
    {
        if (plant.kind[hip] != JointSensor::Hall || !plant.homed[hip]) continue;
        const int x = plant.position(hip);
        if (!hollow) TEST_ASSERT_TRUE_MESSAGE(x <= CAL_HIP_SWEEP_EDGES + 27, std::to_string(x).c_str());
        if (x <= 560) continue;
        const uint8_t knee = static_cast<uint8_t>(hip + 1);
        TEST_ASSERT_TRUE(plant.homed[knee]);
        TEST_ASSERT_TRUE_MESSAGE(isLegCalibrationParkInBand(plant.position(knee), hollow),
                                 std::to_string(plant.position(knee)).c_str());
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
    // Owner decision F4 (2026-11-24): below about 110 drives stall mid-stroke.
    TEST_ASSERT_EQUAL_INT(110, CAL_PWM_MIN);
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
    // The table-safe SWEEP (FL's measured geometry, fix round 2026-11-24): the
    // knee parks 264 out of its stop (band 244-280), the hip sweeps by count to
    // 840 and must re-home within 10 edges; its saved range is FL's 1069-edge
    // stroke; the hollow-stand option parks at 192 (157-232) and must find the
    // extend switch within 20 edges of 1069.
    TEST_ASSERT_EQUAL_INT(840, CAL_HIP_SWEEP_EDGES);
    TEST_ASSERT_EQUAL_INT(1069, CAL_HIP_STROKE_EDGES);
    TEST_ASSERT_EQUAL_INT(20, CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES);
    TEST_ASSERT_EQUAL_INT(10, CAL_HIP_REHOME_TOLERANCE_EDGES);
    TEST_ASSERT_EQUAL_INT(264, CAL_KNEE_PARK_EDGES);
    TEST_ASSERT_EQUAL_INT(244, CAL_KNEE_PARK_BAND_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(280, CAL_KNEE_PARK_BAND_MAX_EDGES);
    TEST_ASSERT_EQUAL_INT(192, CAL_KNEE_HOLLOW_PARK_EDGES);
    TEST_ASSERT_EQUAL_INT(157, CAL_KNEE_HOLLOW_BAND_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(232, CAL_KNEE_HOLLOW_BAND_MAX_EDGES);
    TEST_ASSERT_EQUAL_UINT(0x02, CAL_OPT_HOLLOW_STAND);
    TEST_ASSERT_EQUAL_UINT(CAL_OPT_KNEE_NUDGE, CAL_DEFAULT_OPTIONS);  // hollow stand off by default
    TEST_ASSERT_EQUAL_INT(24, CAL_HIP_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_UINT32(2000, CAL_HIP_NUDGE_MAX_MS);
    TEST_ASSERT_EQUAL_INT(128, CAL_HIP_NUDGE_MAX_PWM);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_HIP_NUDGE_BLIND_MS);
    TEST_ASSERT_EQUAL_INT(12, CAL_KNEE_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_KNEE_NUDGE_MAX_MS);
    TEST_ASSERT_EQUAL_INT(128, CAL_KNEE_NUDGE_MAX_PWM);
    TEST_ASSERT_EQUAL_INT(2, CAL_KNEE_RETURN_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(30, CAL_POT_NUDGE_COUNTS);
    TEST_ASSERT_EQUAL_INT(-1, ACTUATOR_HALL_EXTEND_COUNT_SIGN);
}

// One leg's steps (joint offset within the leg: 1 hip, 2 knee). SEEK: knee
// curl, hip curl, knee curl (R3). SWEEP of a hall-hip leg, table-safe: knee
// curl, hip curl, knee uncurl, knee park, hip sweep, hip curl (completes the
// hip), knee curl (completes the knee). SWEEP of a pot-hip leg: the earlier
// six steps. The SEEK table does not depend on tableSafe.
static void test_step_tables_follow_curl_first_order()
{
    using A = LegCalibrationAction;
    TEST_ASSERT_EQUAL_UINT(3, legCalibrationLegStepCount(CAL_PASS_SEEK, false));
    TEST_ASSERT_EQUAL_UINT(3, legCalibrationLegStepCount(CAL_PASS_SEEK, true));
    TEST_ASSERT_EQUAL_UINT(7, legCalibrationLegStepCount(CAL_PASS_SWEEP, true));
    TEST_ASSERT_EQUAL_UINT(6, legCalibrationLegStepCount(CAL_PASS_SWEEP, false));
    const uint8_t seekJoints[] = {2, 1, 2};
    for (bool tableSafe : {false, true})
        for (uint8_t step = 0; step < 3; ++step)
        {
            const LegCalibrationStep entry = legCalibrationStep(CAL_PASS_SEEK, step, tableSafe);
            TEST_ASSERT_EQUAL_UINT(seekJoints[step], entry.joint);
            TEST_ASSERT_TRUE(entry.curl);
            TEST_ASSERT_FALSE(entry.completesJoint);
            TEST_ASSERT_EQUAL_INT(static_cast<int>(A::Seek), static_cast<int>(entry.action));
        }
    const uint8_t safeJoints[] = {2, 1, 2, 2, 1, 1, 2};
    const bool safeCurl[] = {true, true, false, true, false, true, true};
    const bool safeCompletes[] = {false, false, false, false, false, true, true};
    const A safeAction[] = {A::Seek, A::Seek, A::Seek, A::Park, A::Sweep, A::Seek, A::Seek};
    for (uint8_t step = 0; step < 7; ++step)
    {
        const LegCalibrationStep entry = legCalibrationStep(CAL_PASS_SWEEP, step, true);
        const std::string label = "table-safe step " + std::to_string(step);
        TEST_ASSERT_EQUAL_UINT_MESSAGE(safeJoints[step], entry.joint, label.c_str());
        TEST_ASSERT_EQUAL_MESSAGE(safeCurl[step], entry.curl, label.c_str());
        TEST_ASSERT_EQUAL_MESSAGE(safeCompletes[step], entry.completesJoint, label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(safeAction[step]), static_cast<int>(entry.action),
                                      label.c_str());
    }
    const uint8_t potJoints[] = {2, 1, 1, 1, 2, 2};
    const bool potCurl[] = {true, true, false, true, false, true};
    const bool potCompletes[] = {false, false, false, true, false, true};
    for (uint8_t step = 0; step < 6; ++step)
    {
        const LegCalibrationStep entry = legCalibrationStep(CAL_PASS_SWEEP, step, false);
        const std::string label = "pot-hip step " + std::to_string(step);
        TEST_ASSERT_EQUAL_UINT_MESSAGE(potJoints[step], entry.joint, label.c_str());
        TEST_ASSERT_EQUAL_MESSAGE(potCurl[step], entry.curl, label.c_str());
        TEST_ASSERT_EQUAL_MESSAGE(potCompletes[step], entry.completesJoint, label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(A::Seek), static_cast<int>(entry.action), label.c_str());
    }
    // The two SWEEP orders agree up to the hip's curl (the sequencer picks the
    // order at step 2, once the hip's kind is known).
    for (uint8_t step = 0; step < 2; ++step)
    {
        TEST_ASSERT_EQUAL_UINT(legCalibrationStep(CAL_PASS_SWEEP, step, true).joint,
                               legCalibrationStep(CAL_PASS_SWEEP, step, false).joint);
        TEST_ASSERT_TRUE(legCalibrationStep(CAL_PASS_SWEEP, step, false).curl);
    }
    TEST_ASSERT_EQUAL_UINT(5, legCalibrationKneeOf(4));
    TEST_ASSERT_EQUAL_UINT(2, legCalibrationKneeOf(2));
    TEST_ASSERT_EQUAL_INT(-264, legCalibrationParkTarget(false));
    TEST_ASSERT_EQUAL_INT(-192, legCalibrationParkTarget(true));
    TEST_ASSERT_TRUE(isLegCalibrationParkInBand(-280, false) && isLegCalibrationParkInBand(-244, false));
    TEST_ASSERT_FALSE(isLegCalibrationParkInBand(-281, false) || isLegCalibrationParkInBand(-243, false));
    TEST_ASSERT_TRUE(isLegCalibrationParkInBand(-232, true) && isLegCalibrationParkInBand(-157, true));
    TEST_ASSERT_FALSE(isLegCalibrationParkInBand(-233, true) || isLegCalibrationParkInBand(-156, true));
    TEST_ASSERT_EQUAL_INT(-1, legCalibrationPwmSign(1, true));
    TEST_ASSERT_EQUAL_INT(1, legCalibrationPwmSign(1, false));
    TEST_ASSERT_EQUAL_INT(1, legCalibrationPwmSign(5, true));
    TEST_ASSERT_EQUAL_INT(-1, legCalibrationPwmSign(5, false));
}

// Every joint, curl and uncurl, nudge or not, every action, option on and off,
// every curled mask and every parked mask. A hip: its sweep only with its knee
// parked; any other uncurl (a pot hip's) only with its knee curled; a curl,
// return, retract or nudge with its knee curled or parked. A knee: a curl is
// free unless it is a park or the knee is parked; a park, an uncurl and a curl
// back from the park need the hip curled; a nudge needs the option (or the hip).
static void test_partner_ready_table()
{
    using A = LegCalibrationAction;
    for (uint8_t joint = 0; joint < 6; ++joint)
    {
        const uint8_t hipBit = static_cast<uint8_t>(1U << ((joint / 3) * 3 + 1));
        const uint8_t kneeBit = static_cast<uint8_t>(1U << ((joint / 3) * 3 + 2));
        for (unsigned curled = 0; curled < 64; ++curled)
            for (unsigned parked = 0; parked < 64; ++parked)
                for (A action : {A::Seek, A::Park, A::Sweep})
                    for (int curl = 0; curl < 2; ++curl)
                        for (int nudge = 0; nudge < 2; ++nudge)
                            for (int option = 0; option < 2; ++option)
                            {
                                const bool hipCurled = (curled & hipBit) != 0;
                                const bool kneeCurled = (curled & kneeBit) != 0;
                                const bool kneeParked = (parked & kneeBit) != 0;
                                bool expected;
                                if (joint % 3 == 1)
                                {
                                    if (action == A::Sweep)
                                        expected = kneeParked;
                                    else if (!curl && !nudge)
                                        expected = kneeCurled;
                                    else
                                        expected = kneeCurled || kneeParked;
                                }
                                else if (curl && action != A::Park && !kneeParked)
                                    expected = true;
                                else if (nudge && option)
                                    expected = true;
                                else
                                    expected = hipCurled;
                                TEST_ASSERT_EQUAL(expected, legCalibrationPartnerReady(
                                    static_cast<uint8_t>(curled), static_cast<uint8_t>(parked), joint,
                                    curl != 0, nudge != 0, action, option ? CAL_OPT_KNEE_NUDGE : 0));
                            }
    }
    // Spelled out. A hip's sweep needs its knee parked: curled is not enough.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x04, 0x00, 1, false, false, A::Sweep, 0));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 0x04, 1, false, false, A::Sweep, 0));
    // A pot hip's uncurl needs its knee curled: parked is not enough.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x04, 1, false, false, A::Seek, 0));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x04, 0x00, 1, false, false, A::Seek, 0));
    // A hip curl (its return after the sweep, a retract) or nudge: knee curled or parked.
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 0x20, 4, true, false, A::Seek, 0));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 0x20, 4, false, true, A::Seek, 0));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x20, 0x00, 4, false, true, A::Seek, 0));
    // A hip never moves, even as a nudge, with its knee neither curled nor parked.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x02, 0x00, 1, true, false, A::Seek, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x00, 4, false, true, A::Seek, CAL_OPT_KNEE_NUDGE));
    // A knee curls freely, unless it is parked (its curl back from the park) or
    // the drive is its park: those need its hip curled.
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 0x00, 2, true, false, A::Seek, 0));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x04, 2, true, false, A::Seek, 0));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x02, 0x04, 2, true, false, A::Seek, 0));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x00, 2, true, false, A::Park, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x02, 0x00, 2, true, false, A::Park, 0));
    // A knee leaves its curl end before its hip only as an opted-in nudge.
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x00, 2, false, true, A::Seek, 0));
    TEST_ASSERT_FALSE(legCalibrationPartnerReady(0x00, 0x00, 2, false, false, A::Seek, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x00, 0x00, 2, false, true, A::Seek, CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(legCalibrationPartnerReady(0x02, 0x00, 2, false, false, A::Seek, 0));
}

// Every reason fits CAL_ROBOT_REASON_MAX_CHARS with a 4-character joint name,
// PWM 150 and 5-digit signed numbers, in every unit and direction. value2 takes
// every value the sequencer passes for it: StrokeShort's saved stroke (any
// int16), SweepShort's sweep target (840, or 1089 for the hollow stand),
// LostCounts' expected switch (0 or 1069), KneeOutOfBand's band flag (0 or 1),
// ParkShort's park target (-264 or -192). SweepShort's own number is the
// shortfall of a sweep that ended before its target, having started at its
// home (more than 11 edges back would have failed WrongDirection first):
// 1 .. target + 11.
static void test_every_failure_reason_fits_the_leader_echo()
{
    using F = LegCalibrationFailure;
    const F failures[] = {
        F::PotNotConnected, F::NoEndOfTravel, F::BarelyMoved, F::NotAtCurlEnd, F::WrongDirection,
        F::PotWrongWay, F::NoFeedback, F::NoMotorCurrent, F::NoMotion, F::ShortOfCurlEnd,
        F::NoCurlBack, F::StrokeOutOfRange, F::StrokeShort, F::SweepShort, F::StuckMidStroke,
        F::PartnerNotCurled, F::Aborted, F::LostCounts, F::KneeOutOfBand, F::ParkShort,
        F::NotRetracted, F::TableSafeNeedsHall,
    };
    const JointSensor kinds[] = {JointSensor::Unknown, JointSensor::Pot, JointSensor::Hall};
    const int sweepTargets[] = {CAL_HIP_SWEEP_EDGES, CAL_HIP_STROKE_EDGES + CAL_HIP_EXTEND_SWITCH_TOLERANCE_EDGES};
    size_t longest = 0;
    for (F failure : failures)
    {
        std::vector<int> seconds = {0};
        if (failure == F::StrokeShort) seconds = {-32768, 32767};
        if (failure == F::SweepShort) seconds = {sweepTargets[0], sweepTargets[1]};
        if (failure == F::LostCounts) seconds = {0, CAL_HIP_STROKE_EDGES};
        if (failure == F::KneeOutOfBand) seconds = {0, 1};
        if (failure == F::ParkShort) seconds = {legCalibrationParkTarget(false), legCalibrationParkTarget(true)};
        for (int value2 : seconds)
        {
            std::vector<int32_t> values = {-99999, 99999};
            if (failure == F::SweepShort) values = {1, value2 + CAL_HALL_DETECT_EDGES - 1};
            for (JointSensor kind : kinds)
                for (int curl = 0; curl < 2; ++curl)
                    for (int32_t value : values)
                    {
                        FakeOutput out;
                        printLegCalibrationFailureReason(out, failure, "FRKL", kind, curl != 0,
                            failure == F::Aborted ? 'H' : value, value2, CAL_PWM_MAX);
                        TEST_ASSERT_TRUE(out.text.size() > 4);
                        TEST_ASSERT_TRUE_MESSAGE(out.text.size() <= CAL_ROBOT_REASON_MAX_CHARS, out.text.c_str());
                        if (out.text.size() > longest) longest = out.text.size();
                    }
        }
    }
    // The hollow-stand SweepShort at its longest ("... 1100 edges short of its
    // 1089-edge sweep: obstacle? check its home") is exactly the limit.
    TEST_ASSERT_EQUAL_UINT(78, longest);

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
        {LegCalibrationFailure::SweepShort, JointSensor::Hall, 280, 840,
         "FRKL stopped 280 edges short of its 840-edge sweep: obstacle? check its home"},
        {LegCalibrationFailure::LostCounts, JointSensor::Hall, 15, 0,
         "FRKL lost counts: switch at 15 edges, expected 0"},
        {LegCalibrationFailure::LostCounts, JointSensor::Hall, 1000, 1069,
         "FRKL lost counts: switch at 1000 edges, expected 1069"},
        {LegCalibrationFailure::KneeOutOfBand, JointSensor::Hall, -300, 0,
         "FRKL at -300 edges, outside its park band -280..-244"},
        {LegCalibrationFailure::KneeOutOfBand, JointSensor::Hall, -150, 1,
         "FRKL at -150 edges, outside its park band -232..-157"},
        {LegCalibrationFailure::ParkShort, JointSensor::Hall, 150, -264,
         "FRKL stopped 150 edges short of its park at -264: obstacle?"},
        {LegCalibrationFailure::NotRetracted, JointSensor::Hall, 412, 0,
         "FRKL not back at its curl switch (hall=412): jog it home"},
        {LegCalibrationFailure::TableSafeNeedsHall, JointSensor::Hall, 0, 0,
         "FRKL and its knee must both be hall encoders"},
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
    const char *known[] = {"PASS ", "PASS_DONE ", "MOVE ", "END ", "SENSOR ", "HOME ", "WARN ", "JOINT ", "PARK "};
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
            // "MOVE <j> curl|uncurl|nudge", then " park" (a knee's park), " sweep"
            // (a hip's table-safe sweep) or " retract" (a hip after an obstacle).
            const std::string rest = event.substr(event.find(' ', 5) + 1);
            TEST_ASSERT_TRUE_MESSAGE(rest == "curl" || rest == "uncurl" || rest == "nudge" ||
                                         rest == "curl park" || rest == "uncurl sweep" || rest == "curl retract",
                                     line.c_str());
        }
    }
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLKL curl park"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL uncurl sweep"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT PARK FLKL hall=-264 target=-264 (FL values)"));
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
    // SEEK per leg: knee, hip, knee again (R3). A pot leg's SWEEP keeps the
    // earlier six steps (no park, no table-safe sweep).
    const std::vector<std::string> expectedMoves = {
        "CAL FRONT MOVE FLKL curl", "CAL FRONT MOVE FLHL curl", "CAL FRONT MOVE FLKL curl",
        "CAL FRONT MOVE FRKL curl", "CAL FRONT MOVE FRHL curl", "CAL FRONT MOVE FRKL curl",
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
    // 113: the lowest valid PWMs are 110-150; 113 is not a multiple of the step.
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, 113, 0));
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
    TEST_ASSERT_EQUAL_UINT(23, plant.drives.size());
    for (size_t i = 0; i + 1 < plant.drives.size(); ++i)
        TEST_ASSERT_EQUAL_INT(CAL_PWM_RAMP_STEP * static_cast<int>(i + 1), plant.drives[i].pwm);
    TEST_ASSERT_EQUAL_INT(113, plant.drives.back().pwm);
    TEST_ASSERT_EQUAL_INT(113, calibration.rampPwm());
    runToEnd(calibration, plant, 113);
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
    TEST_ASSERT_EQUAL_UINT(6, moveLines(seekPlant.output).size());  // 3 per leg

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
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0x04));
    TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0x80 | CAL_OPT_KNEE_NUDGE));
    TEST_ASSERT_TRUE(plant.output.text.empty());
    TEST_ASSERT_EQUAL_INT(0, plant.stopAllCount);
    TEST_ASSERT_FALSE(calibration.isRunning());
    TEST_ASSERT_TRUE(isValidCalibrationRequest(CAL_PASS_SWEEP, CAL_PWM_MAX));
    // PWM below 110 is rejected (F4); 110 and the hollow-stand option are accepted.
    for (int pwm : {40, 80, 100, 109})
        TEST_ASSERT_FALSE(calibration.start(plant, CAL_PASS_SEEK, pwm, 0));
    TEST_ASSERT_TRUE(plant.output.text.empty());
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, 110, CAL_OPT_KNEE_NUDGE | CAL_OPT_HOLLOW_STAND));
    TEST_ASSERT_EQUAL_UINT(CAL_OPT_ALL, calibration.options());
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
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SWEEP, CAL_PWM_MIN, 0));
    runToEnd(calibration, plant, CAL_PWM_MIN);
    assertFailure(calibration, LegCalibrationFailure::BarelyMoved);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl pot=100 travel=0"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl pot=100 travel=0"));
    TEST_ASSERT_TRUE(plant.output.has(
        "CAL FRONT FAIL FLHL barely moved (0 counts) at PWM 110: check pot wiring or raise speed"));
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
    runUntilStep(calibration, plant, 4);  // hip sweeping, its knee parked
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
// home them, and the table-safe SWEEP saves canonical ranges: a hip's derived
// [0, CAL_HIP_STROKE_EDGES] (it sweeps only to CAL_HIP_SWEEP_EDGES, its knee
// parked), a knee's stroke [-S, 0].
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
    // The SEEK's second knee curl and SWEEP steps 0 and 1 rest at their homes
    // (a press: no running phase, a plain stall; the hip's at its switch-open
    // level), no new HOME.
    TEST_ASSERT_EQUAL_INT(2, plant.output.count("CAL FRONT END FLKL curl hall=0 travel=0"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=0 travel=0"));
    TEST_ASSERT_TRUE(plant.output.hasOpenEnd("CAL FRONT END FLHL curl hall=0 travel=0"));
    TEST_ASSERT_EQUAL_INT(4, plant.output.count("CAL FRONT HOME"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLKL uncurl hall=-1000 travel=-1000", "switch"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=-264 travel=736"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT PARK FLKL hall=-264 target=-264 (FL values)"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL uncurl hall=840 travel=840"));
    TEST_ASSERT_TRUE(plant.output.hasEnd("CAL FRONT END FLHL curl hall=0 travel=-840", "switch"));
    TEST_ASSERT_TRUE(plant.output.has(
        "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_TRUE(plant.output.has(
        "CAL FRONT JOINT FRHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    // The knee's curl back from its park (264 edges at 400 edges/s arms only in
    // its end approach: a stop or the short curl's plain stall).
    TEST_ASSERT_FALSE(plant.output.find("CAL FRONT END FLKL curl hall=0 travel=264").empty());
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT FAIL"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FLKL min=-1000 max=0 edges"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT JOINT FRKL min=-1000 max=0 edges"));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT PASS_DONE SWEEP saved", plant.output.last().c_str());
    TEST_ASSERT_EQUAL_INT(1, plant.saves);
    for (uint8_t joint : {1, 4})
    {
        TEST_ASSERT_EQUAL_INT(0, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, plant.maxStop[joint]);
    }
    for (uint8_t joint : {2, 5})
    {
        TEST_ASSERT_EQUAL_INT(-1000, plant.minStop[joint]);
        TEST_ASSERT_EQUAL_INT(0, plant.maxStop[joint]);
    }
    TEST_ASSERT_EQUAL_UINT(0, calibration.parkedMask());
    TEST_ASSERT_EQUAL_UINT(0, calibration.switchVerifiedMask());
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
// no motion. It fails before any nudge, so no return re-homes it at the
// obstacle and its partner never moves next to it. A knee (and a hip whose
// stalled current happens to sit flat at its switch-open level: only the
// counts tell) fails ShortOfCurlEnd; a hip whose stalled current is raised and
// wobbling (no switch-open signature) fails stuck mid-stroke first.
static void test_curl_without_motion_short_of_a_home_proven_this_boot_fails()
{
    for (int variant = 0; variant < 3; ++variant)
    {
        const uint8_t joint = variant == 0 ? 2 : 1;
        const bool flat = variant == 1;
        FakePlant plant;
        LegCalibration calibration;
        seekHall(calibration, plant);
        const bool hip = joint == 1;
        plant.extension[joint] = hip ? 100 : 900;  // 100 edges uncurled
        plant.stallSign[joint] = hip ? -1 : +1;    // blocked toward its curl end only
        if (flat)
        {
            plant.currentStall[joint] = plant.currentOpen[joint];
            plant.currentWobble[joint] = 0;
        }
        const size_t drivesBefore = plant.drives.size();
        const size_t homesBefore = plant.setHomeJoints.size();
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        const bool stuck = hip && !flat;
        assertFailure(calibration, stuck ? LegCalibrationFailure::StuckMidStroke : LegCalibrationFailure::ShortOfCurlEnd);
        const std::string expected = std::string("CAL FRONT FAIL ") + plant.names[joint] +
            (stuck ? " stuck mid-stroke after 0 edges at PWM 128: raise PWM or check load"
                   : " stopped 100 edges short of its curl end");
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
    TEST_ASSERT_TRUE(plant.output.hasOpenEnd("CAL FRONT END FLHL curl hall=0 travel=0 pot=12"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Nudge), static_cast<int>(phaseAtStop));
    TEST_ASSERT_TRUE(nudgeStop - nudgeStart < CAL_HIP_NUDGE_MAX_MS);
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT SENSOR FLHL hall"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL nudge hall=24 travel=24"));
    // The return, unarmed (24 edges), ends as a plain stall at the switch-open level.
    TEST_ASSERT_TRUE_MESSAGE(plant.output.hasOpenEnd("CAL FRONT END FLHL curl hall=0 travel=-24"),
                             plant.output.text.c_str());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));
    TEST_ASSERT_EQUAL_INT(0, plant.extension[1]);
}

// Slow hip (10 edges/s at the lowest PWM): the nudge counts edges, so it is
// not blind and ends on its 2 s cap with 12 or more edges, which is evidence;
// the return proves the end.
static void test_hip_nudge_by_timeout_with_evidence()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.moveEvery[1] = 10;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MIN, 0));
    uint32_t firstOut = 0, lastOut = 0;
    while (calibration.isRunning() && calibration.moveStage() != LegCalibrationMoveStage::Return)
    {
        tick(calibration, plant, CAL_PWM_MIN);
        if (plant.pwm[1] > 0)
        {
            if (firstOut == 0) firstOut = plant.now;
            lastOut = plant.now;
        }
    }
    const long out = plant.output.valueAfter("CAL FRONT END FLHL nudge", "travel");
    TEST_ASSERT_TRUE(out >= CAL_HALL_DETECT_EDGES && out < CAL_HIP_NUDGE_EDGES);
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT SENSOR FLHL hall"));
    // It counted edges, so the blind stop at 1 s does not apply: on to its 2 s cap.
    TEST_ASSERT_EQUAL_UINT32(CAL_HIP_NUDGE_MAX_MS - CAL_PWM_RAMP_INTERVAL_MS - 10, lastOut - firstOut);
    runUntil(calibration, plant, CAL_PWM_MIN, [&]() { return calibration.currentStep() >= 2; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    TEST_ASSERT_TRUE(calibration.isRunning());
}

// No edges at all (hall A dead): the nudge stops CAL_HIP_NUDGE_BLIND_MS after
// it started (R4: not on to its 2 s cap) without evidence, the return still
// drives the hip back blind, then the pass fails.
static void test_hip_nudge_without_evidence_returns_blind_then_fails()
{
    FakePlant plant;
    plant.useHall();
    plant.extension[1] = 0;
    plant.hallDead[1] = true;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    // Out until the 1 s blind stop (ramp included), then driven back before the FAIL.
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
    // Driven out from the first ramp step until the blind stop, 1 s from nudge start.
    TEST_ASSERT_EQUAL_UINT32(CAL_HIP_NUDGE_BLIND_MS - CAL_PWM_RAMP_INTERVAL_MS - 10, lastOut - firstOut);
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
    // Stuck on the way back: no motion. Its stalled current (raised, wobbling:
    // no switch-open signature) fails it stuck mid-stroke; one that happens to
    // sit flat at the switch-open level passes the detector and fails on its
    // lack of motion back. Either way it is not homed.
    for (bool flat : {false, true})
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
            return calibration.moveStage() == LegCalibrationMoveStage::Return;
        });
        plant.stuck[1] = true;
        if (flat)
        {
            plant.currentStall[1] = plant.currentOpen[1];
            plant.currentWobble[1] = 0;
        }
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        if (flat)
        {
            assertFailure(calibration, LegCalibrationFailure::NoCurlBack);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL curled back only 0 edges after its nudge",
                plant.output.last().c_str());
        }
        else
        {
            assertFailure(calibration, LegCalibrationFailure::StuckMidStroke);
            TEST_ASSERT_EQUAL_STRING(
                "CAL FRONT FAIL FLHL stuck mid-stroke after 0 edges at PWM 128: raise PWM or check load",
                plant.output.last().c_str());
        }
        TEST_ASSERT_TRUE(plant.homeLost[1] == false);  // never homed: nothing to lose
        TEST_ASSERT_FALSE(plant.homed[1]);
        TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT HOME FLHL"));
    }
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
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MIN, CAL_DEFAULT_OPTIONS));
    runToEnd(calibration, plant, CAL_PWM_MIN);
    assertFailure(calibration, LegCalibrationFailure::NoFeedback);
    TEST_ASSERT_EQUAL_STRING(
        "CAL FRONT FAIL FLKL no feedback at PWM 110: check sensor cable, or calibrate at PWM 128",
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
    // The return drives back at twice the nudge's speed, so the rod reaches its
    // curl end and shows its switch-open current before the return's minimum
    // drive time (the nudge's) is up: the detector then ends it as a plain
    // stall and the classification decides.
    mixed.moveEvery[1] = 1;
    runToEnd(other, mixed, CAL_DEFAULT_PWM);
    TEST_ASSERT_TRUE_MESSAGE(mixed.output.has("CAL FRONT SENSOR FLHL pot (was hall)"), mixed.output.text.c_str());
    TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(LegCalibrationFailure::NoFeedback), static_cast<int>(other.failure()),
                                  mixed.output.text.c_str());
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

// Output lines from index `from` on (a later run's lines).
std::vector<std::string> linesFrom(const FakeOutput &output, size_t from)
{
    const std::vector<std::string> all = output.lines();
    return std::vector<std::string>(all.begin() + static_cast<long>(from < all.size() ? from : all.size()), all.end());
}

int countFrom(const FakeOutput &output, size_t from, const std::string &prefix)
{
    int result = 0;
    for (const std::string &line : linesFrom(output, from))
        result += line.compare(0, prefix.size(), prefix) == 0;
    return result;
}

// The table-safe SWEEP of a hall leg (fix round 2026-11-24): knee curl, hip
// curl, knee uncurl to its switch, knee park by count to -264 (PARK line, FL
// values), hip sweep by count to 840 with the knee held parked (driven for its
// 840 edges only, not into a stall), hip curl back to its switch (re-homed
// within 10 edges), knee curl back from its park. The hip saves the derived
// [0, 1069] whatever an earlier SWEEP saved (no stroke check or stroke WARN
// applies to a hall hip); the knee saves its full stroke. The knee is parked
// (parkedMask) from its park until it is driven again, never curled meanwhile.
static void test_hall_table_safe_sweep_parks_the_knee_and_sweeps_by_count()
{
    HallSweep sweep;
    sweep.plant.speed = 1;                // 100 edges/s
    sweep.plant.limitsValid[1] = true;    // an older record's [0, 1000]
    sweep.plant.maxStop[1] = 1000;
    const std::vector<std::string> before = sweep.plant.output.lines();
    size_t mark = 0;  // the SWEEP's PASS line (start() printed its first MOVE)
    while (mark < before.size() && before[mark] != "CAL FRONT PASS SWEEP pwm=128") ++mark;
    TEST_ASSERT_TRUE(mark < before.size());
    int furthest = 0;
    uint32_t firstOut = 0;
    uint32_t lastOut = 0;
    int sweepLoops = 0;
    while (sweep.calibration.isRunning() && sweep.calibration.currentLeg() == 0)
    {
        tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
        const LegCalibration &c = sweep.calibration;
        if (sweep.plant.pwm[1] > 0)
        {
            if (firstOut == 0) firstOut = sweep.plant.now;
            lastOut = sweep.plant.now;
        }
        if (sweep.plant.extension[1] > furthest) furthest = sweep.plant.extension[1];
        if (c.currentLeg() != 0) break;
        TEST_ASSERT_TRUE(c.isTableSafeLeg() || c.currentLegStep() < 2);
        const uint8_t legStep = c.currentLegStep();
        if (legStep == 4 || legStep == 5)
        {
            // Parked from its park until it is driven again, and not curled.
            TEST_ASSERT_EQUAL_UINT(0x04, c.parkedMask());
            TEST_ASSERT_FALSE(c.isJointCurled(2));
        }
        if (legStep <= 3 || sweep.plant.pwm[2] != 0) TEST_ASSERT_EQUAL_UINT(0, c.parkedMask());
        if (legStep == 4)
        {
            ++sweepLoops;
            TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationAction::Sweep), static_cast<int>(c.currentAction()));
            TEST_ASSERT_EQUAL_UINT(4, c.currentStep());
            TEST_ASSERT_EQUAL_UINT(1, c.currentJoint());
        }
        if (legStep == 3)
            TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationAction::Park), static_cast<int>(c.currentAction()));
    }
    TEST_ASSERT_TRUE(sweepLoops > 0);
    const std::vector<std::string> expectedMoves = {
        "CAL FRONT MOVE FLKL curl", "CAL FRONT MOVE FLHL curl", "CAL FRONT MOVE FLKL uncurl",
        "CAL FRONT MOVE FLKL curl park", "CAL FRONT MOVE FLHL uncurl sweep", "CAL FRONT MOVE FLHL curl",
        "CAL FRONT MOVE FLKL curl",
    };
    std::vector<std::string> moves;
    for (const std::string &line : linesFrom(sweep.plant.output, mark))
        if (line.find(" MOVE FL") != std::string::npos) moves.push_back(line);
    TEST_ASSERT_EQUAL_UINT(expectedMoves.size(), moves.size());
    for (size_t i = 0; i < moves.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expectedMoves[i].c_str(), moves[i].c_str());
    // The sweep stopped by count at its target (no coast in this plant).
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, furthest);
    // Driven for the 840 edges only (8.4 s at 100 edges/s), not into a stall.
    TEST_ASSERT_TRUE_MESSAGE(lastOut - firstOut <= 8400, std::to_string(lastOut - firstOut).c_str());
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLKL uncurl hall=-1000 travel=-1000", "switch"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLKL curl hall=-264 travel=736"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT PARK FLKL hall=-264 target=-264 (FL values)"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLHL uncurl hall=840 travel=840"));
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL curl hall=0 travel=-840", "switch"));
    TEST_ASSERT_TRUE(sweep.plant.output.has(
        "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLKL curl hall=0 travel=264", "stop"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1000 max=0 edges"));
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT WARN"));
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT PARK FRKL hall=-264 target=-264 (FL values)"));
    TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.parkedMask());
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.minStop[1]);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, sweep.plant.maxStop[1]);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, sweep.calibration.stagedMax(1));
    TEST_ASSERT_EQUAL_INT(-1000, sweep.plant.minStop[2]);
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.maxStop[2]);
    TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.switchVerifiedMask());
}

// The settled reading of a hip stopped at its sweep target may relax an edge
// or two back toward its curl end: the sweep still counts as reached.
static void test_hall_hip_sweep_target_survives_relaxation()
{
    HallSweep sweep;
    runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
        return sweep.calibration.currentStep() == 4 &&
               sweep.calibration.movePhase() == LegCalibrationMovePhase::Settle;
    });
    TEST_ASSERT_EQUAL_INT(CAL_HIP_SWEEP_EDGES, sweep.plant.extension[1]);
    sweep.plant.countBias[1] += 2;  // the count rises: 2 edges back
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLHL uncurl hall=838 travel=838"));
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT WARN"));
    TEST_ASSERT_TRUE(sweep.plant.output.has(
        "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
}

// R3: a knee curled first while its hip rests extended may stop on the body
// (here 150 edges short of its stop). The SEEK curls it again once its hip is
// curled: it goes deeper, its home moves there with a WARN, and it is proven
// at its real stop.
static void test_seek_second_knee_curl_rehomes_a_knee_stopped_on_the_body()
{
    FakePlant plant;
    plant.useHall();
    plant.high[2] = 850;  // the body, while the hip rests extended
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() == 1; });
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-850"));
    plant.high[2] = 1000;  // the hip curls: the body is out of the knee's way
    runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentLeg() == 1; });
    TEST_ASSERT_TRUE(calibration.isRunning());
    const std::vector<std::string> expected = {
        "CAL FRONT MOVE FLKL curl", "CAL FRONT MOVE FLHL curl", "CAL FRONT MOVE FLKL curl"};
    const std::vector<std::string> moves = moveLines(plant.output);
    TEST_ASSERT_TRUE(moves.size() >= 3);
    for (size_t i = 0; i < 3; ++i) TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), moves[i].c_str());
    TEST_ASSERT_FALSE(plant.output.find("CAL FRONT END FLKL curl hall=150 travel=150").empty());
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT WARN FLKL home moved 150 edges"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLKL count=-1000"));
    TEST_ASSERT_EQUAL_INT(-1000, plant.home[2]);
    TEST_ASSERT_TRUE(calibration.isJointCurled(1));
    TEST_ASSERT_TRUE(calibration.isJointCurled(2));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_EQUAL_INT(1, plant.output.count("CAL FRONT WARN"));
}

// The knee's park is a curl-direction drive that stops at -264. Any detector
// end before that is an obstacle: ParkShort, the knee left out (WARN), and the
// hip never sweeps. The settled park must lie inside [-280, -244]: a coast of
// 20 past the target (or a relaxation of 16 back) is accepted, 21 (or 17) and
// a large coast fail KneeOutOfBand.
static void test_knee_park_short_or_out_of_band_fails()
{
    {
        HallSweep sweep;
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.calibration.currentAction() == LegCalibrationAction::Park;
        });
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[2]);  // at its uncurl switch
        sweep.plant.high[2] = 600;  // something in the way, 400 out
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::ParkShort);
        const std::vector<std::string> lines = sweep.plant.output.lines();
        TEST_ASSERT_EQUAL_STRING("CAL FRONT WARN FLKL left out at -400 edges: jog it to its curl end",
            lines[lines.size() - 2].c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 136 edges short of its park at -264: obstacle?",
            lines.back().c_str());
        TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLKL curl hall=-400 travel=600", "stop"));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT PARK"));
        TEST_ASSERT_FALSE(sweep.plant.output.has("CAL FRONT MOVE FLHL uncurl sweep"));
        TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.parkedMask());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[2]);
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
    const struct
    {
        int springOut;
        int settled;
        bool fails;
    } cases[] = {
        {20, -244, false}, {21, -243, true}, {60, -204, true}, {-16, -280, false}, {-17, -281, true},
    };
    for (const auto &entry : cases)
    {
        HallSweep sweep;
        sweep.plant.springSign[2] = 1;  // the knee coasts (or relaxes) after a curl-direction stop
        sweep.plant.springOut[2] = entry.springOut;
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.calibration.currentStep() > 3 || !sweep.calibration.isRunning();
        });
        const std::string park = "CAL FRONT PARK FLKL hall=" + std::to_string(entry.settled) +
            " target=-264 (FL values)";
        if (!entry.fails)
        {
            TEST_ASSERT_TRUE_MESSAGE(sweep.calibration.isRunning(), sweep.plant.output.last().c_str());
            TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.has(park), park.c_str());
            TEST_ASSERT_EQUAL_UINT(0x04, sweep.calibration.parkedMask());
            continue;
        }
        assertFailure(sweep.calibration, LegCalibrationFailure::KneeOutOfBand);
        const std::string fail = "CAL FRONT FAIL FLKL at " + std::to_string(entry.settled) +
            " edges, outside its park band -280..-244";
        const std::vector<std::string> lines = sweep.plant.output.lines();
        TEST_ASSERT_EQUAL_STRING(fail.c_str(), lines.back().c_str());
        const std::string warn = "CAL FRONT WARN FLKL left out at " + std::to_string(entry.settled) +
            " edges: jog it to its curl end";
        TEST_ASSERT_EQUAL_STRING(warn.c_str(), lines[lines.size() - 2].c_str());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT PARK"));
        TEST_ASSERT_FALSE(sweep.plant.output.has("CAL FRONT MOVE FLHL uncurl sweep"));
        TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.parkedMask());
    }
}

// Any detector end of the sweep before 840 is an obstacle: WARN, the hip
// retracts to its curl switch ("curl retract"), and the pass fails, with no
// retry and the knee left parked. A stop (the current rises, here 600 and 420
// out) or a stall where it never moved (0) fails SweepShort and keeps the
// hip's home for its later curls; an extend switch found before 840 (the
// current drops to the switch-open level, 700 out) fails LostCounts, which
// drops the kept home.
static void test_hall_hip_sweep_obstacle_retracts_then_fails()
{
    for (int reach : {600, 420, 0})
    {
        HallSweep sweep;
        sweep.plant.high[1] = reach;
        sweep.plant.extendSwitch[1] = false;  // an obstacle, not the switch
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() { return sweep.calibration.currentStep() >= 4; });
        const size_t drivesAtSweep = sweep.plant.drives.size();
        sweep.finish();
        const std::string label = "reach " + std::to_string(reach) + "\n" + sweep.plant.output.text;
        assertFailure(sweep.calibration, LegCalibrationFailure::SweepShort);
        const std::vector<std::string> lines = sweep.plant.output.lines();
        const std::string fail = "CAL FRONT FAIL FLHL stopped " + std::to_string(CAL_HIP_SWEEP_EDGES - reach) +
            " edges short of its 840-edge sweep: obstacle? check its home";
        TEST_ASSERT_EQUAL_STRING_MESSAGE(fail.c_str(), lines.back().c_str(), label.c_str());
        const std::string warn = "CAL FRONT WARN FLHL stopped at " + std::to_string(reach) + " edges: retracting";
        TEST_ASSERT_EQUAL_STRING_MESSAGE(warn.c_str(), lines[lines.size() - 4].c_str(), label.c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT MOVE FLHL curl retract", lines[lines.size() - 3].c_str());
        const std::string end = "CAL FRONT END FLHL uncurl hall=" + std::to_string(reach) +
            " travel=" + std::to_string(reach);
        // An obstacle after a running phase is a stop; a hip that never ran
        // shows no switch-open current: stuck. The retract ends at the switch.
        TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.hasEnd(end, reach > 0 ? "stop" : "stuck"), label.c_str());
        const std::string back = "CAL FRONT END FLHL curl hall=0 travel=" + std::to_string(-reach);
        TEST_ASSERT_TRUE_MESSAGE(reach > 0 ? sweep.plant.output.hasEnd(back, "switch")
                                           : sweep.plant.output.hasOpenEnd(back), label.c_str());
        TEST_ASSERT_EQUAL_INT(1, sweep.plant.output.count("CAL FRONT MOVE FLHL uncurl sweep"));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT WARN FLHL left out"));
        TEST_ASSERT_FALSE(drovePartnerAfter(sweep.plant, drivesAtSweep, 2));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[1]);
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_EQUAL_UINT(0x02, sweep.calibration.keptHomeMask() & 0x02);
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
    {
        HallSweep sweep;
        sweep.plant.high[1] = 700;  // an extend switch where the counts do not put it
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::LostCounts);
        const std::vector<std::string> lines = sweep.plant.output.lines();
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL lost counts: switch at 700 edges, expected 1069",
            lines.back().c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT WARN FLHL stopped at 700 edges: retracting", lines[lines.size() - 4].c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT MOVE FLHL curl retract", lines[lines.size() - 3].c_str());
        TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL uncurl hall=700 travel=700", "switch"));
        TEST_ASSERT_EQUAL_INT(1, sweep.plant.output.count("CAL FRONT MOVE FLHL uncurl sweep"));
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.keptHomeMask() & 0x02);
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
}

// The knee must stay inside its park band for the whole sweep (checked every
// loop): a knee drifting out stops the sweep at once, the hip retracts, and the
// pass fails naming the knee.
static void test_knee_leaving_its_band_during_the_sweep_fails()
{
    HallSweep sweep;
    sweep.plant.checkEnvelope = false;  // the knee leaves its band on purpose
    runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
        return sweep.plant.pwm[1] > 0 && sweep.plant.extension[1] >= 300;
    });
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationAction::Sweep),
        static_cast<int>(sweep.calibration.currentAction()));
    sweep.plant.countBias[2] += 30;  // the knee's count drifts 30 out: -294
    tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(LegCalibrationMovePhase::Settle),
        static_cast<int>(sweep.calibration.movePhase()));
    const int stoppedAt = sweep.plant.extension[1];
    TEST_ASSERT_TRUE(stoppedAt >= 300 && stoppedAt <= 308);
    sweep.finish();
    assertFailure(sweep.calibration, LegCalibrationFailure::KneeOutOfBand);
    const std::vector<std::string> lines = sweep.plant.output.lines();
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL at -294 edges, outside its park band -280..-244",
        lines.back().c_str());
    const std::string warn = "CAL FRONT WARN FLHL stopped at " + std::to_string(stoppedAt) + " edges: retracting";
    TEST_ASSERT_EQUAL_STRING(warn.c_str(), lines[lines.size() - 4].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT MOVE FLHL curl retract", lines[lines.size() - 3].c_str());
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[1]);
    TEST_ASSERT_TRUE(sweep.plant.homeLost[2]);
    TEST_ASSERT_EQUAL_STRING("FLKL at -294 edges, outside its park band -280..-244",
        reasonOf(sweep.calibration).c_str());
}

// A retract that cannot get back to its curl switch (the hip frozen 300 out:
// the sweep itself ended on the stop rule, its current raised) fails
// NotRetracted, with the hip left out.
static void test_hip_sweep_retract_that_cannot_get_home_fails()
{
    HallSweep sweep;
    sweep.plant.speed = 1;
    runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
        return sweep.plant.pwm[1] > 0 && sweep.plant.extension[1] >= 300;
    });
    sweep.plant.stuck[1] = true;
    sweep.finish();
    assertFailure(sweep.calibration, LegCalibrationFailure::NotRetracted);
    const std::vector<std::string> lines = sweep.plant.output.lines();
    TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.hasEnd("CAL FRONT END FLHL uncurl hall=300 travel=300", "stop"),
                             sweep.plant.output.text.c_str());
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT WARN FLHL stopped at 300 edges: retracting"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT MOVE FLHL curl retract"));
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL curl hall=300 travel=0", "stuck"));
    TEST_ASSERT_EQUAL_STRING("CAL FRONT WARN FLHL left out at 300 edges: jog it to its curl end",
        lines[lines.size() - 2].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL not back at its curl switch (hall=300): jog it home",
        lines.back().c_str());
    TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
    TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
}

// After its sweep the hip curls back to its curl switch, which must read within
// CAL_HIP_REHOME_TOLERANCE_EDGES (10) of its home either way: 10 short is
// accepted (home kept), 10 deeper re-homes; 11 or 15 off either way is lost
// counts, which drops the hip's home without keeping it (keptHomeMask).
static void test_hip_curl_back_must_rehome_within_its_tolerance()
{
    for (int at : {10, -10, 11, -11, 15})
    {
        HallSweep sweep;
        sweep.runToStep(5);  // hip curling back from 840
        TEST_ASSERT_EQUAL_UINT(1, sweep.calibration.currentJoint());
        sweep.plant.low[1] = at;  // its curl switch now reads `at`
        const size_t homesBefore = sweep.plant.setHomeJoints.size();
        sweep.runToStep(6);
        const std::string label = "switch at " + std::to_string(at);
        if (at >= -CAL_HIP_REHOME_TOLERANCE_EDGES && at <= CAL_HIP_REHOME_TOLERANCE_EDGES)
        {
            TEST_ASSERT_TRUE_MESSAGE(sweep.calibration.isRunning(), label.c_str());
            TEST_ASSERT_TRUE(sweep.calibration.isJointCurled(1));
            TEST_ASSERT_EQUAL_UINT_MESSAGE(homesBefore + (at < 0 ? 1 : 0), sweep.plant.setHomeJoints.size(),
                                           label.c_str());
            TEST_ASSERT_EQUAL_INT(at < 0 ? -at : 0, sweep.plant.home[1]);
            continue;
        }
        assertFailure(sweep.calibration, LegCalibrationFailure::LostCounts);
        const std::string fail = "CAL FRONT FAIL FLHL lost counts: switch at " + std::to_string(at) +
            " edges, expected 0";
        TEST_ASSERT_EQUAL_STRING_MESSAGE(fail.c_str(), sweep.plant.output.last().c_str(), label.c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_EQUAL_UINT_MESSAGE(0, sweep.calibration.keptHomeMask(), label.c_str());
        TEST_ASSERT_EQUAL_UINT(homesBefore, sweep.plant.setHomeJoints.size());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
}

// A pot hip (here a proven pot on a quadrature board) keeps the earlier
// six-step SWEEP: its full sweep to its uncurl end, no park, and its pot range
// saved. Leg B's hall hip takes the table-safe order.
static void test_pot_hip_keeps_the_full_sweep()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[1] = false;
    plant.extendSwitch[1] = false;
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
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL uncurl"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT MOVE FLKL curl park"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT PARK FLKL"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FRKL curl park"));
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FRHL uncurl sweep"));
    TEST_ASSERT_TRUE(plant.output.has(
        "CAL FRONT JOINT FRHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_EQUAL_INT(900, plant.maxStop[1]);
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, plant.maxStop[4]);
}

// A hall hip whose knee is not a hall encoder has no table-safe park: the pass
// fails at SWEEP step 2, before the knee leaves its curl end.
static void test_table_safe_sweep_needs_a_hall_knee()
{
    FakePlant plant;
    plant.useHall();
    plant.hallRod[2] = false;
    plant.kind[2] = JointSensor::Pot;
    plant.low[2] = 100;
    plant.high[2] = 900;
    plant.extension[2] = 500;
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::TableSafeNeedsHall);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL and its knee must both be hall encoders",
        plant.output.last().c_str());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SWEEP, calibration.currentPass());
    TEST_ASSERT_EQUAL_UINT(2, calibration.currentLegStep());
    TEST_ASSERT_EQUAL_STRING("FLHL and its knee must both be hall encoders", reasonOf(calibration).c_str());
    TEST_ASSERT_FALSE(plant.output.has("CAL FRONT MOVE FLKL uncurl"));
    TEST_ASSERT_EQUAL_INT(0, plant.output.count("CAL FRONT MOVE FLHL uncurl"));
    TEST_ASSERT_EQUAL_INT(0, plant.saves);
}

// A knee uncurl must travel the minimum stroke.
static void test_sweep_uncurl_must_travel_the_minimum_stroke()
{
    for (int travel : {CAL_HALL_MIN_STROKE_EDGES - 1, CAL_HALL_MIN_STROKE_EDGES})
    {
        HallSweep sweep;
        sweep.plant.speed = 1;
        sweep.plant.low[2] = 1000 - travel;
        sweep.runToStep(3);
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

// The hip's curl back after its sweep (step 5) and the knee's from its park
// (step 6) must each come back to a home proven this run, and the hip must
// leave its sweep end by the minimum stroke.
static void test_sweep_curl_back_must_leave_the_uncurl_end_with_its_home()
{
    for (bool flat : {false, true})
    {
        // The hip cannot come back. Its stalled current is raised and wobbling
        // (stuck mid-stroke) or, as it happens, flat at the switch-open level
        // (a plain stall with no motion: not at its curl end).
        HallSweep sweep;
        sweep.runToStep(5);
        sweep.plant.stuck[1] = true;
        if (flat)
        {
            sweep.plant.currentStall[1] = sweep.plant.currentOpen[1];
            sweep.plant.currentWobble[1] = 0;
        }
        sweep.finish();
        assertFailure(sweep.calibration, flat ? LegCalibrationFailure::NotAtCurlEnd : LegCalibrationFailure::StuckMidStroke);
        TEST_ASSERT_EQUAL_STRING(flat ? "CAL FRONT FAIL FLHL did not reach its curl end (hall=840)"
                                      : "CAL FRONT FAIL FLHL stuck mid-stroke after 0 edges at PWM 128: raise PWM or check load",
            sweep.plant.output.last().c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_EQUAL_UINT(5, sweep.calibration.currentLegStep());  // the knee stays parked
    }
    {
        HallSweep sweep;
        sweep.runToStep(5);
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
        sweep.runToStep(5);
        sweep.plant.low[1] = CAL_HIP_SWEEP_EDGES - CAL_HALL_MIN_STROKE_EDGES + 1;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::NotAtCurlEnd);
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL did not reach its curl end (hall=391)",
            sweep.plant.output.last().c_str());
    }
}

// The knee's curl back from its park: 40 edges short of its home is accepted,
// 41 fails, before leg B moves.
static void test_sweep_knee_curl_back_short_of_its_home_fails()
{
    for (int end : {960, 959})
    {
        HallSweep sweep;
        sweep.runToStep(6);  // knee curling back from its park
        TEST_ASSERT_EQUAL_UINT(2, sweep.calibration.currentJoint());
        sweep.plant.high[2] = end;
        const size_t drivesBefore = sweep.plant.drives.size();
        sweep.runToStep(7);
        if (end == 959)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::ShortOfCurlEnd);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 41 edges short of its curl end",
                sweep.plant.output.last().c_str());
            TEST_ASSERT_TRUE(sweep.plant.homeLost[2]);
            TEST_ASSERT_FALSE(drovePartnerAfter(sweep.plant, drivesBefore, 5));
            continue;
        }
        TEST_ASSERT_TRUE(sweep.calibration.isRunning());
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1000 max=0 edges"));
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
        sweep.runToStep(3);
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
        sweep.runToStep(7);
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
        sweep.runToStep(7);
        if (entry.fails)
        {
            assertFailure(sweep.calibration, LegCalibrationFailure::StrokeShort);
            TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stroke 849 edges, saved 1000: obstacle?",
                sweep.plant.output.last().c_str());
            TEST_ASSERT_EQUAL_INT(2, sweep.plant.output.count("CAL FRONT MOVE FRKL curl"));  // the SEEK's two
            continue;
        }
        TEST_ASSERT_TRUE(sweep.calibration.isRunning());
        const std::string warn = "CAL FRONT WARN FLKL stroke " + std::to_string(entry.saved) +
                                 " -> " + std::to_string(entry.stroke) + " edges";
        TEST_ASSERT_EQUAL(entry.warns, sweep.plant.output.has(warn));
        TEST_ASSERT_EQUAL_INT(entry.warns ? 1 : 0, sweep.plant.output.count("CAL FRONT WARN"));
    }
}

// A curl back that ends deeper than the home re-homes there: a hip within its
// re-home tolerance (8 deeper) keeps its derived range; a knee's stroke is
// measured to its new home (50 deeper).
static void test_sweep_deeper_curl_back_rehomes_and_recomputes_the_stroke()
{
    HallSweep sweep;
    sweep.runToStep(5);
    sweep.plant.low[1] = -8;
    sweep.runToStep(6);
    TEST_ASSERT_TRUE(sweep.calibration.isRunning());
    TEST_ASSERT_TRUE(sweep.plant.output.hasEnd("CAL FRONT END FLHL curl hall=-8 travel=-848", "switch"));
    TEST_ASSERT_TRUE(sweep.plant.output.has(
        "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT HOME FLHL count=8"));
    TEST_ASSERT_EQUAL_INT(8, sweep.plant.home[1]);
    sweep.plant.high[2] = 1050;  // the knee curls back 50 deeper than its home
    sweep.runToStep(7);
    TEST_ASSERT_FALSE_MESSAGE(sweep.plant.output.find("CAL FRONT END FLKL curl hall=50 travel=314").empty(),
                              sweep.plant.output.text.c_str());
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT JOINT FLKL min=-1050 max=0 edges"));
    TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT HOME FLKL count=-1050"));
    sweep.finish();
    assertCompleted(sweep.calibration);
    TEST_ASSERT_EQUAL_INT(-1050, sweep.calibration.stagedMin(2));
    TEST_ASSERT_EQUAL_INT(0, sweep.calibration.stagedMax(2));
    TEST_ASSERT_EQUAL_INT(CAL_HIP_STROKE_EDGES, sweep.plant.maxStop[1]);
    TEST_ASSERT_EQUAL_INT(-1050, sweep.plant.minStop[2]);
}

// A failed hall joint loses its home (kept as the reference of its later curls:
// keptHomeMask); a joint that did not fail keeps its home. Here the knee cannot
// curl back from its park: 264 edges short.
static void test_fail_clears_only_the_failed_hall_home()
{
    HallSweep sweep;
    sweep.runToStep(6);  // knee curling back from its park
    sweep.plant.stallSign[2] = 1;
    sweep.finish();
    assertFailure(sweep.calibration, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLKL stopped 264 edges short of its curl end",
        sweep.plant.output.last().c_str());
    TEST_ASSERT_FALSE(sweep.plant.homed[2]);
    TEST_ASSERT_TRUE(sweep.plant.homeLost[2]);
    TEST_ASSERT_TRUE(sweep.plant.homed[1]);
    TEST_ASSERT_EQUAL_UINT(1, sweep.plant.clearHomeJoints.size());
    TEST_ASSERT_EQUAL_UINT(2, sweep.plant.clearHomeJoints[0]);
    TEST_ASSERT_EQUAL_UINT(0x04, sweep.calibration.keptHomeMask());
    TEST_ASSERT_EQUAL_INT(-1000, sweep.calibration.keptHome(2));
}

// The hollow-stand option (a build opt-in): the knee parks at -192 and the hip
// sweeps on to its extend switch. Found within 20 edges of 1069 (here 1049 and
// 1080) it is switch-verified and saved where it was found; at 1048 or 1000 the
// hip retracts and the pass fails lost counts; with no switch by 1089 likewise.
static void test_hollow_stand_sweep_finds_the_extend_switch()
{
    const uint8_t options = CAL_OPT_KNEE_NUDGE | CAL_OPT_HOLLOW_STAND;
    for (int at : {1049, 1080})
    {
        HallSweep sweep(options);
        sweep.plant.high[1] = at;
        sweep.plant.high[4] = at;
        int furthest = 0;
        while (sweep.calibration.isRunning())
        {
            tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
            if (sweep.plant.extension[1] > furthest) furthest = sweep.plant.extension[1];
        }
        const std::string label = "switch at " + std::to_string(at) + "\n" + sweep.plant.output.text;
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(LegCalibrationOutcome::Completed),
                                      static_cast<int>(sweep.calibration.outcome()), label.c_str());
        TEST_ASSERT_EQUAL_INT(at, furthest);
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT PARK FLKL hall=-192 target=-192 (FL values)"));
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLKL curl hall=-192 travel=808"));
        const std::string end = "CAL FRONT END FLHL uncurl hall=" + std::to_string(at) + " travel=" + std::to_string(at);
        TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.hasEnd(end, "switch"), label.c_str());
        const std::string joint = "CAL FRONT JOINT FLHL min=0 max=" + std::to_string(at) +
            " edges (extend switch-verified)";
        TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.has(joint), label.c_str());
        TEST_ASSERT_EQUAL_UINT(0x12, sweep.calibration.switchVerifiedMask());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.minStop[1]);
        TEST_ASSERT_EQUAL_INT(at, sweep.plant.maxStop[1]);
        TEST_ASSERT_EQUAL_INT(at, sweep.plant.maxStop[4]);
    }
    for (int at : {1048, 1000})
    {
        HallSweep sweep(options);
        sweep.plant.high[1] = at;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::LostCounts);
        const std::vector<std::string> lines = sweep.plant.output.lines();
        const std::string fail = "CAL FRONT FAIL FLHL lost counts: switch at " + std::to_string(at) +
            " edges, expected 1069";
        TEST_ASSERT_EQUAL_STRING(fail.c_str(), lines.back().c_str());
        const std::string warn = "CAL FRONT WARN FLHL stopped at " + std::to_string(at) + " edges: retracting";
        TEST_ASSERT_EQUAL_STRING(warn.c_str(), lines[lines.size() - 4].c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT MOVE FLHL curl retract", lines[lines.size() - 3].c_str());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[1]);
        TEST_ASSERT_EQUAL_UINT(0, sweep.calibration.switchVerifiedMask());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.saves);
    }
    {
        HallSweep sweep(options);
        sweep.plant.speed = 1;
        sweep.plant.high[1] = 1200;  // no switch where the counts put it
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::LostCounts);
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT END FLHL uncurl hall=1089 travel=1089"));
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT WARN FLHL stopped at 1089 edges: retracting"));
        TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL lost counts: switch at 1089 edges, expected 1069",
            sweep.plant.output.last().c_str());
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[1]);
    }
}

// A run aborted while the hip is out on its sweep, or the knee out on its
// park, says which joint is left out and where; an abort during the hip's
// curl back (a seek) does not.
static void test_abort_during_the_sweep_or_park_warns_the_joint_is_out()
{
    {
        HallSweep sweep;
        sweep.runToStep(4);
        for (int i = 0; i < 50; ++i) tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(sweep.plant.pwm[1] > 0);
        const int at = sweep.plant.position(1);
        TEST_ASSERT_TRUE(at > 0 && at < CAL_HIP_SWEEP_EDGES);
        TEST_ASSERT_TRUE(sweep.calibration.abort(sweep.plant, 'H'));
        const std::vector<std::string> lines = sweep.plant.output.lines();
        const std::string warn = "CAL FRONT WARN FLHL left out at " + std::to_string(at) +
            " edges: jog it to its curl end";
        TEST_ASSERT_EQUAL_STRING(warn.c_str(), lines[lines.size() - 2].c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT ABORTED H", lines.back().c_str());
        TEST_ASSERT_TRUE(sweep.plant.homed[1] && sweep.plant.homed[2]);
    }
    {
        HallSweep sweep;
        sweep.runToStep(3);
        for (int i = 0; i < 50; ++i) tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(sweep.plant.pwm[2] > 0);
        const int at = sweep.plant.position(2);
        TEST_ASSERT_TRUE(sweep.calibration.abort(sweep.plant, 'J'));
        const std::vector<std::string> lines = sweep.plant.output.lines();
        const std::string warn = "CAL FRONT WARN FLKL left out at " + std::to_string(at) +
            " edges: jog it to its curl end";
        TEST_ASSERT_EQUAL_STRING(warn.c_str(), lines[lines.size() - 2].c_str());
        TEST_ASSERT_EQUAL_STRING("CAL FRONT ABORTED J", lines.back().c_str());
    }
    {
        HallSweep sweep;
        sweep.runToStep(5);
        for (int i = 0; i < 50; ++i) tick(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM);
        TEST_ASSERT_TRUE(sweep.plant.pwm[1] < 0);
        TEST_ASSERT_TRUE(sweep.calibration.abort(sweep.plant, 'H'));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.output.count("CAL FRONT WARN"));
    }
}

// ------------------------------------------------- kept home and V1 (no home on a stall)

// V1: a hall hip that is homed and then fails (here stuck mid-stroke) keeps its
// home as the reference of its later curls (keptHomeMask, keptHome), across
// start(). A rerun with the hip resting 878 edges short of it (as FRHL on the
// body, 2026-11-24) whose curl shows no motion fails "878 short" with no HOME
// and no nudge, even when its stalled current sits flat at the switch-open
// level. A rerun with the hip back at its home re-homes it and drops the kept
// home.
static void test_kept_home_judges_the_rerun_curls_of_a_failed_hall_hip()
{
    FakePlant plant;
    LegCalibration calibration;
    seekHall(calibration, plant);
    TEST_ASSERT_EQUAL_UINT(0, calibration.keptHomeMask());
    plant.extension[1] = 300;
    plant.stuck[1] = true;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::StuckMidStroke);
    TEST_ASSERT_FALSE(plant.homed[1]);
    TEST_ASSERT_EQUAL_UINT(0x02, calibration.keptHomeMask());
    TEST_ASSERT_EQUAL_INT(0, calibration.keptHome(1));

    plant.extension[1] = 878;
    plant.currentStall[1] = plant.currentOpen[1];  // flat at the switch-open level
    plant.currentWobble[1] = 0;
    size_t mark = plant.output.lines().size();
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertFailure(calibration, LegCalibrationFailure::ShortOfCurlEnd);
    TEST_ASSERT_EQUAL_STRING("CAL FRONT FAIL FLHL stopped 878 edges short of its curl end",
        plant.output.last().c_str());
    TEST_ASSERT_TRUE(plant.output.hasOpenEnd("CAL FRONT END FLHL curl hall=878 travel=0"));
    TEST_ASSERT_EQUAL_INT(0, countFrom(plant.output, mark, "CAL FRONT HOME FLHL"));
    TEST_ASSERT_EQUAL_INT(0, countFrom(plant.output, mark, "CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_FALSE(plant.homed[1]);
    TEST_ASSERT_EQUAL_UINT(0x02, calibration.keptHomeMask());

    plant.extension[1] = 0;
    plant.stuck[1] = false;
    plant.currentStall[1] = 170;
    plant.currentWobble[1] = 10;
    mark = plant.output.lines().size();
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, CAL_DEFAULT_OPTIONS));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    TEST_ASSERT_EQUAL_INT(1, countFrom(plant.output, mark, "CAL FRONT HOME FLHL count=0"));
    TEST_ASSERT_EQUAL_INT(0, countFrom(plant.output, mark, "CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_TRUE(plant.homed[1]);
    TEST_ASSERT_EQUAL_UINT(0, calibration.keptHomeMask());
}

// V1 without a kept home (a fresh boot): a hip curl whose stall shows no
// switch-open current (raised above the drive's floor and wobbling) is never
// taken as its home, however it got there: a short curl that moved 30 edges
// and stalled, a press of a hip resting mid-stroke (no nudge follows), and a
// nudge's return that came back 14 edges and stalled 40 short of where it
// started (inside the curl-end tolerance). Each fails stuck mid-stroke with no
// HOME line, before the knee moves again.
static void test_hip_stall_without_switch_open_current_is_never_homed()
{
    for (int path = 0; path < 3; ++path)
    {
        FakePlant plant;
        plant.useHall();
        if (path == 0)
        {
            plant.extension[1] = 340;
            plant.moveEvery[1] = 2;  // 50 edges/s: it stalls before it arms
        }
        else if (path == 1)
        {
            plant.extension[1] = 500;
            plant.stallSign[1] = -1;  // resting mid-stroke, blocked toward curl
        }
        else
        {
            plant.extension[1] = 0;  // at its curl switch: press, nudge, return
        }
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
        if (path == 1)
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return calibration.currentJoint() == 1; });
        size_t drivesAtStall = 0;
        if (path == 0)
        {
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() { return plant.pwm[1] < 0 && plant.extension[1] <= 310; });
            plant.stuck[1] = true;
        }
        else if (path == 2)
        {
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
                return calibration.moveStage() == LegCalibrationMoveStage::NudgeOut &&
                       calibration.movePhase() == LegCalibrationMovePhase::Settle;
            });
            plant.extension[1] += 30;  // pushed out further while settling: 54 out
            runUntil(calibration, plant, CAL_DEFAULT_PWM, [&]() {
                return calibration.moveStage() == LegCalibrationMoveStage::Return && plant.pwm[1] < 0 &&
                       plant.extension[1] <= 40;
            });
            plant.stuck[1] = true;
        }
        drivesAtStall = plant.drives.size();
        runToEnd(calibration, plant, CAL_DEFAULT_PWM);
        const std::string label = "path " + std::to_string(path) + "\n" + plant.output.text;
        assertFailure(calibration, LegCalibrationFailure::StuckMidStroke);
        const int edges = path == 0 ? 30 : path == 1 ? 0 : 14;
        const std::string fail = "CAL FRONT FAIL FLHL stuck mid-stroke after " + std::to_string(edges) +
            " edges at PWM 128: raise PWM or check load";
        TEST_ASSERT_EQUAL_STRING_MESSAGE(fail.c_str(), plant.output.last().c_str(), label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, plant.output.count("CAL FRONT HOME FLHL"), label.c_str());
        TEST_ASSERT_EQUAL_INT(path == 2 ? 1 : 0, plant.output.count("CAL FRONT MOVE FLHL nudge"));
        TEST_ASSERT_FALSE(plant.homed[1]);
        TEST_ASSERT_FALSE(calibration.isJointCurled(1));
        TEST_ASSERT_EQUAL_UINT(0, calibration.keptHomeMask());
        TEST_ASSERT_FALSE_MESSAGE(drovePartnerAfter(plant, drivesAtStall, 2), label.c_str());
        std::string end;  // the last hip curl END: the stalled drive's
        for (const std::string &line : plant.output.lines())
            if (line.compare(0, 23, "CAL FRONT END FLHL curl") == 0) end = line;
        TEST_ASSERT_TRUE_MESSAGE(end.find(" by=stuck cur=") != std::string::npos, end.c_str());
    }
}

// R4: a hip nudge never ramps past CAL_HIP_NUDGE_MAX_PWM (C 150 nudges at 128)
// while its curls run at 150; a blind hip nudge (no edge) stops 1 s after it
// started, not at its 2 s cap, and is driven back.
static void test_hip_nudge_is_capped_and_a_blind_one_stops_at_one_second()
{
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;  // parked curled: press, nudge, return
        plant.moveEvery[1] = 3;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MAX, 0));
        int peakOut = 0;
        int peakCurl = 0;
        while (calibration.isRunning() && calibration.currentStep() < 2)
        {
            tick(calibration, plant, CAL_PWM_MAX);
            if (plant.pwm[1] > peakOut) peakOut = plant.pwm[1];
            if (-plant.pwm[1] > peakCurl) peakCurl = -plant.pwm[1];
        }
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT MOVE FLHL nudge"));
        TEST_ASSERT_EQUAL_INT(CAL_HIP_NUDGE_MAX_PWM, peakOut);
        TEST_ASSERT_EQUAL_INT(CAL_PWM_MAX, peakCurl);
        TEST_ASSERT_TRUE(plant.output.has("CAL FRONT HOME FLHL count=0"));
    }
    {
        FakePlant plant;
        plant.useHall();
        plant.extension[1] = 0;
        plant.hallDead[1] = true;
        LegCalibration calibration;
        TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_SEEK, CAL_PWM_MAX, 0));
        uint32_t firstOut = 0, lastOut = 0, firstBack = 0;
        int peakOut = 0;
        while (calibration.isRunning())
        {
            tick(calibration, plant, CAL_PWM_MAX);
            if (plant.pwm[1] > 0)
            {
                if (firstOut == 0) firstOut = plant.now;
                lastOut = plant.now;
                if (plant.pwm[1] > peakOut) peakOut = plant.pwm[1];
            }
            if (plant.pwm[1] < 0 && lastOut != 0 && firstBack == 0) firstBack = plant.now;
        }
        TEST_ASSERT_EQUAL_UINT32(CAL_HIP_NUDGE_BLIND_MS - CAL_PWM_RAMP_INTERVAL_MS - 10, lastOut - firstOut);
        TEST_ASSERT_EQUAL_INT(CAL_HIP_NUDGE_MAX_PWM, peakOut);
        TEST_ASSERT_TRUE(firstBack > lastOut);
        TEST_ASSERT_EQUAL_INT(0, plant.extension[1]);  // driven back to its curl end
        assertFailure(calibration, LegCalibrationFailure::NoFeedback);
    }
}

// END line tokens: a detector end on a drive into a limit switch adds the
// drive's floor (" by=switch cur=<a>/<b> floor=<f>", the current at the switch
// within 1.5 counts of the floor); an extend drive's end does not; a switch-end
// drive's plain stall at the switch-open level adds " open=<current>/<floor>",
// and a switch-end stall without it is stuck (" by=stuck ... floor=<f>").
static void test_end_lines_carry_the_floor_and_the_open_level()
{
    FakePlant plant;
    plant.useHall();
    LegCalibration calibration;
    TEST_ASSERT_TRUE(calibration.start(plant, CAL_PASS_ALL, CAL_DEFAULT_PWM, 0));
    runToEnd(calibration, plant, CAL_DEFAULT_PWM);
    assertCompleted(calibration);
    const std::string hip = plant.output.find("CAL FRONT END FLHL curl hall=0 travel=-500 by=switch cur=");
    TEST_ASSERT_FALSE_MESSAGE(hip.empty(), plant.output.text.c_str());
    long current = 0, reference = 0, floor = 0;
    TEST_ASSERT_EQUAL_INT_MESSAGE(3, std::sscanf(hip.c_str(),
        "CAL FRONT END FLHL curl hall=0 travel=-500 by=switch cur=%ld/%ld floor=%ld", &current, &reference, &floor),
        hip.c_str());
    TEST_ASSERT_EQUAL_INT(75, current);
    TEST_ASSERT_TRUE(reference > current);
    TEST_ASSERT_TRUE_MESSAGE(floor > 0 && current <= floor + 15, hip.c_str());
    const std::string knee = plant.output.find("CAL FRONT END FLKL curl hall=1000 travel=500 by=stop cur=");
    TEST_ASSERT_FALSE(knee.empty());
    TEST_ASSERT_TRUE(knee.find(" floor=") == std::string::npos);
    TEST_ASSERT_TRUE(knee.find(" open=") == std::string::npos);
    // The SWEEP's press of the hip at its switch: the whole drive at the offset.
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLHL curl hall=0 travel=0 open=75/75"));
    // A knee press (an extend drive) prints neither.
    TEST_ASSERT_TRUE(plant.output.has("CAL FRONT END FLKL curl hall=0 travel=0"));

    FakePlant blocked;
    blocked.useHall();
    blocked.stallSign[1] = -1;
    LegCalibration other;
    TEST_ASSERT_TRUE(other.start(blocked, CAL_PASS_SEEK, CAL_DEFAULT_PWM, 0));
    runToEnd(other, blocked, CAL_DEFAULT_PWM);
    assertFailure(other, LegCalibrationFailure::StuckMidStroke);
    // Never homed (and still Unknown: no edges): its hall frame reads the raw
    // count, 500 out, and the line carries its pot pin's reading.
    const std::string stuck = blocked.output.find("CAL FRONT END FLHL curl hall=500 travel=0 pot=12 by=stuck cur=");
    TEST_ASSERT_FALSE_MESSAGE(stuck.empty(), blocked.output.text.c_str());
    TEST_ASSERT_EQUAL_INT_MESSAGE(3, std::sscanf(stuck.c_str(),
        "CAL FRONT END FLHL curl hall=500 travel=0 pot=12 by=stuck cur=%ld/%ld floor=%ld", &current, &reference,
        &floor),
        stuck.c_str());
    TEST_ASSERT_EQUAL_INT(floor, reference);  // an unarmed stall is judged against the floor
    TEST_ASSERT_TRUE_MESSAGE(current > floor + 15, stuck.c_str());
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
// up to the recorded EN drop) and within 0.5 s of it; the probe's
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
    // The switch end now also needs the switch-open signature: CAL_END_OPEN_SAMPLES
    // (6, 300 ms) currents flat at the floor. This model drops the current 150 ms
    // after the count stops, so the call comes up to 500 ms after it.
    TEST_ASSERT_TRUE_MESSAGE(result.stalls.front() > stoppedMs && result.stalls.front() <= stoppedMs + 500,
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
// actuator limit switch (hip curl, knee uncurl) stops at most 350 ms after the
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
    // SEEK: 2 first knee curls; SWEEP: 2 knee curls back from the park.
    TEST_ASSERT_EQUAL_INT(4, stops);
    // SEEK: each leg's second knee curl; SWEEP: steps 0 and 1 of each leg. (The
    // park and the sweep stop by count, at no end.)
    TEST_ASSERT_EQUAL_INT(6, presses);
    // A switch end needs the switch-open signature: CAL_END_OPEN_SAMPLES (6,
    // 300 ms) currents flat at the floor after the switch opens.
    TEST_ASSERT_TRUE_MESSAGE(worstSwitch <= 350, std::to_string(worstSwitch).c_str());
    TEST_ASSERT_TRUE_MESSAGE(worstStop <= 350, std::to_string(worstStop).c_str());
    TEST_ASSERT_TRUE_MESSAGE(worstPress <= 1050, std::to_string(worstPress).c_str());
}

// Stuck mid-stroke: a stall with no end signature after a running phase fails
// the pass before the partner joint moves, the joint's hall home is cleared and
// nothing is saved. A knee uncurl (FRKL at 771 edges, 18 short of its switch as
// on 2026-11-24) and a hip curl. A hip sweep's stuck is an obstacle instead.
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
        // Hip sweep frozen 300 edges out (its current raised: the stop rule),
        // which on the table-safe sweep is an obstacle like any end before its
        // target, not "stuck mid-stroke". The hip retracts (here it comes free)
        // and the pass fails short of its sweep; the knee stays parked.
        HallSweep sweep;
        sweep.plant.speed = 1;
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.plant.pwm[1] > 0 && sweep.plant.extension[1] >= 300;
        });
        sweep.plant.stuck[1] = true;
        const size_t before = sweep.plant.drives.size();
        runUntil(sweep.calibration, sweep.plant, CAL_DEFAULT_PWM, [&]() {
            return sweep.calibration.moveStage() == LegCalibrationMoveStage::Retract;
        });
        sweep.plant.stuck[1] = false;
        sweep.finish();
        assertFailure(sweep.calibration, LegCalibrationFailure::SweepShort);
        TEST_ASSERT_TRUE_MESSAGE(sweep.plant.output.hasEnd("CAL FRONT END FLHL uncurl hall=300 travel=300", "stop"),
                             sweep.plant.output.text.c_str());
        TEST_ASSERT_TRUE(sweep.plant.output.has("CAL FRONT WARN FLHL stopped at 300 edges: retracting"));
        TEST_ASSERT_EQUAL_STRING(
            "CAL FRONT FAIL FLHL stopped 540 edges short of its 840-edge sweep: obstacle? check its home",
            sweep.plant.output.last().c_str());
        TEST_ASSERT_TRUE(sweep.plant.homeLost[1]);
        TEST_ASSERT_FALSE(drovePartnerAfter(sweep.plant, before, 2));
        TEST_ASSERT_EQUAL_INT(0, sweep.plant.extension[1]);
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
        // Per knee: the SEEK curl and its second curl (a press), the SWEEP press
        // (step 0), the park (a curl-direction drive) and the curl back.
        TEST_ASSERT_EQUAL_INT_MESSAGE(10, plant.output.count("CAL FRONT END FLKL curl") +
                                             plant.output.count("CAL FRONT END FRKL curl"), label.c_str());
        for (uint8_t knee : {2, 5})
        {
            const long home = -plant.home[knee];
            TEST_ASSERT_TRUE_MESSAGE(home <= 1000 && home >= 990, label.c_str());
            TEST_ASSERT_TRUE_MESSAGE(plant.minStop[knee] <= -980 && plant.maxStop[knee] == 0, label.c_str());
        }
        // Signature ends: 2 SEEK hip curls, 2 hip curls back from the sweep and 2
        // knee uncurls by switch (the SWEEP's hip presses are plain stalls at the
        // switch-open level), 4 knee curls by stop.
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
    RUN_TEST(test_hall_table_safe_sweep_parks_the_knee_and_sweeps_by_count);
    RUN_TEST(test_hall_hip_sweep_target_survives_relaxation);
    RUN_TEST(test_seek_second_knee_curl_rehomes_a_knee_stopped_on_the_body);
    RUN_TEST(test_knee_park_short_or_out_of_band_fails);
    RUN_TEST(test_hall_hip_sweep_obstacle_retracts_then_fails);
    RUN_TEST(test_knee_leaving_its_band_during_the_sweep_fails);
    RUN_TEST(test_hip_sweep_retract_that_cannot_get_home_fails);
    RUN_TEST(test_hip_curl_back_must_rehome_within_its_tolerance);
    RUN_TEST(test_pot_hip_keeps_the_full_sweep);
    RUN_TEST(test_table_safe_sweep_needs_a_hall_knee);
    RUN_TEST(test_sweep_uncurl_must_travel_the_minimum_stroke);
    RUN_TEST(test_sweep_curl_back_must_leave_the_uncurl_end_with_its_home);
    RUN_TEST(test_sweep_knee_curl_back_short_of_its_home_fails);
    RUN_TEST(test_sweep_stroke_out_of_range);
    RUN_TEST(test_sweep_stroke_against_the_saved_stroke);
    RUN_TEST(test_sweep_deeper_curl_back_rehomes_and_recomputes_the_stroke);
    RUN_TEST(test_fail_clears_only_the_failed_hall_home);
    RUN_TEST(test_hollow_stand_sweep_finds_the_extend_switch);
    RUN_TEST(test_abort_during_the_sweep_or_park_warns_the_joint_is_out);
    RUN_TEST(test_kept_home_judges_the_rerun_curls_of_a_failed_hall_hip);
    RUN_TEST(test_hip_stall_without_switch_open_current_is_never_homed);
    RUN_TEST(test_hip_nudge_is_capped_and_a_blind_one_stops_at_one_second);
    RUN_TEST(test_end_lines_carry_the_floor_and_the_open_level);
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

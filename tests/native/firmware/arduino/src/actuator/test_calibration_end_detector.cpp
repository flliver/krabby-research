#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "unity.h"

#include "src/actuator/calibration_end_detector.h"

#include "robot_end_current_replay.hpp"

// CalibrationEndDetector: every hip and knee jog of the 2026-11-23/24 robot
// logs (the fix round's replay goldens: the FL end test, both mid-stroke hip
// stalls from every later start, every real switch end from every later
// start, the V3 perturbations, the short curls), then synthetic plants for
// each rule (the switch-open signature, switch end, a hip's extend switch,
// compliant mechanical end, stuck joint, rippling stall, short curl, noise),
// the unarmed rules and the sampling. In the synthetic plants the switch-open
// level is the drive's floor: what the IS channel reads in the drive's first
// samples, with the PWM still low.

namespace
{

struct Sample
{
    int e;
    int i;
};

struct Outcome
{
    CalibrationEnd kind = CalibrationEnd::None;
    int k = -1;  // sample index of the decision
    int edges = 0;
};

// Feeds samples (ramp done from rampK on) until the detector decides.
Outcome feed(CalibrationEndDetector &detector, const std::vector<Sample> &samples, int rampK = 0)
{
    Outcome result;
    for (size_t n = 0; n < samples.size(); ++n)
    {
        const CalibrationEnd kind = detector.sample(samples[n].e, static_cast<uint16_t>(samples[n].i),
                                                    static_cast<int>(n) >= rampK);
        if (kind != CalibrationEnd::None)
        {
            result.kind = kind;
            result.k = static_cast<int>(n);
            result.edges = samples[n].e;
            return result;
        }
    }
    return result;
}

CalibrationEndDetector started(bool retract, bool leaving, uint32_t minStillMs = CAL_END_STILL_MIN_DRIVE_MS,
                               bool extendSwitch = false)
{
    CalibrationEndDetector detector;
    detector.begin(0, retract, leaving, minStillMs, extendSwitch);
    return detector;
}

// A seeded LCG: uniform integer in [-amplitude, amplitude].
struct Noise
{
    uint32_t state;
    explicit Noise(uint32_t seed) : state(seed * 2654435761UL + 1U) {}
    int next(int amplitude)
    {
        state = state * 1103515245UL + 12345UL;
        if (amplitude <= 0) return 0;
        return static_cast<int>((state >> 16) % static_cast<uint32_t>(2 * amplitude + 1)) - amplitude;
    }
};

// A run of `count` samples at `perSample` edges each from edges `e`, current `i`.
void run(std::vector<Sample> &out, int &e, int count, int perSample, int i, Noise *noise = nullptr, int amp = 0)
{
    for (int n = 0; n < count; ++n)
    {
        e += perSample;
        out.push_back({e, i + (noise != nullptr ? noise->next(amp) : 0)});
    }
}

const char *kindName(CalibrationEnd kind)
{
    switch (kind)
    {
        case CalibrationEnd::Switch: return "switch";
        case CalibrationEnd::Stop: return "stop";
        case CalibrationEnd::Stuck: return "stuck";
        case CalibrationEnd::Still: return "still";
        default: return "none";
    }
}

void assertKind(CalibrationEnd want, const Outcome &got, const char *what)
{
    std::string message = std::string(what) + ": got " + kindName(got.kind) + " at k=" + std::to_string(got.k);
    TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(want), static_cast<int>(got.kind), message.c_str());
}

}  // namespace

void setUp() {}
void tearDown() {}

static void test_constants_match_the_design()
{
    TEST_ASSERT_EQUAL_INT(128, CAL_DEFAULT_PWM);
    TEST_ASSERT_EQUAL_UINT32(50, CAL_END_SAMPLE_MS);
    TEST_ASSERT_EQUAL_INT(24, CAL_END_RING);
    TEST_ASSERT_EQUAL_UINT32(200, CAL_END_LEARN_DELAY_MS);
    TEST_ASSERT_EQUAL_INT(4, CAL_END_RUN_MIN_EDGES);
    TEST_ASSERT_EQUAL_INT(6, CAL_END_ARM_SAMPLES);
    TEST_ASSERT_EQUAL_INT(75, CAL_END_NORMAL_SPEED_PCT);
    TEST_ASSERT_EQUAL_INT(3, CAL_END_IREF_STEP);
    TEST_ASSERT_EQUAL_INT(3, CAL_END_SWITCH_STILL_SAMPLES);
    TEST_ASSERT_EQUAL_INT(20, CAL_END_SWITCH_REF_SAMPLES);
    TEST_ASSERT_EQUAL_INT(50, CAL_END_ARRIVE_FAST_PCT);
    TEST_ASSERT_EQUAL_INT(88, CAL_END_SWITCH_FAST_PCT);
    TEST_ASSERT_EQUAL_INT(88, CAL_END_SWITCH_SLOW_PCT);
    TEST_ASSERT_EQUAL_INT(90, CAL_END_SWITCH_SLOW_IREF_PCT);
    TEST_ASSERT_EQUAL_INT(12, CAL_END_SHORT_NORMAL_SAMPLES);
    TEST_ASSERT_EQUAL_INT(40, CAL_END_SHORT_RUN_SAMPLES);
    TEST_ASSERT_EQUAL_INT(25, CAL_END_STOP_SPEED_PCT);
    TEST_ASSERT_EQUAL_INT(125, CAL_END_STOP_CURRENT_PCT);
    TEST_ASSERT_EQUAL_INT(3, CAL_END_STOP_HOLD_SAMPLES);
    TEST_ASSERT_EQUAL_UINT32(400, CAL_END_STUCK_STILL_MS);
    TEST_ASSERT_EQUAL_UINT32(500, CAL_END_STILL_WINDOW_MS);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_END_STILL_MIN_DRIVE_MS);
    // The fix round (V1/R1, R2, V3).
    TEST_ASSERT_EQUAL_INT(4, CAL_END_FLOOR_FIRST_SAMPLE);
    TEST_ASSERT_EQUAL_INT(5, CAL_END_FLOOR_LAST_MIN_SAMPLE);
    TEST_ASSERT_EQUAL_INT(10, CAL_END_OPEN_WAIT_SAMPLES);
    TEST_ASSERT_EQUAL_INT(6, CAL_END_OPEN_SAMPLES);
    TEST_ASSERT_EQUAL_UINT16(15, CAL_END_OPEN_MARGIN_TENTHS);
    TEST_ASSERT_EQUAL_UINT16(10, CAL_END_OPEN_FLAT_TENTHS);
    TEST_ASSERT_EQUAL_INT(9, CAL_END_OPEN_SETTLE_SAMPLES);
    TEST_ASSERT_EQUAL_UINT32(600, CAL_END_SWITCH_STUCK_STILL_MS);
    TEST_ASSERT_EQUAL_UINT16(5, CAL_END_FALL_TENTHS);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_END_STUCK_STILL_MAX_MS);
    TEST_ASSERT_EQUAL_INT(2, CAL_END_ARRIVE_LOOKBACK_SAMPLES);
    TEST_ASSERT_EQUAL_INT(5, CAL_END_SWITCH_STOP_STILL_SAMPLES);
    TEST_ASSERT_EQUAL_INT(8, CAL_END_SWITCH_STOP_LONG_SAMPLES);
    TEST_ASSERT_EQUAL_INT(110, CAL_PWM_MIN);
    TEST_ASSERT_EQUAL_INT(8, ACTUATOR_ENC_SWITCH_INSET_EDGES);
    TEST_ASSERT_EQUAL_INT(25, ACTUATOR_ENC_STOP_INSET_EDGES);
}

// The replay of a robot move: its samples, then, when its count had stopped
// before the host stopped it (or `force`), 3 s more at the final count, cycling
// the currents of its stopped tail after the first 3 (the drop), else holding
// the last one. The generator of robot_end_current_replay.hpp pads the same way.
std::vector<RobotEndSample> robotEndReplaySamples(const std::vector<RobotEndSample> &raw, bool pad)
{
    std::vector<RobotEndSample> out = raw;
    if (!pad || raw.empty())
        return out;
    const int16_t end = raw.back().e;
    size_t j = raw.size();
    while (j > 0 && raw[j - 1].e == end)
        --j;
    std::vector<uint16_t> tail;
    for (size_t n = j + 3; n < raw.size(); ++n)
        tail.push_back(raw[n].i);
    if (tail.size() > 8)
        tail.erase(tail.begin(), tail.end() - 8);
    if (tail.empty())
        tail.push_back(raw.back().i);
    for (size_t n = 0; n < 60; ++n)
        out.push_back({end, tail[n % tail.size()], 1});
    return out;
}

std::vector<RobotEndSample> rawSamples(const RobotEndMove &move)
{
    return std::vector<RobotEndSample>(move.samples, move.samples + move.count);
}

// The same approach started `cut` full-PWM samples later (its ramp kept, the
// drive edges shifted): empty when nothing is left after the cut.
std::vector<RobotEndSample> laterStart(const std::vector<RobotEndSample> &raw, size_t cut)
{
    size_t full = 0;
    while (full < raw.size() && raw[full].r == 0)
        ++full;
    if (full + cut >= raw.size())
        return {};
    std::vector<RobotEndSample> out(raw.begin(), raw.begin() + static_cast<long>(full));
    const int base = full > 0 ? raw[full - 1].e : 0;
    const int prev = full + cut > 0 ? raw[full + cut - 1].e : 0;
    for (size_t n = full + cut; n < raw.size(); ++n)
        out.push_back({static_cast<int16_t>(base + raw[n].e - prev), raw[n].i, raw[n].r});
    return out;
}

// Replays samples on a fresh detector set up as the move's drive.
Outcome replaySamples(const RobotEndMove &move, const std::vector<RobotEndSample> &samples,
                      CalibrationEndDetector &detector, int currentOffset = 0)
{
    detector.begin(0, move.retract, !move.curl, CAL_END_STILL_MIN_DRIVE_MS, move.extendSwitch);
    Outcome got;
    for (size_t n = 0; n < samples.size(); ++n)
    {
        const int current = static_cast<int>(samples[n].i) + (samples[n].i > 0 ? currentOffset : 0);
        const CalibrationEnd kind = detector.sample(samples[n].e, static_cast<uint16_t>(current), samples[n].r != 0);
        if (kind != CalibrationEnd::None)
        {
            got.kind = kind;
            got.k = static_cast<int>(n);
            got.edges = samples[n].e;
            break;
        }
    }
    return got;
}

// Replays one robot move to its first decision.
Outcome replay(const RobotEndMove &move, CalibrationEndDetector &detector)
{
    return replaySamples(move, robotEndReplaySamples(rawSamples(move), move.stopped), detector);
}

const RobotEndMove &robotMove(const char *name)
{
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
        if (std::string(kRobotEndMoves[m].name) == name) return kRobotEndMoves[m];
    TEST_FAIL_MESSAGE(name);
    return kRobotEndMoves[0];
}

std::string moveLabel(const RobotEndMove &move, const Outcome &got, const char *what = "")
{
    return std::string(move.name) + what + " " + move.joint + " PWM " + std::to_string(move.pwm) + ": got " +
           kindName(got.kind) + " k=" + std::to_string(got.k) + " e=" + std::to_string(got.edges);
}

// Every hip and knee jog of the 2026-11-23/24 robot logs (231 moves, 26639
// samples): the verdict recorded in the fixture, at the same sample and drive
// edges. A regression net: the end kinds are checked against what the joints
// did in the tests below.
static void test_robot_trace_regression()
{
    int decided[5][5] = {};
    int count[5] = {};
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        const std::string label = moveLabel(move, got);
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(move.expected), static_cast<int>(got.kind), label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(move.k, got.k, label.c_str());
        if (move.k >= 0) TEST_ASSERT_EQUAL_INT_MESSAGE(move.edges, got.edges, label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(std::string(move.joint).substr(2) == "HL" && !move.retract ? 1 : 0,
                                      move.extendSwitch ? 1 : 0, move.name);
        const int group = static_cast<int>(move.group);
        ++count[group];
        ++decided[group][static_cast<int>(got.kind)];
    }
    TEST_ASSERT_EQUAL_INT(231, static_cast<int>(kRobotEndMoveCount));
    // The scripted end-current jogs keep the verdicts the design and the review
    // fix gave them: 10 switches, 11 stops, 6 stuck (the PWM 80 uncurls that
    // stalled short of their switch and the FR knee curl at 80 whose current
    // rose 11%), 4 presses and 9 moves the host stopped while running.
    const int endcur = static_cast<int>(RobotEndGroup::EndCurrent);
    TEST_ASSERT_EQUAL_INT(40, count[endcur]);
    TEST_ASSERT_EQUAL_INT(9, decided[endcur][static_cast<int>(CalibrationEnd::None)]);
    TEST_ASSERT_EQUAL_INT(10, decided[endcur][static_cast<int>(CalibrationEnd::Switch)]);
    TEST_ASSERT_EQUAL_INT(11, decided[endcur][static_cast<int>(CalibrationEnd::Stop)]);
    TEST_ASSERT_EQUAL_INT(6, decided[endcur][static_cast<int>(CalibrationEnd::Stuck)]);
    TEST_ASSERT_EQUAL_INT(4, decided[endcur][static_cast<int>(CalibrationEnd::Still)]);
    const int endTest = static_cast<int>(RobotEndGroup::EndTest);
    TEST_ASSERT_EQUAL_INT(20, count[endTest]);
    TEST_ASSERT_EQUAL_INT(9, decided[endTest][static_cast<int>(CalibrationEnd::Switch)]);
    TEST_ASSERT_EQUAL_INT(3, decided[endTest][static_cast<int>(CalibrationEnd::Stop)]);
    TEST_ASSERT_EQUAL_INT(1, decided[endTest][static_cast<int>(CalibrationEnd::Stuck)]);
    TEST_ASSERT_EQUAL_INT(4, decided[endTest][static_cast<int>(CalibrationEnd::Still)]);
    TEST_ASSERT_EQUAL_INT(3, decided[endTest][static_cast<int>(CalibrationEnd::None)]);
    TEST_ASSERT_EQUAL_INT(48, count[static_cast<int>(RobotEndGroup::Morning)]);
    TEST_ASSERT_EQUAL_INT(48, count[static_cast<int>(RobotEndGroup::Clearance)]);
    TEST_ASSERT_EQUAL_INT(75, count[static_cast<int>(RobotEndGroup::QuickTest)]);
}

// The verdicts of the review-fix reference (the Python detector of the first
// fix stage, firmware/fixround-inputs/2026-11-24/tools/ced-fix) for the 82 robot
// moves the earlier fixture held, independent of this firmware. A move this
// fix round changed names its new verdict and why; every other must agree.
struct ReferenceVerdict
{
    const char *name;
    CalibrationEnd reference;
    CalibrationEnd now;  // the same unless the fix round changed it
    const char *why;
};
static const ReferenceVerdict kReferenceVerdicts[] = {
    {"endcur-111808#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"endcur-111808#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"endcur-111808#2", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-111808#3", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#4", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-111808#5", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#6", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-111808#7", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#8", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-111808#9", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#10", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-111808#11", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#12", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-111808#13", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-111808#14", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-111808#15", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-111808#16", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-111808#17", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-111808#18", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-111808#19", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-112315#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"endcur-112315#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"endcur-112315#2", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#3", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-112315#4", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#5", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-112315#6", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#7", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-112315#8", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"endcur-112315#9", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-112315#10", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-112315#11", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-112315#12", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-112315#13", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"endcur-112315#14", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#15", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-112315#16", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#17", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"endcur-112315#18", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"endcur-112315#19", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"clearance-FL-115032#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FL-115032#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FL-115032#2", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#3", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#4", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#5", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#6", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#7", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-115032#8", CalibrationEnd::Stop, CalibrationEnd::None, "a hip extend into the body at PWM 80: its stop waits for the count to be still 5 samples (R2), after the log ends"},
    {"clearance-FL-115032#9", CalibrationEnd::Still, CalibrationEnd::Stuck, "a hip extend stall at PWM 80 without the switch-open level: an obstacle either way"},
    {"clearance-FL-121754#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FL-121754#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FL-121754#2", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-121754#3", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FL-121754#4", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"clearance-FL-121754#5", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"clearance-FL-121754#6", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"clearance-FL-140513#0", CalibrationEnd::None, CalibrationEnd::Still, "a knee press whose count had stopped: now held 3 s, the plain stall"},
    {"clearance-FL-140513#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FR-115750#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FR-115750#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"clearance-FR-115750#2", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FR-115750#3", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FR-115750#4", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FR-115750#5", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"clearance-FR-115750#6", CalibrationEnd::Stuck, CalibrationEnd::Switch, "R2: FRHL extended into its extend switch, flat at 11"},
    {"clearance-FR-115750#7", CalibrationEnd::Stuck, CalibrationEnd::Stuck, nullptr},
    {"clearance-FR-115750#8", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"fr-recover-120231#0", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"fr-recover-120231#1", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"fr-recover-120231#2", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"hipend-FL-115546#0", CalibrationEnd::Stuck, CalibrationEnd::Switch, "R2: FLHL extended 29 edges into its extend switch"},
    {"hipend-FL-115546#1", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"hipend-FL-115546#2", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"rodscale-hip_back-114914#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"rodscale-hip_back-114914#1", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
    {"rodscale-knee_back_hip_out-114339#0", CalibrationEnd::Stop, CalibrationEnd::Stop, nullptr},
    {"rodscale-knee_back_hip_out-114339#1", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"rodscale-knee_back_hip_out-114339#2", CalibrationEnd::None, CalibrationEnd::None, nullptr},
    {"rodscale-knee_out-114217#0", CalibrationEnd::Still, CalibrationEnd::Still, nullptr},
    {"rodscale-knee_out-114217#1", CalibrationEnd::None, CalibrationEnd::Still, "a hip press at its curl switch whose count had stopped: held 3 s, switch-open"},
    {"rodscale-knee_out-114217#2", CalibrationEnd::Switch, CalibrationEnd::Switch, nullptr},
};

// The earlier reference agrees with this firmware on every move it judged,
// except the six this fix round changed on purpose (R2's extend switches, the
// hip extend obstacles, presses now held 3 s), each named with its reason.
static void test_robot_moves_agree_with_the_earlier_reference()
{
    int changed = 0;
    for (const ReferenceVerdict &ref : kReferenceVerdicts)
    {
        const RobotEndMove &move = robotMove(ref.name);
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        const std::string label = moveLabel(move, got);
        assertKind(ref.now, got, label.c_str());
        if (ref.now != ref.reference)
        {
            TEST_ASSERT_NOT_NULL_MESSAGE(ref.why, label.c_str());
            ++changed;
        }
    }
    TEST_ASSERT_EQUAL_INT(82, static_cast<int>(sizeof(kReferenceVerdicts) / sizeof(kReferenceVerdicts[0])));
    TEST_ASSERT_EQUAL_INT(6, changed);
}

// The FL end test of 2026-11-24 (clearance-FL-173941), the fix round's replay
// goldens: the knee's limit switch (uncurl) twice; its compliant curl stop twice
// and once from a short drive; the hip's curl switch four times, one of them a
// short 109-edge drive; the hip's extend switch three times, as a switch (R2:
// the current rises 25-50% over its last edges, then falls to 7-8); and the hip
// curl at PWM 80 that stalled 229 edges short of its switch, at 841, stuck.
// Every switch decision is at the count where motion stopped.
static void test_fl_end_test_goldens()
{
    struct Golden
    {
        const char *name;
        CalibrationEnd kind;
        int edges;  // drive edges at the decision (-1: any)
    };
    static const Golden goldens[] = {
        {"clearance-FL-173941#2", CalibrationEnd::Switch, 781},   // knee uncurl to its switch
        {"clearance-FL-173941#4", CalibrationEnd::Switch, 778},
        {"clearance-FL-173941#3", CalibrationEnd::Stop, -1},      // knee curl into its stop
        {"clearance-FL-173941#5", CalibrationEnd::Stop, -1},
        {"clearance-FL-173941#17", CalibrationEnd::Stop, -1},     // a short (85-edge) knee curl
        {"clearance-FL-173941#8", CalibrationEnd::Switch, 1070},  // hip curl to its switch
        {"clearance-FL-173941#10", CalibrationEnd::Switch, 1070},
        {"clearance-FL-173941#13", CalibrationEnd::Switch, 841},
        {"clearance-FL-173941#15", CalibrationEnd::Switch, 109},  // a short 109-edge hip curl
        {"clearance-FL-173941#7", CalibrationEnd::Switch, 1070},  // hip extend to its switch
        {"clearance-FL-173941#9", CalibrationEnd::Switch, 1070},
        {"clearance-FL-173941#11", CalibrationEnd::Switch, 1070},
        {"clearance-FL-173941#12", CalibrationEnd::Stuck, -1},    // PWM 80 hip curl stalled at 841
    };
    for (const Golden &golden : goldens)
    {
        const RobotEndMove &move = robotMove(golden.name);
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        const std::string label = moveLabel(move, got);
        assertKind(golden.kind, got, label.c_str());
        if (golden.edges >= 0) TEST_ASSERT_EQUAL_INT_MESSAGE(golden.edges, got.edges, label.c_str());
        if (golden.kind == CalibrationEnd::Switch)
        {
            // At the switch: flat at its floor, the switch-open signature.
            TEST_ASSERT_TRUE_MESSAGE(detector.isSwitchOpen(), label.c_str());
            int deepest = 0;
            for (size_t n = 0; n < move.count; ++n)
                if (move.samples[n].e > deepest) deepest = move.samples[n].e;
            TEST_ASSERT_TRUE_MESSAGE(deepest - got.edges >= 0 && deepest - got.edges <= 1, label.c_str());
        }
    }
    // The hip's extend end on its own is judged as a switch only because its
    // end is one: the same trace replayed with the knee's rules (no
    // extendSwitch) ends as a stop, on the rise before the switch opens.
    const RobotEndMove &extend = robotMove("clearance-FL-173941#7");
    RobotEndMove asKnee = extend;
    asKnee.extendSwitch = false;
    CalibrationEndDetector detector;
    assertKind(CalibrationEnd::Stop, replay(asKnee, detector), "FLHL extend with the stop rule only");
}

// Every later start of a real mid-stroke stall is stuck: the drive started
// `cut` full-PWM samples later in the same approach, then held 3 s at the stall.
// No start point may home a hip there.
void assertEveryLaterStartIsStuck(const char *name, int expectedCuts)
{
    const RobotEndMove &move = robotMove(name);
    const std::vector<RobotEndSample> raw = rawSamples(move);
    int cuts = 0;
    for (size_t cut = 0;; ++cut)
    {
        const std::vector<RobotEndSample> later = laterStart(raw, cut);
        if (later.empty())
            break;
        CalibrationEndDetector detector;
        const Outcome got = replaySamples(move, robotEndReplaySamples(later, true), detector);
        const std::string label = moveLabel(move, got, (" from +" + std::to_string(cut)).c_str());
        assertKind(CalibrationEnd::Stuck, got, label.c_str());
        ++cuts;
    }
    TEST_ASSERT_EQUAL_INT_MESSAGE(expectedCuts, cuts, name);
}

// The FL hip curl at PWM 80 that stalled at 841 (clearance-FL-173941#12): its
// current held 9-14 counts, above the running level and never flat at the
// channel's 7-8. Before the fix every start within 51 edges of the stall ended
// Still and was homed there.
static void test_pwm80_stall_at_841_every_later_start_is_stuck()
{
    assertEveryLaterStartIsStuck("clearance-FL-173941#12", 281);
}

// V1/R1: the FRHL curl at PWM 128 from the body (clearance-FR-115750#7) stalled
// 878 edges short of its switch, its current wobbling 17-31 counts. Before the
// fix only 13 of its 117 later starts were stuck: 20 ended as a switch (iRef
// learned in the slowdown) and 84 as a plain stall, every one homed there.
static void test_frhl_878_stall_every_later_start_is_stuck()
{
    assertEveryLaterStartIsStuck("clearance-FR-115750#7", 117);
    const RobotEndMove &move = robotMove("clearance-FR-115750#7");
    CalibrationEndDetector detector;
    const Outcome got = replay(move, detector);
    assertKind(CalibrationEnd::Stuck, got, "FRHL as recorded");
    TEST_ASSERT_FALSE(detector.isSwitchOpen());
    // Where the review fix's rule would still have switched (k = 104, the third
    // sample at 17), the current is 6 counts above the floor.
    TEST_ASSERT_TRUE(170 > detector.currentFloor() + CAL_END_OPEN_MARGIN_TENTHS);
}

// The same stalls started from rest at the stall point, their ramp under the
// stall's load: the ramp's current climbs with the PWM from the channel's offset
// toward the stall current, so the floor sits higher than on a free ramp. The
// stall current at full PWM is still above it plus the margin, or not flat:
// stuck.
static void test_stalls_started_from_rest_under_load_are_stuck()
{
    struct Stall
    {
        int offset;  // the channel's switch-open level (tenths)
        int stall;   // the stall's median at full PWM (tenths)
        int wobble;  // +- tenths
    };
    static const Stall stalls[] = {{80, 110, 25}, {110, 230, 60}, {80, 110, 0}};
    for (const Stall &stall : stalls)
    {
        for (uint32_t seed = 1; seed <= 20; ++seed)
        {
            Noise noise(seed);
            std::vector<Sample> samples;
            // A 26-step ramp to PWM 128 over about 10 samples: the current
            // climbs from the offset to the stall level.
            for (int k = 0; k < 10; ++k)
                samples.push_back({k < 8 ? 0 : 1, stall.offset + (stall.stall - stall.offset) * k / 10});
            for (int k = 0; k < 60; ++k)
                samples.push_back({1, stall.stall + noise.next(stall.wobble)});
            CalibrationEndDetector detector = started(true, false);
            const Outcome got = feed(detector, samples, 10);
            assertKind(CalibrationEnd::Stuck, got, "stall from rest");
        }
    }
}

// Every later start of every real limit-switch end in the logs (47 ends: hip
// curls, knee uncurls and hip extends, PWM 80-150) homes at the switch, at the
// count where the joint stopped, never stuck (F2, V3): 10,353 replays. Each of
// the 47 as recorded ends Switch. A later start may end Still instead (1,440 of
// them): one cut in so close to the switch that the drive never runs (or arms
// only in its own deceleration) has no running level to see the switch's
// current drop against, so it is homed by the switch-open signature alone,
// which every one of them shows (a stall without it is Stuck).
static void test_every_later_start_of_a_real_switch_end_homes_at_the_switch()
{
    int ends = 0;
    int replays = 0;
    int switches = 0;
    int stills = 0;
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &base = kRobotEndMoves[m];
        if (base.expected != CalibrationEnd::Switch)
            continue;
        ++ends;
        const std::vector<RobotEndSample> raw = rawSamples(base);
        for (size_t cut = 0;; ++cut)
        {
            const std::vector<RobotEndSample> later = laterStart(raw, cut);
            if (later.empty())
                break;
            CalibrationEndDetector detector;
            const Outcome got = replaySamples(base, robotEndReplaySamples(later, true), detector);
            const std::string label = moveLabel(base, got, (" from +" + std::to_string(cut)).c_str());
            if (cut == 0)
                assertKind(CalibrationEnd::Switch, got, label.c_str());
            TEST_ASSERT_TRUE_MESSAGE(got.kind == CalibrationEnd::Switch || got.kind == CalibrationEnd::Still,
                                     label.c_str());
            if (got.kind == CalibrationEnd::Still)
            {
                TEST_ASSERT_TRUE_MESSAGE(detector.isSwitchOpen(), label.c_str());
                ++stills;
            }
            else
            {
                ++switches;
            }
            TEST_ASSERT_EQUAL_INT_MESSAGE(later.back().e, got.edges, label.c_str());
            ++replays;
        }
    }
    TEST_ASSERT_EQUAL_INT(47, ends);
    TEST_ASSERT_EQUAL_INT(10353, replays);
    TEST_ASSERT_EQUAL_INT(8913, switches);
    TEST_ASSERT_EQUAL_INT(1440, stills);
}

// V3: two real limit-switch ends at PWM 120 that sat one sample or one count
// from "stuck mid-stroke". FLKL crawling into its switch (endcur-111808#10)
// passed only because of a one-frame hall blip (789, 790, 789); with it removed
// its current drops to 7 counts 0.41 s after its last edge, and Stuck now waits
// for the drop. FLHL curling into its switch (endcur-111808#17) turned slow, and
// stuck, when its last edge came one sample late; its arrival speed now looks
// back over the samples before that edge. Each at +0, +0.5 and +0.9 count.
static void test_v3_switch_ends_survive_their_perturbations()
{
    const RobotEndMove &knee = robotMove("endcur-111808#10");
    std::vector<RobotEndSample> unblipped = rawSamples(knee);
    int blips = 0;
    for (size_t n = 1; n + 1 < unblipped.size(); ++n)
    {
        if (unblipped[n].e > unblipped[n - 1].e && unblipped[n].e > unblipped[n + 1].e)
        {
            unblipped[n].e = unblipped[n - 1].e;
            ++blips;
        }
    }
    TEST_ASSERT_EQUAL_INT(1, blips);
    const RobotEndMove &hip = robotMove("endcur-111808#17");
    std::vector<RobotEndSample> late = rawSamples(hip);
    int16_t deepest = 0;
    for (const RobotEndSample &sample : late)
        if (sample.e > deepest) deepest = sample.e;
    size_t last = 0;
    while (late[last].e != deepest)
        ++last;
    late[last].e = late[last - 1].e;
    // Every current raised alike (+0, +0.5, +0.9 count), and, which moves the
    // switch-open level against the floor, only the currents after the last
    // edge raised half a count.
    const auto tailRaised = [](std::vector<RobotEndSample> samples) {
        int16_t deepestE = 0;
        for (const RobotEndSample &sample : samples)
            if (sample.e > deepestE) deepestE = sample.e;
        bool after = false;
        for (RobotEndSample &sample : samples)
        {
            if (after) sample.i = static_cast<uint16_t>(sample.i + 5);
            if (sample.e == deepestE) after = true;
        }
        return samples;
    };
    {
        CalibrationEndDetector a;
        assertKind(CalibrationEnd::Switch,
                   replaySamples(knee, robotEndReplaySamples(tailRaised(unblipped), true), a),
                   "FLKL crawl, blip removed, switch-open level +0.5 count");
        CalibrationEndDetector b;
        assertKind(CalibrationEnd::Switch, replaySamples(hip, robotEndReplaySamples(tailRaised(late), true), b),
                   "FLHL curl, last edge late, switch-open level +0.5 count");
    }
    for (int offset : {0, 5, 9})
    {
        CalibrationEndDetector a;
        const Outcome k = replaySamples(knee, robotEndReplaySamples(unblipped, true), a, offset);
        assertKind(CalibrationEnd::Switch, k, "FLKL crawl, blip removed");
        TEST_ASSERT_EQUAL_INT(789, k.edges);
        CalibrationEndDetector b;
        const Outcome h = replaySamples(hip, robotEndReplaySamples(late, true), b, offset);
        assertKind(CalibrationEnd::Switch, h, "FLHL curl, last edge one sample late");
        TEST_ASSERT_EQUAL_INT(deepest, h.edges);
    }
}

// F2: a curl that starts 50-130 edges from its end arms in its own end
// approach. Every real curl end at PWM 120-150 started later in its approach,
// from the middle of the band the design's rule called stuck, ends Still (the
// plain stall of a knee's curl, or a hip curl's switch-open stall) after a
// running phase too short to judge a signature against; none is stuck. The FRHL
// stall started 5 samples later stays stuck.
static void test_short_curls_into_their_end_are_not_stuck()
{
    struct Short
    {
        const char *name;
        size_t cut;
    };
    static const Short shorts[] = {
        {"endcur-111808#5", 12}, {"endcur-111808#7", 6}, {"endcur-111808#9", 9}, {"endcur-111808#11", 152},
        {"endcur-111808#17", 20}, {"endcur-112315#11", 146}, {"endcur-112315#13", 142},
        {"clearance-FL-121754#6", 21}, {"clearance-FR-115750#8", 15}, {"hipend-FL-115546#2", 14},
        {"rodscale-knee_back_hip_out-114339#0", 136},
    };
    for (const Short &entry : shorts)
    {
        const RobotEndMove &move = robotMove(entry.name);
        const std::vector<RobotEndSample> later = laterStart(rawSamples(move), entry.cut);
        CalibrationEndDetector detector;
        const Outcome got = replaySamples(move, robotEndReplaySamples(later, true), detector);
        const std::string label = moveLabel(move, got, (" from +" + std::to_string(entry.cut)).c_str());
        TEST_ASSERT_TRUE_MESSAGE(move.curl, label.c_str());
        assertKind(CalibrationEnd::Still, got, label.c_str());
        TEST_ASSERT_TRUE_MESSAGE(detector.isArmed(), label.c_str());
        TEST_ASSERT_TRUE_MESSAGE(detector.normalSamples() < CAL_END_SHORT_NORMAL_SAMPLES, label.c_str());
        TEST_ASSERT_TRUE_MESSAGE(got.edges >= 40 && got.edges <= 115, label.c_str());
    }
    const RobotEndMove &frhl = robotMove("clearance-FR-115750#7");
    CalibrationEndDetector detector;
    assertKind(CalibrationEnd::Stuck,
               replaySamples(frhl, robotEndReplaySamples(laterStart(rawSamples(frhl), 5), true), detector),
               "FRHL stall from +5");
}

// Every robot move by end kind: no retract decision is a stop; a knee's extend
// decision is never a switch; every switch, and every Still of a drive into a
// limit switch (a retract, a hip extend), shows the switch-open signature (a
// consistency check: the rules require it); every knee curl at PWM 120-150
// whose count stopped before the host stopped it is a stop or the plain stall.
static void test_robot_moves_by_end_kind()
{
    int switchOpenEnds = 0;
    int kneeCurlEnds = 0;
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        const std::string label = moveLabel(move, got);
        const bool switchEnd = move.retract || move.extendSwitch;
        if (move.retract)
            TEST_ASSERT_TRUE_MESSAGE(got.kind != CalibrationEnd::Stop, label.c_str());
        if (!switchEnd)
            TEST_ASSERT_TRUE_MESSAGE(got.kind != CalibrationEnd::Switch, label.c_str());
        if (switchEnd && (got.kind == CalibrationEnd::Switch || got.kind == CalibrationEnd::Still))
        {
            TEST_ASSERT_TRUE_MESSAGE(detector.isSwitchOpen(), label.c_str());
            ++switchOpenEnds;
        }
        const bool knee = std::string(move.joint).substr(2) == "KL";
        if (knee && move.curl && move.pwm >= 120 && move.stopped)
        {
            TEST_ASSERT_TRUE_MESSAGE(got.kind == CalibrationEnd::Stop || got.kind == CalibrationEnd::Still,
                                     label.c_str());
            ++kneeCurlEnds;
        }
    }
    TEST_ASSERT_TRUE(switchOpenEnds > 50);
    TEST_ASSERT_TRUE(kneeCurlEnds >= 10);
}

// Switch-end plant, fast arrival: a ramp at the channel's offset (80, the
// floor), 20 edges per 200 ms at 120 +- 30 tenths, then the count stops dead
// and the current steps back to the switch-open level. Switch at the sixth
// stopped sample: still >= 3, the drop, and six samples flat at the floor.
static void test_switch_fast_arrival_fires_at_six_switch_open_samples()
{
    for (uint32_t seed = 1; seed <= 50; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 12, 0, 80);  // ramp
        run(samples, e, 40, 5, 120, &noise, 30);
        const int stopK = static_cast<int>(samples.size());
        run(samples, e, 20, 0, 80);
        CalibrationEndDetector detector = started(true, false);
        const Outcome got = feed(detector, samples, 11);
        assertKind(CalibrationEnd::Switch, got, "fast switch");
        // The sixth stopped sample, or the fifth when the last running
        // sample's noise put it within a count of the switch-open level.
        TEST_ASSERT_TRUE(got.k >= stopK + CAL_END_OPEN_SAMPLES - 2 && got.k <= stopK + CAL_END_OPEN_SAMPLES - 1);
        TEST_ASSERT_EQUAL_INT(200, got.edges);
        TEST_ASSERT_EQUAL_UINT16(80, detector.endCurrent());
        TEST_ASSERT_EQUAL_UINT16(80, detector.currentFloor());
        TEST_ASSERT_TRUE(detector.isSwitchOpen());
    }
}

static void test_switch_fast_arrival_with_a_transition_sample()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 120);
    const int stopK = static_cast<int>(samples.size());
    samples.push_back({e, 110});
    run(samples, e, 20, 0, 80);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Switch, got, "transition");
    // Six samples at 80 after the transition sample at 110 (not flat with it).
    TEST_ASSERT_EQUAL_INT(stopK + 1 + CAL_END_OPEN_SAMPLES - 1, got.k);
}

// A drop to 92% of the reference is not a switch: the stop ends stuck (400 ms
// without progress and under 3 edges over 1 s).
static void test_switch_needs_the_drop_else_stuck()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 120);
    const int stopK = static_cast<int>(samples.size());
    run(samples, e, 40, 0, 110);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Stuck, got, "92% stop");
    // The last forward edge is at stopK - 1: still >= 8 and 20 samples since.
    TEST_ASSERT_EQUAL_INT(stopK - 1 + 20, got.k);
}

// A knee crawling into its switch: current raised to 140, a forward edge every
// 2-6 samples (100-300 ms), then the switch opens (70). Switch, never stuck
// before it; the same crawl that stops at 115 (82%) without a dip is stuck.
static void test_switch_slow_arrival_crawl()
{
    for (int final = 0; final < 2; ++final)
    {
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 40, 5, 120);
        static const int gaps[] = {2, 3, 4, 6, 5, 2, 6, 3, 6, 4};
        for (int gap : gaps)
        {
            for (int n = 1; n < gap; ++n) samples.push_back({e, 140});
            ++e;
            samples.push_back({e, 140});
        }
        const int lastEdgeK = static_cast<int>(samples.size()) - 1;
        run(samples, e, 40, 0, final == 0 ? 70 : 115);
        CalibrationEndDetector detector = started(true, true);
        const Outcome got = feed(detector, samples);
        if (final == 0)
        {
            assertKind(CalibrationEnd::Switch, got, "crawl into switch");
            TEST_ASSERT_EQUAL_INT(lastEdgeK + CAL_END_OPEN_SAMPLES, got.k);
        }
        else
        {
            assertKind(CalibrationEnd::Stuck, got, "crawl stall without dip");
        }
    }
}

// A one-edge bounce back after the stop neither restarts the stop nor turns the
// fast arrival into a slow one (a slow arrival would also need the current
// below every reference sample).
static void test_switch_bounce_back_keeps_the_stop()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 120);
    const int stopK = static_cast<int>(samples.size());
    samples.push_back({e - 1, 100});
    samples.push_back({e, 100});
    samples.push_back({e - 1, 100});
    run(samples, e, 10, 0, 100);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Switch, got, "bounce");  // 100 <= 88% of 120, fast
    TEST_ASSERT_EQUAL_INT(stopK + CAL_END_OPEN_SAMPLES - 1, got.k);
}

// Compliant mechanical end: the speed tapers from 20 per 200 ms to 0 over 8
// samples while the current ramps 100 -> 150. Stop, at most 6 edges before the
// final creep depth, and within 0.35 s of the joint coming to rest.
static void test_stop_on_a_compliant_mechanical_end()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 100);
    static const int taper[] = {4, 4, 3, 3, 2, 2, 1, 1, 1, 0, 1, 0, 0, 1};
    int cur = 100;
    for (int step : taper)
    {
        e += step;
        cur += cur < 150 ? 7 : 0;
        samples.push_back({e, cur > 150 ? 150 : cur});
    }
    const int restK = static_cast<int>(samples.size()) - 1;
    const int finalE = e;
    run(samples, e, 30, 0, 150);
    CalibrationEndDetector detector = started(false, false);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Stop, got, "compliant stop");
    TEST_ASSERT_TRUE(finalE - got.edges <= 6);
    TEST_ASSERT_TRUE(got.k <= restK + 7);
    TEST_ASSERT_TRUE(detector.endCurrent() * 100U >= 125U * detector.endReference());
}

// FLKL's full curl: a 2 s deceleration with the current rising 100 -> 140.
static void test_stop_after_a_long_deceleration()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 60, 5, 100);
    for (int n = 0; n < 40; ++n)
    {
        e += n < 10 ? 4 : n < 20 ? 3 : n < 30 ? (n % 2) + 1 : (n % 3 == 0 ? 1 : 0);
        samples.push_back({e, 100 + n});
    }
    run(samples, e, 30, 0, 140);
    CalibrationEndDetector detector = started(false, false);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Stop, got, "long deceleration");
}

// A weak rise (+10%, the FR knee curl end at PWM 80) is not a stop: stuck.
static void test_weak_stop_signature_is_stuck()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 100);
    run(samples, e, 40, 0, 110);
    CalibrationEndDetector detector = started(false, false);
    assertKind(CalibrationEnd::Stuck, feed(detector, samples), "weak rise");
}

// The PWM sign picks the rule: a stop signature on a retract drive and a switch
// signature on an extend drive are both stuck.
static void test_rule_follows_the_drive_direction()
{
    std::vector<Sample> rising;
    int e = 0;
    run(rising, e, 40, 5, 100);
    run(rising, e, 40, 0, 170);
    CalibrationEndDetector retract = started(true, false);
    assertKind(CalibrationEnd::Stuck, feed(retract, rising), "stop signature while retracting");

    std::vector<Sample> dropping;
    e = 0;
    run(dropping, e, 40, 5, 120);
    run(dropping, e, 40, 0, 60);
    CalibrationEndDetector extend = started(false, false);
    assertKind(CalibrationEnd::Stuck, feed(extend, dropping), "switch signature while extending");
}

// Stuck-joint plant: a crawl with an edge every 10 samples (0.5 s) and flat
// current is stuck at the first 400 ms gap once net progress is under 3 per s.
static void test_crawl_with_flat_current_is_stuck()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 120);
    for (int n = 0; n < 200; ++n)
    {
        if (n % 10 == 9) ++e;
        samples.push_back({e, 120});
    }
    CalibrationEndDetector detector = started(false, true);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Stuck, got, "crawl");
    TEST_ASSERT_TRUE(got.k < 40 + 40);
}

// A joint frozen mid-run at running current (hall dropout, obstacle without a
// signature) is stuck; a creep of one edge every 7 samples (350 ms) never is.
static void test_frozen_joint_is_stuck_but_a_steady_creep_is_not()
{
    std::vector<Sample> frozen;
    int e = 0;
    run(frozen, e, 40, 5, 120);
    run(frozen, e, 40, 0, 120);
    CalibrationEndDetector a = started(false, false);
    const Outcome got = feed(a, frozen);
    assertKind(CalibrationEnd::Stuck, got, "frozen");
    TEST_ASSERT_EQUAL_INT(39 + 20, got.k);

    for (int retract = 0; retract < 2; ++retract)
    {
        std::vector<Sample> creep;
        e = 0;
        run(creep, e, 40, 5, 120);
        for (int n = 0; n < 2000; ++n)
        {
            if (n % 7 == 6) ++e;
            creep.push_back({e, 120});
        }
        CalibrationEndDetector b = started(retract != 0, false);
        assertKind(CalibrationEnd::None, feed(b, creep), "creep");
    }
}

// Noisy-current plant: +-50 tenths on a running current of 120, constant speed,
// 200 seeds x 400 samples: never a decision.
static void test_noise_while_running_never_decides()
{
    for (uint32_t seed = 1; seed <= 200; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 400, 5, 120, &noise, 50);
        CalibrationEndDetector retract = started(true, true);
        assertKind(CalibrationEnd::None, feed(retract, samples), "noise retract");
        CalibrationEndDetector extend = started(false, false);
        assertKind(CalibrationEnd::None, feed(extend, samples), "noise extend");
    }
}

// Noisy ends (+-30 tenths while the motor conducts; a switch-open level is
// flat, +-4, since the motor no longer does): switch and stop are still found
// within 0.3 s of the last edge.
static void test_noisy_ends_are_found_within_300_ms()
{
    for (uint32_t seed = 1; seed <= 100; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> sw;
        int e = 0;
        run(sw, e, 40, 5, 130, &noise, 30);
        run(sw, e, 30, 0, 70, &noise, 4);
        CalibrationEndDetector retract = started(true, false);
        const Outcome a = feed(retract, sw);
        assertKind(CalibrationEnd::Switch, a, "noisy switch");
        TEST_ASSERT_TRUE(a.k - 39 <= 6);

        std::vector<Sample> stop;
        e = 0;
        run(stop, e, 40, 5, 110, &noise, 30);
        run(stop, e, 30, 0, 190, &noise, 30);
        CalibrationEndDetector extend = started(false, false);
        const Outcome b = feed(extend, stop);
        assertKind(CalibrationEnd::Stop, b, "noisy stop");
        TEST_ASSERT_TRUE(b.k - 39 <= 6);
    }
}

// Hall loss: the count freezes mid-run while the real running current goes on.
static void test_hall_loss_is_stuck()
{
    for (uint32_t seed = 1; seed <= 50; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 30, 5, 120, &noise, 10);
        run(samples, e, 40, 0, 120, &noise, 10);
        CalibrationEndDetector retract = started(true, false);
        assertKind(CalibrationEnd::Stuck, feed(retract, samples), "hall loss retract");
        CalibrationEndDetector extend = started(false, false);
        assertKind(CalibrationEnd::Stuck, feed(extend, samples), "hall loss extend");
    }
}

// A single spike sample (+-60) never fires either rule (all-3 maximum for the
// switch, median of 4 for the stop): with the count still moving nothing ends.
static void test_single_spike_never_fires()
{
    for (int sign = -1; sign <= 1; sign += 2)
    {
        for (int at = 20; at < 40; ++at)
        {
            std::vector<Sample> samples;
            int e = 0;
            run(samples, e, 80, 5, 120);
            samples[static_cast<size_t>(at)].i += 60 * sign;
            CalibrationEndDetector retract = started(true, false);
            assertKind(CalibrationEnd::None, feed(retract, samples), "spike retract");
            CalibrationEndDetector extend = started(false, false);
            assertKind(CalibrationEnd::None, feed(extend, samples), "spike extend");
        }
    }
}

// Unarmed curl: a press with no edges ends Still at k = 20 (1 s); a Return after
// a 1.6 s blind nudge not before k = 32; a creep of 5 edges then still: Still.
static void test_unarmed_curl_still()
{
    std::vector<Sample> press(60, Sample{0, 150});
    CalibrationEndDetector a = started(true, false);
    const Outcome first = feed(a, press, 10);
    assertKind(CalibrationEnd::Still, first, "press");
    TEST_ASSERT_EQUAL_INT(20, first.k);
    TEST_ASSERT_FALSE(a.isArmed());

    CalibrationEndDetector b = started(true, false, 1600);
    const Outcome blind = feed(b, press, 10);
    assertKind(CalibrationEnd::Still, blind, "blind return");
    TEST_ASSERT_EQUAL_INT(32, blind.k);

    // Not before the ramp is done.
    CalibrationEndDetector c = started(false, false);
    const Outcome ramp = feed(c, press, 25);
    TEST_ASSERT_EQUAL_INT(25, ramp.k);

    std::vector<Sample> creep;
    int e = 0;
    run(creep, e, 5, 1, 150);
    run(creep, e, 40, 0, 150);
    CalibrationEndDetector d = started(false, false);
    const Outcome crept = feed(d, creep, 3);
    assertKind(CalibrationEnd::Still, crept, "creep then still");
    TEST_ASSERT_EQUAL_INT(20, crept.k);
    TEST_ASSERT_EQUAL_INT(5, crept.edges);

    // The still window itself bounds a shorter minimum: never before k = 10.
    CalibrationEndDetector early = started(false, false, 0);
    const Outcome windowed = feed(early, press, 0);
    assertKind(CalibrationEnd::Still, windowed, "minimum below the window");
    TEST_ASSERT_EQUAL_INT(10, windowed.k);

    // A minimum drive time beyond the uint16 sample count saturates.
    CalibrationEndDetector e2 = started(false, false, 0xFFFFFFFFUL);
    assertKind(CalibrationEnd::None, feed(e2, press, 0), "never before the minimum");
}

// Unarmed uncurl: no edges ends Still only 2500 ms after the ramp; one that
// breaks free 2.0 s after the ramp arms and runs on.
static void test_unarmed_uncurl_break_free_allowance()
{
    std::vector<Sample> stuck(120, Sample{0, 150});
    CalibrationEndDetector a = started(true, true);
    const Outcome still = feed(a, stuck, 10);
    assertKind(CalibrationEnd::Still, still, "no break free");
    TEST_ASSERT_EQUAL_INT(10 + 50, still.k);

    std::vector<Sample> breaks(50, Sample{0, 150});
    int e = 0;
    run(breaks, e, 200, 5, 120);
    CalibrationEndDetector b = started(true, true);
    assertKind(CalibrationEnd::None, feed(b, breaks, 10), "breaks free");
    TEST_ASSERT_TRUE(b.isArmed());
}

// Five running samples do not arm; samples within 200 ms of the ramp do not
// count; arming takes the median of six currents and six speeds.
static void test_arming()
{
    // Bursts of 8 edges, one per sample, 4 still samples apart: v (edges over
    // 4 samples) is >= 4 on exactly 5 samples in a row of each burst.
    std::vector<Sample> five;
    int e = 0;
    for (int burst = 0; burst < 6; ++burst)
    {
        run(five, e, 8, 1, 120);
        run(five, e, 4, 0, 120);
    }
    CalibrationEndDetector a = started(false, true);
    feed(a, five);
    TEST_ASSERT_FALSE(a.isArmed());

    // Ramp done at sample 3: samples 3-6 are the learn delay; 7-12 arm.
    std::vector<Sample> arm;
    e = 0;
    run(arm, e, 3, 5, 100);
    run(arm, e, 4, 5, 100);
    static const int currents[] = {90, 130, 110, 150, 100, 120};
    static const int steps[] = {6, 4, 5, 7, 5, 3};
    for (int n = 0; n < 6; ++n)
    {
        e += steps[n];
        arm.push_back({e, currents[n]});
    }
    CalibrationEndDetector b = started(false, false);
    for (size_t n = 0; n < arm.size(); ++n)
    {
        TEST_ASSERT_FALSE(b.isArmed());
        b.sample(arm[n].e, static_cast<uint16_t>(arm[n].i), n >= 3);
    }
    TEST_ASSERT_TRUE(b.isArmed());
    TEST_ASSERT_EQUAL_UINT16(120, b.currentReference());  // sorted 90 100 110 120 130 150 -> [3]
    // Speeds over 4 samples at samples 7..12: 21, 20, 20, 22, 21, 20 -> [3] = 21.
    TEST_ASSERT_EQUAL_INT32(21, b.speedReference());

    // Too few edges (under CAL_HALL_CURL_EVIDENCE_EDGES) never arms.
    std::vector<Sample> few;
    e = -200;
    run(few, e, 30, 5, 120);
    CalibrationEndDetector c = started(false, false);
    feed(c, few);
    TEST_ASSERT_FALSE(c.isArmed());
}

// update(): loop periods of 1, 3, 10 and 37 ms give the same decisions on a
// continuous switch-end plant (100 edges/s, current 120 then 75 at t = 3000;
// the floor is the running level here, so 75 sits below it).
static void test_update_sampling_is_independent_of_the_loop_period()
{
    static const uint32_t periods[] = {1, 3, 10, 37};
    for (uint32_t period : periods)
    {
        CalibrationEndDetector detector;
        detector.begin(1000, true, false, CAL_END_STILL_MIN_DRIVE_MS);
        uint32_t decidedAt = 0;
        for (uint32_t t = 1000 + period; t < 9000 && decidedAt == 0; t += period)
        {
            const uint32_t driven = t - 1000;
            const int32_t edges = driven < 3000 ? static_cast<int32_t>(driven / 10) : 300;
            const uint16_t current = driven < 3000 ? 120 : 75;
            if (detector.update(t, edges, current, driven >= 500) == CalibrationEnd::Switch) decidedAt = driven;
        }
        // The count stops at 3000; six 50 ms samples at 75 after it.
        TEST_ASSERT_TRUE(decidedAt >= 3250 && decidedAt <= 3400);
        TEST_ASSERT_EQUAL_UINT16(120, detector.endReference());
    }
}

// The sample is the boxcar mean of every loop's current over its 50 ms; a long
// loop (120 ms) closes one sample and resynchronises, never a burst.
static void test_update_boxcar_mean_and_resync()
{
    CalibrationEndDetector detector;
    detector.begin(0, false, false, CAL_END_STILL_MIN_DRIVE_MS);
    for (uint32_t t = 1; t <= 50; ++t) detector.update(t, 0, t % 2 == 0 ? 200 : 100, true);
    TEST_ASSERT_EQUAL_UINT16(1, detector.samples());
    TEST_ASSERT_EQUAL_UINT16(150, detector.lastSampleCurrent());
    // Rounded: three loops of 100, 100, 101 -> 100.33 -> 100.
    detector.update(60, 0, 100, true);
    detector.update(80, 0, 100, true);
    detector.update(100, 0, 101, true);
    TEST_ASSERT_EQUAL_UINT16(2, detector.samples());
    TEST_ASSERT_EQUAL_UINT16(100, detector.lastSampleCurrent());
    // 120 ms gap: one sample at 220, the next due at 270, not 150 / 200.
    detector.update(220, 7, 90, true);
    TEST_ASSERT_EQUAL_UINT16(3, detector.samples());
    TEST_ASSERT_EQUAL_INT16(7, detector.lastSampleEdges());
    detector.update(260, 7, 90, true);
    TEST_ASSERT_EQUAL_UINT16(3, detector.samples());
    detector.update(270, 7, 90, true);
    TEST_ASSERT_EQUAL_UINT16(4, detector.samples());
    // Wrapping millis: a drive across the 32-bit rollover samples on time.
    detector.begin(0xFFFFFFF0UL, false, false, CAL_END_STILL_MIN_DRIVE_MS);
    detector.update(0xFFFFFFFFUL, 0, 100, true);
    TEST_ASSERT_EQUAL_UINT16(0, detector.samples());
    detector.update(0x22UL, 0, 100, true);
    TEST_ASSERT_EQUAL_UINT16(1, detector.samples());
}

// Saturation: drive edges clamp to int16, the loop count and sample count
// saturate instead of wrapping.
static void test_saturation()
{
    CalibrationEndDetector detector = started(false, false);
    detector.sample(40000, 100, true);
    TEST_ASSERT_EQUAL_INT16(32767, detector.lastSampleEdges());
    detector.sample(-40000, 100, true);
    TEST_ASSERT_EQUAL_INT16(-32768, detector.lastSampleEdges());

    CalibrationEndDetector loops;
    loops.begin(0, false, false, CAL_END_STILL_MIN_DRIVE_MS);
    for (uint32_t n = 0; n < 70000UL; ++n) loops.update(10, 0, 100, true);
    loops.update(50, 0, 100, true);
    TEST_ASSERT_EQUAL_UINT16(100, loops.lastSampleCurrent());

    CalibrationEndDetector many = started(false, false, 0xFFFFFFFFUL);
    for (uint32_t n = 0; n < 70000UL; ++n) many.sample(static_cast<int32_t>(n % 3), 100, true);
    TEST_ASSERT_EQUAL_UINT16(0xFFFF, many.samples());
}

// A last forward edge from before arming (a large bounce back before the run):
// the switch reference falls back to the arming window and the arrival is slow.
static void test_switch_reference_after_an_early_last_edge()
{
    std::vector<Sample> samples;
    samples.push_back({0, 120});
    samples.push_back({400, 120});  // eMax 400 at k = 1: no later sample is progress
    int e = 0;
    run(samples, e, 25, 0, 120);
    run(samples, e, 40, 5, 120);
    run(samples, e, 30, 0, 60);
    CalibrationEndDetector detector = started(true, true);
    const Outcome got = feed(detector, samples);
    assertKind(CalibrationEnd::Switch, got, "early last edge");
    TEST_ASSERT_EQUAL_UINT16(120, detector.endReference());
}

// A drive whose first sample is already its deepest count (no later forward
// edge): the switch stop is measured from the arming sample.
static void test_switch_without_any_forward_edge_uses_the_arming_sample()
{
    std::vector<Sample> samples;
    samples.push_back({300, 120});  // seeds the deepest count
    int e = 0;
    run(samples, e, 40, 5, 120);   // to 200: never past 300
    run(samples, e, 30, 0, 60);
    CalibrationEndDetector detector = started(true, true);
    // Ramp done at 12: armed only once the 1 s stall window has left sample 0.
    const Outcome got = feed(detector, samples, 12);
    assertKind(CalibrationEnd::Switch, got, "no forward edge");
    TEST_ASSERT_TRUE(detector.isArmed());
}

// After a slow arrival the current must also fall below every reference
// sample: one low reference sample (a glitch while crawling) blocks the switch
// at 64% of the median, and the stop ends stuck instead.
static void test_slow_arrival_needs_the_current_below_the_reference_minimum()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 40, 5, 140);
    for (int n = 0; n < 24; ++n)
    {
        if (n % 3 == 2) ++e;
        samples.push_back({e, n == 10 ? 50 : 140});
    }
    run(samples, e, 40, 0, 90);
    CalibrationEndDetector detector = started(true, true);
    assertKind(CalibrationEnd::Stuck, feed(detector, samples), "below the median, not the minimum");
}

// F7: a retract stall whose current ripples +-30% (the FRHL stall at 128): a
// curl ramps at the channel's offset (130), runs at 160, crawls in at 250 +- 50
// (one edge every 2 samples), then stops with its current swinging 170 / 310
// five samples at a time. Its troughs are 68% of the crawl reference and below
// its minimum, as a slow switch arrival would be, but never flat at the floor:
// Stuck, never a switch. The same stall whose current then falls flat to the
// switch-open 130 is a switch.
static void test_retract_stall_with_rippling_current_is_stuck()
{
    for (int switchOpen = 0; switchOpen < 2; ++switchOpen)
        for (uint32_t seed = 1; seed <= 50; ++seed)
        {
            Noise noise(seed);
            std::vector<Sample> samples;
            int e = 0;
            run(samples, e, 10, 0, 130);  // ramp
            run(samples, e, 40, 5, 160, &noise, 10);
            for (int n = 0; n < 20; ++n)
            {
                if (n % 2 == 1) ++e;
                samples.push_back({e, 250 + noise.next(50)});
            }
            for (int n = 0; n < 60; ++n)
                samples.push_back({e, switchOpen != 0 ? 130 : (n / 5) % 2 == 0 ? 170 : 310});
            CalibrationEndDetector detector = started(true, false);
            const Outcome got = feed(detector, samples, 10);
            if (switchOpen != 0)
            {
                assertKind(CalibrationEnd::Switch, got, "switch-open troughs");
                continue;
            }
            assertKind(CalibrationEnd::Stuck, got, "rippling stall");
            TEST_ASSERT_TRUE(detector.normalSamples() >= CAL_END_SHORT_NORMAL_SAMPLES);
        }
}

// F2, synthetic: a knee curl that arms in its end approach (it decelerates
// from 5 to 0 edges per sample while its current rises 100 -> 170) and then
// stalls at 170 +- 15 has no 125% rise over the iRef learned there: Still, the
// plain stall, after the stuck time; the same short drive as an uncurl is
// stuck. After 40 samples running at 160 the same stall (a 6% rise, no
// signature) is stuck.
static void test_short_curl_stall_ends_still_long_one_stuck()
{
    for (int longRun = 0; longRun < 2; ++longRun)
    {
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 10, 1, 60);  // ramp
        if (longRun != 0) run(samples, e, 40, 5, 160);
        static const int steps[] = {5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 0};
        int cur = longRun != 0 ? 160 : 100;
        for (int step : steps)
        {
            e += step;
            cur += cur < 170 ? 10 : 0;
            samples.push_back({e, cur});
        }
        Noise noise(7);
        run(samples, e, 40, 0, 170, &noise, 15);
        CalibrationEndDetector curl = started(false, false);
        const Outcome got = feed(curl, samples, 10);
        TEST_ASSERT_TRUE(curl.isArmed());
        if (longRun != 0)
        {
            assertKind(CalibrationEnd::Stuck, got, "long curl, no signature");
            TEST_ASSERT_TRUE(curl.normalSamples() >= CAL_END_SHORT_NORMAL_SAMPLES);
            continue;
        }
        assertKind(CalibrationEnd::Still, got, "short curl into its end");
        TEST_ASSERT_TRUE(curl.normalSamples() < CAL_END_SHORT_NORMAL_SAMPLES);
        TEST_ASSERT_EQUAL_UINT16(curl.currentReference(), curl.endReference());

        CalibrationEndDetector uncurl = started(false, true);
        assertKind(CalibrationEnd::Stuck, feed(uncurl, samples, 10), "short uncurl");
    }
}

// F2's retract guard: a short hip curl (armed in its end approach) that stops
// flat at the switch-open level, its floor, and not above the running level
// ends Still. One whose current rose above the running level (a motor stalled
// mid-stroke) stays stuck, and so does one that stops below the running level
// but well above its floor (no switch-open signature). A short curl whose last
// edge came 40 samples after arming (a long crawl) is stuck too.
static void test_short_retract_curl_needs_switch_open_and_not_above_running()
{
    struct Case
    {
        int floor;
        int end;
        CalibrationEnd want;
        const char *what;
    };
    static const Case cases[] = {
        {92, 92, CalibrationEnd::Still, "short curl to its switch"},
        {92, 101, CalibrationEnd::Stuck, "short curl, current above running"},
        {60, 95, CalibrationEnd::Stuck, "short curl, no switch-open level"},
    };
    for (const Case &c : cases)
    {
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 10, 1, c.floor);  // ramp
        static const int steps[] = {5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2};
        for (int step : steps)
        {
            e += step;
            samples.push_back({e, 100});
        }
        run(samples, e, 40, 0, c.end);
        CalibrationEndDetector detector = started(true, false);
        const Outcome got = feed(detector, samples, 10);
        TEST_ASSERT_TRUE(detector.isArmed());
        assertKind(c.want, got, c.what);
    }

    std::vector<Sample> crawl;
    int e = 0;
    run(crawl, e, 10, 1, 60);
    run(crawl, e, 10, 5, 100);
    for (int n = 0; n < 50; ++n)
    {
        if (n % 3 == 0) ++e;
        crawl.push_back({e, 100});
    }
    run(crawl, e, 40, 0, 100);
    CalibrationEndDetector detector = started(false, false);
    const Outcome got = feed(detector, crawl, 10);
    assertKind(CalibrationEnd::Stuck, got, "long crawl after arming");
    TEST_ASSERT_TRUE(detector.normalSamples() < CAL_END_SHORT_NORMAL_SAMPLES);
}

// F4: an unarmed curl (slower than arming speed, 3 edges per 200 ms) that has
// made 12 edges and pauses 0.6 s mid-stroke is not at its end: the 1000 ms
// window. Before 12 edges the 500 ms window holds (a press), and a slow curl
// that really stopped ends Still a second after its last edge.
static void test_unarmed_slow_curl_pause_is_not_an_end()
{
    std::vector<Sample> samples;
    int e = 0;
    const auto slow = [&](int count) {
        for (int n = 0; n < count; ++n)
        {
            if (n % 4 != 3) ++e;
            samples.push_back({e, 120});
        }
    };
    slow(40);
    TEST_ASSERT_EQUAL_INT(30, e);
    run(samples, e, 12, 0, 120);  // 0.6 s pause: 0.5 s alone would end it
    slow(20);
    const int lastEdgeK = static_cast<int>(samples.size()) - 2;  // the 20th sample adds none
    run(samples, e, 40, 0, 120);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples, 4);
    TEST_ASSERT_FALSE(detector.isArmed());
    assertKind(CalibrationEnd::Still, got, "slow curl stopped");
    // Fewer than 3 edges over the last 20 samples: the last edge is 18 back.
    TEST_ASSERT_EQUAL_INT(lastEdgeK + 18, got.k);
}

// The floor and the switch-open signature: the floor is the lowest current of
// samples 4 through the one that completes the ramp (at least through 5); it
// reads 65535 before sample 4. switchOpen: the last 6 currents within 10 tenths
// of each other and at most 15 above the floor.
static void test_floor_and_switch_open_signature()
{
    // The IS settles after EN goes high (20, 50, 70, 75), then the ramp climbs.
    CalibrationEndDetector detector = started(true, false);
    static const int rise[] = {20, 50, 70, 75, 80, 90, 100, 110, 120};
    for (int k = 0; k < 9; ++k)
    {
        detector.sample(0, static_cast<uint16_t>(rise[k]), k >= 8);
        if (k < CAL_END_FLOOR_FIRST_SAMPLE)
            TEST_ASSERT_EQUAL_UINT16(0xFFFFU, detector.currentFloor());
    }
    TEST_ASSERT_EQUAL_UINT16(80, detector.currentFloor());
    // After the ramp a lower current no longer moves the floor.
    detector.sample(0, 40, true);
    TEST_ASSERT_EQUAL_UINT16(80, detector.currentFloor());
    // A ramp done at once still takes samples 4 and 5, and only those.
    CalibrationEndDetector early = started(true, false);
    static const int quick[] = {30, 60, 70, 75, 95, 90, 40};
    for (int k = 0; k < 7; ++k) early.sample(0, static_cast<uint16_t>(quick[k]), true);
    TEST_ASSERT_EQUAL_UINT16(90, early.currentFloor());

    struct Window
    {
        int low;
        int high;
        bool open;
    };
    // Floor 80: 95 is its margin; a 10-tenth spread is flat, 11 is not.
    static const Window windows[] = {
        {80, 80, true}, {85, 95, true}, {86, 96, false}, {75, 85, true}, {75, 86, false},
    };
    for (const Window &w : windows)
    {
        CalibrationEndDetector d = started(true, false);
        for (int k = 0; k < 10; ++k) d.sample(0, 80, true);
        for (int k = 0; k < 6; ++k) d.sample(0, static_cast<uint16_t>(k % 2 == 0 ? w.low : w.high), true);
        TEST_ASSERT_EQUAL_UINT16(80, d.currentFloor());
        TEST_ASSERT_EQUAL_INT_MESSAGE(w.open ? 1 : 0, d.isSwitchOpen() ? 1 : 0,
                                      (std::to_string(w.low) + "/" + std::to_string(w.high)).c_str());
    }
}

// V1 at the detector: a drive into a limit switch that never ran (a press, a
// nudge return, a start next to a stall point) ends Still only with the
// switch-open signature, judged once the count has been still 9 samples: a
// press flat at the floor is Still; one flat 30 tenths above it (a motor
// stalled mid-stroke, FL at PWM 80) or wobbling at it is Stuck, reporting its
// current and the floor. A knee curl (extend, no switch) still ends Still on
// the stall alone.
static void test_unarmed_switch_end_stall_needs_switch_open()
{
    struct Press
    {
        int level;
        int wobble;
        CalibrationEnd want;
        int latestK;  // the stall is judged at k = 20; at the floor it may wait 10 more
    };
    static const Press presses[] = {
        {80, 0, CalibrationEnd::Still, 20}, {90, 4, CalibrationEnd::Still, 20},
        {110, 0, CalibrationEnd::Stuck, 20}, {85, 20, CalibrationEnd::Stuck, 20 + CAL_END_OPEN_WAIT_SAMPLES},
    };
    for (const Press &press : presses)
    {
        Noise noise(3);
        std::vector<Sample> samples;
        for (int k = 0; k < 10; ++k) samples.push_back({0, 80});  // ramp at the offset
        for (int k = 0; k < 40; ++k) samples.push_back({0, press.level + noise.next(press.wobble)});
        CalibrationEndDetector detector = started(true, false);
        const Outcome got = feed(detector, samples, 10);
        assertKind(press.want, got, ("press at " + std::to_string(press.level)).c_str());
        TEST_ASSERT_TRUE(got.k >= 20 && got.k <= press.latestK);
        TEST_ASSERT_EQUAL_UINT16(80, detector.endReference());
        TEST_ASSERT_FALSE(detector.isArmed());
        // The same press of a knee curl (extend: its stop has no switch).
        CalibrationEndDetector knee = started(false, false);
        assertKind(CalibrationEnd::Still, feed(knee, samples, 10), "knee press");
    }
    // A short return that reached a stall point 5 samples before the stall
    // window closed: not judged until the count has been still 9 samples.
    std::vector<Sample> late;
    int e = 0;
    for (int k = 0; k < 10; ++k) late.push_back({e, 80});
    run(late, e, 10, 1, 110);
    run(late, e, 40, 0, 110);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, late, 10);
    assertKind(CalibrationEnd::Stuck, got, "short return to a stall point");
    TEST_ASSERT_EQUAL_INT(19 + CAL_END_OPEN_SETTLE_SAMPLES, got.k);  // last edge at k = 19
}

// R2: a hip's extended end is a limit switch. Its current rises 25-50% while
// the hip slows over its last edges (FLHL, 3 of 3 on 2026-11-24), then falls to
// the switch-open level: with extendSwitch that is a switch, where the stop
// rule alone (a knee's) would have called the rise a mechanical stop. An
// obstacle (the current stays high once the count stops) is a stop once the
// count has been still 5 samples; one the hip keeps creeping into after a
// streak of 8.
static void test_hip_extend_switch_before_its_stop_rule()
{
    std::vector<Sample> sw;
    int e = 0;
    run(sw, e, 10, 0, 80);  // ramp at the offset
    run(sw, e, 40, 5, 120);
    static const int tail[] = {2, 2, 1, 1, 1, 0, 1};
    for (int step : tail)
    {
        e += step;
        sw.push_back({e, 160});
    }
    sw.push_back({e, 140});
    sw.push_back({e, 100});
    run(sw, e, 20, 0, 80);
    CalibrationEndDetector hip = started(false, true, CAL_END_STILL_MIN_DRIVE_MS, true);
    const Outcome a = feed(hip, sw, 10);
    assertKind(CalibrationEnd::Switch, a, "hip extend switch");
    TEST_ASSERT_EQUAL_INT(e, a.edges);
    CalibrationEndDetector knee = started(false, true);
    assertKind(CalibrationEnd::Stop, feed(knee, sw, 10), "the same trace with the stop rule only");

    std::vector<Sample> obstacle;
    e = 0;
    run(obstacle, e, 10, 0, 80);
    run(obstacle, e, 40, 5, 120);
    const int stopK = static_cast<int>(obstacle.size());
    run(obstacle, e, 30, 0, 180);
    CalibrationEndDetector blocked = started(false, true, CAL_END_STILL_MIN_DRIVE_MS, true);
    const Outcome b = feed(blocked, obstacle, 10);
    assertKind(CalibrationEnd::Stop, b, "hip extend obstacle");
    TEST_ASSERT_TRUE(b.k >= stopK - 1 + CAL_END_SWITCH_STOP_STILL_SAMPLES);
    TEST_ASSERT_TRUE(b.k <= stopK + 9);

    std::vector<Sample> creep;
    e = 0;
    run(creep, e, 10, 0, 80);
    run(creep, e, 40, 5, 120);
    for (int n = 0; n < 40; ++n)
    {
        if (n % 2 == 1) ++e;
        creep.push_back({e, 180});
    }
    CalibrationEndDetector creeping = started(false, true, CAL_END_STILL_MIN_DRIVE_MS, true);
    const Outcome c = feed(creeping, creep, 10);
    assertKind(CalibrationEnd::Stop, c, "hip creeping into an obstacle");
    TEST_ASSERT_TRUE(c.k <= 50 + 14);
}

// V3: on a drive into a limit switch, Stuck waits (600 ms without an edge, not
// 400) and then waits on while the current is falling or already at the floor,
// up to 1000 ms. A knee crawling into its switch whose current drops only 0.45 s
// after its last edge is a switch; a current stuck at the floor but never flat
// is stuck at 1000 ms; one that stays at the running level is stuck at 600 ms.
static void test_switch_end_stuck_waits_for_a_falling_current()
{
    const auto plant = [](std::vector<Sample> &samples, int &e) {
        run(samples, e, 10, 0, 75);  // ramp at the offset
        run(samples, e, 40, 5, 120);
        for (int n = 0; n < 10; ++n)
        {
            if (n % 3 == 2) ++e;
            samples.push_back({e, 140});  // a crawl at raised current
        }
    };
    std::vector<Sample> drop;
    int e = 0;
    plant(drop, e);
    const int lastEdge = static_cast<int>(drop.size()) - 2;
    for (int n = 0; n < 8; ++n) drop.push_back({e, 140});
    static const int falling[] = {130, 115, 100, 85};
    for (int value : falling) drop.push_back({e, value});
    run(drop, e, 20, 0, 75);
    CalibrationEndDetector a = started(true, true);
    const Outcome switched = feed(a, drop, 10);
    assertKind(CalibrationEnd::Switch, switched, "late drop");
    TEST_ASSERT_TRUE(switched.k > lastEdge + static_cast<int>(CAL_END_SWITCH_STUCK_STILL_MS / CAL_END_SAMPLE_MS));

    std::vector<Sample> ripple;
    e = 0;
    plant(ripple, e);
    const int rippleEdge = static_cast<int>(ripple.size()) - 2;
    for (int n = 0; n < 60; ++n) ripple.push_back({e, n % 2 == 0 ? 75 : 89});
    CalibrationEndDetector b = started(true, true);
    const Outcome stuckLate = feed(b, ripple, 10);
    assertKind(CalibrationEnd::Stuck, stuckLate, "at the floor, never flat");
    TEST_ASSERT_TRUE(stuckLate.k - rippleEdge >= static_cast<int>(CAL_END_STUCK_STILL_MAX_MS / CAL_END_SAMPLE_MS) - 1);

    std::vector<Sample> stall;
    e = 0;
    plant(stall, e);
    const int stallEdge = static_cast<int>(stall.size()) - 2;
    run(stall, e, 60, 0, 140);
    CalibrationEndDetector c = started(true, true);
    const Outcome stuck = feed(c, stall, 10);
    assertKind(CalibrationEnd::Stuck, stuck, "stall at the crawl current");
    // 600 ms still, and under 3 edges in its last 1 s (the crawl's last edges
    // leave that window 14 samples after the last one): no wait at a raised
    // current, so sooner than the stall at the floor.
    TEST_ASSERT_TRUE(stuck.k - stallEdge >= static_cast<int>(CAL_END_SWITCH_STUCK_STILL_MS / CAL_END_SAMPLE_MS));
    TEST_ASSERT_TRUE(stuck.k - stallEdge <= 14);
    TEST_ASSERT_TRUE(stuck.k < stuckLate.k);
}

// V3: the arrival speed is the fastest of the speeds at the last forward edge
// and the two samples before it. A fast hip whose last edge straggles two
// samples late still arrives fast: an 85% drop flat at the floor is a switch. A slow
// arrival would also need the current below every reference sample, and one
// reference sample here dipped to 100 (below the switch-open 102), so judged
// slow it would be stuck.
static void test_arrival_speed_looks_back_two_samples()
{
    std::vector<Sample> samples;
    int e = 0;
    run(samples, e, 10, 0, 102);  // ramp at the offset
    run(samples, e, 40, 5, 120);
    samples[samples.size() - 6].i = 100;  // one low reference sample
    samples.push_back({e, 120});
    samples.push_back({e, 120});
    ++e;
    samples.push_back({e, 115});  // one straggling edge, two samples late
    run(samples, e, 20, 0, 102);
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples, 10);
    assertKind(CalibrationEnd::Switch, got, "straggling last edge");
    TEST_ASSERT_EQUAL_INT(e, got.edges);
}

// The unarmed wait for switchOpen is bounded: a press whose current stays at
// its floor but never flat (80 / 94 around a floor of 80) is judged at k = 20,
// waits 10 more samples, and is stuck at k = 30.
static void test_unarmed_wait_for_switch_open_is_bounded()
{
    std::vector<Sample> samples;
    for (int k = 0; k < 10; ++k) samples.push_back({0, 80});
    for (int k = 0; k < 60; ++k) samples.push_back({0, k % 2 == 0 ? 80 : 94});
    CalibrationEndDetector detector = started(true, false);
    const Outcome got = feed(detector, samples, 10);
    assertKind(CalibrationEnd::Stuck, got, "press at the floor, never flat");
    TEST_ASSERT_EQUAL_INT(20 + CAL_END_OPEN_WAIT_SAMPLES, got.k);
}

// calibrationCurrentTenths: round(10 * avgIS), clamped.
static void test_current_tenths_from_avg_is()
{
    TEST_ASSERT_EQUAL_UINT16(0, calibrationCurrentTenths(0.0f));
    TEST_ASSERT_EQUAL_UINT16(0, calibrationCurrentTenths(-3.0f));
    TEST_ASSERT_EQUAL_UINT16(123, calibrationCurrentTenths(12.34f));
    TEST_ASSERT_EQUAL_UINT16(124, calibrationCurrentTenths(12.36f));
    TEST_ASSERT_EQUAL_UINT16(10230, calibrationCurrentTenths(1023.0f));
    TEST_ASSERT_EQUAL_UINT16(0xFFFF, calibrationCurrentTenths(9000.0f));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_constants_match_the_design);
    RUN_TEST(test_robot_trace_regression);
    RUN_TEST(test_robot_moves_agree_with_the_earlier_reference);
    RUN_TEST(test_fl_end_test_goldens);
    RUN_TEST(test_pwm80_stall_at_841_every_later_start_is_stuck);
    RUN_TEST(test_frhl_878_stall_every_later_start_is_stuck);
    RUN_TEST(test_stalls_started_from_rest_under_load_are_stuck);
    RUN_TEST(test_every_later_start_of_a_real_switch_end_homes_at_the_switch);
    RUN_TEST(test_v3_switch_ends_survive_their_perturbations);
    RUN_TEST(test_short_curls_into_their_end_are_not_stuck);
    RUN_TEST(test_robot_moves_by_end_kind);
    RUN_TEST(test_retract_stall_with_rippling_current_is_stuck);
    RUN_TEST(test_short_curl_stall_ends_still_long_one_stuck);
    RUN_TEST(test_short_retract_curl_needs_switch_open_and_not_above_running);
    RUN_TEST(test_floor_and_switch_open_signature);
    RUN_TEST(test_unarmed_switch_end_stall_needs_switch_open);
    RUN_TEST(test_hip_extend_switch_before_its_stop_rule);
    RUN_TEST(test_switch_end_stuck_waits_for_a_falling_current);
    RUN_TEST(test_arrival_speed_looks_back_two_samples);
    RUN_TEST(test_unarmed_wait_for_switch_open_is_bounded);
    RUN_TEST(test_unarmed_slow_curl_pause_is_not_an_end);
    RUN_TEST(test_switch_fast_arrival_fires_at_six_switch_open_samples);
    RUN_TEST(test_switch_fast_arrival_with_a_transition_sample);
    RUN_TEST(test_switch_needs_the_drop_else_stuck);
    RUN_TEST(test_switch_slow_arrival_crawl);
    RUN_TEST(test_switch_bounce_back_keeps_the_stop);
    RUN_TEST(test_stop_on_a_compliant_mechanical_end);
    RUN_TEST(test_stop_after_a_long_deceleration);
    RUN_TEST(test_weak_stop_signature_is_stuck);
    RUN_TEST(test_rule_follows_the_drive_direction);
    RUN_TEST(test_crawl_with_flat_current_is_stuck);
    RUN_TEST(test_frozen_joint_is_stuck_but_a_steady_creep_is_not);
    RUN_TEST(test_noise_while_running_never_decides);
    RUN_TEST(test_noisy_ends_are_found_within_300_ms);
    RUN_TEST(test_hall_loss_is_stuck);
    RUN_TEST(test_single_spike_never_fires);
    RUN_TEST(test_unarmed_curl_still);
    RUN_TEST(test_unarmed_uncurl_break_free_allowance);
    RUN_TEST(test_arming);
    RUN_TEST(test_update_sampling_is_independent_of_the_loop_period);
    RUN_TEST(test_update_boxcar_mean_and_resync);
    RUN_TEST(test_saturation);
    RUN_TEST(test_switch_reference_after_an_early_last_edge);
    RUN_TEST(test_switch_without_any_forward_edge_uses_the_arming_sample);
    RUN_TEST(test_slow_arrival_needs_the_current_below_the_reference_minimum);
    RUN_TEST(test_current_tenths_from_avg_is);
    return UNITY_END();
}

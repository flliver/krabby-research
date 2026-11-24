#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "unity.h"

#include "src/actuator/calibration_end_detector.h"

#include "robot_end_current_replay.hpp"

// CalibrationEndDetector: parity with the Python reference on the 2026-11-24
// robot traces (the scripted end-current jogs, the PWM 128/150 runs and real
// end approaches started later), then synthetic plants for each rule (switch
// end, compliant mechanical end, stuck joint, rippling stall, short curl,
// noise), the unarmed rules and the sampling.

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

CalibrationEndDetector started(bool retract, bool leaving, uint32_t minStillMs = CAL_END_STILL_MIN_DRIVE_MS)
{
    CalibrationEndDetector detector;
    detector.begin(0, retract, leaving, minStillMs);
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
    TEST_ASSERT_EQUAL_INT(70, CAL_END_SWITCH_SLOW_PCT);
    TEST_ASSERT_EQUAL_INT(90, CAL_END_SWITCH_SLOW_IREF_PCT);
    TEST_ASSERT_EQUAL_INT(12, CAL_END_SHORT_NORMAL_SAMPLES);
    TEST_ASSERT_EQUAL_INT(40, CAL_END_SHORT_RUN_SAMPLES);
    TEST_ASSERT_EQUAL_INT(25, CAL_END_STOP_SPEED_PCT);
    TEST_ASSERT_EQUAL_INT(125, CAL_END_STOP_CURRENT_PCT);
    TEST_ASSERT_EQUAL_INT(3, CAL_END_STOP_HOLD_SAMPLES);
    TEST_ASSERT_EQUAL_UINT32(400, CAL_END_STUCK_STILL_MS);
    TEST_ASSERT_EQUAL_UINT32(500, CAL_END_STILL_WINDOW_MS);
    TEST_ASSERT_EQUAL_UINT32(1000, CAL_END_STILL_MIN_DRIVE_MS);
    TEST_ASSERT_EQUAL_INT(8, ACTUATOR_ENC_SWITCH_INSET_EDGES);
    TEST_ASSERT_EQUAL_INT(25, ACTUATOR_ENC_STOP_INSET_EDGES);
}

// Replays one robot move to its first decision.
Outcome replay(const RobotEndMove &move, CalibrationEndDetector &detector)
{
    detector.begin(0, move.retract, !move.curl, CAL_END_STILL_MIN_DRIVE_MS);
    Outcome got;
    for (size_t n = 0; n < move.count; ++n)
    {
        const RobotEndSample &sample = move.samples[n];
        const CalibrationEnd kind = detector.sample(sample.e, sample.i, sample.r != 0);
        if (kind != CalibrationEnd::None)
        {
            got.kind = kind;
            got.k = static_cast<int>(n);
            got.edges = sample.e;
            break;
        }
    }
    return got;
}

const RobotEndMove &robotMove(const char *name)
{
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
        if (std::string(kRobotEndMoves[m].name) == name) return kRobotEndMoves[m];
    TEST_FAIL_MESSAGE(name);
    return kRobotEndMoves[0];
}

// Every 2026-11-24 robot move (94 moves, 8560 samples): the same verdict at the
// same sample as the Python reference, at the same drive edges. The 40 scripted
// end-current jogs keep the design's verdicts (among them the FRKL curl at PWM
// 80 called stuck, its current rose only about 11%, and nine moves the host
// stopped while running); the 42 jogs of the PWM 128/150 runs and the 12
// shortened approaches follow.
static void test_robot_trace_parity_with_the_python_reference()
{
    int decided[3][5] = {};
    int count[3] = {};
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        std::string label = std::string(move.name) + " " + move.joint + " PWM " + std::to_string(move.pwm) +
                            ": got " + kindName(got.kind) + " k=" + std::to_string(got.k);
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(move.expected), static_cast<int>(got.kind), label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(move.k, got.k, label.c_str());
        if (move.k >= 0) TEST_ASSERT_EQUAL_INT_MESSAGE(move.edges, got.edges, label.c_str());
        const int group = static_cast<int>(move.group);
        ++count[group];
        ++decided[group][static_cast<int>(got.kind)];
    }
    TEST_ASSERT_EQUAL_INT(94, static_cast<int>(kRobotEndMoveCount));
    const int endcur = static_cast<int>(RobotEndGroup::EndCurrent);
    TEST_ASSERT_EQUAL_INT(40, count[endcur]);
    TEST_ASSERT_EQUAL_INT(9, decided[endcur][static_cast<int>(CalibrationEnd::None)]);
    TEST_ASSERT_EQUAL_INT(10, decided[endcur][static_cast<int>(CalibrationEnd::Switch)]);
    TEST_ASSERT_EQUAL_INT(11, decided[endcur][static_cast<int>(CalibrationEnd::Stop)]);
    TEST_ASSERT_EQUAL_INT(6, decided[endcur][static_cast<int>(CalibrationEnd::Stuck)]);
    TEST_ASSERT_EQUAL_INT(4, decided[endcur][static_cast<int>(CalibrationEnd::Still)]);
    const int today = static_cast<int>(RobotEndGroup::Today);
    TEST_ASSERT_EQUAL_INT(42, count[today]);
    TEST_ASSERT_EQUAL_INT(16, decided[today][static_cast<int>(CalibrationEnd::None)]);
    TEST_ASSERT_EQUAL_INT(5, decided[today][static_cast<int>(CalibrationEnd::Switch)]);
    TEST_ASSERT_EQUAL_INT(7, decided[today][static_cast<int>(CalibrationEnd::Stop)]);
    TEST_ASSERT_EQUAL_INT(3, decided[today][static_cast<int>(CalibrationEnd::Stuck)]);
    TEST_ASSERT_EQUAL_INT(11, decided[today][static_cast<int>(CalibrationEnd::Still)]);
    const int shortened = static_cast<int>(RobotEndGroup::Short);
    TEST_ASSERT_EQUAL_INT(12, count[shortened]);
    TEST_ASSERT_EQUAL_INT(11, decided[shortened][static_cast<int>(CalibrationEnd::Still)]);
    TEST_ASSERT_EQUAL_INT(1, decided[shortened][static_cast<int>(CalibrationEnd::Stuck)]);
}

// F1: the FRHL curl at PWM 128 from the body (clearance-FR-115750#7) stalled
// 878 edges short of its switch, its current wobbling 17-31 counts over a
// running level of 16. Three samples at 17 are 68% of the reference and below
// its minimum, but above 90% of iRef: a stalled motor, not a switch-open
// level. It is stuck, not homed there; so is the same stall started later.
static void test_frhl_mid_stroke_stall_at_128_is_stuck_not_a_switch()
{
    const char *names[] = {"clearance-FR-115750#7", "clearance-FR-115750#7 from +5"};
    for (const char *name : names)
    {
        const RobotEndMove &move = robotMove(name);
        TEST_ASSERT_TRUE_MESSAGE(move.retract && move.curl && move.pwm == 128, name);
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        assertKind(CalibrationEnd::Stuck, got, name);
        TEST_ASSERT_TRUE_MESSAGE(detector.isArmed(), name);
        // Its stall current never fell below the running level.
        TEST_ASSERT_TRUE_MESSAGE(detector.endCurrent() > detector.endReference(), name);
    }
    // Where the design's rule switched (k = 104, the third sample at 17), the
    // sample is above 90% of iRef.
    const RobotEndMove &move = robotMove("clearance-FR-115750#7");
    CalibrationEndDetector detector;
    detector.begin(0, true, false, CAL_END_STILL_MIN_DRIVE_MS);
    for (int n = 0; n <= 104; ++n)
    {
        const RobotEndSample &sample = move.samples[n];
        TEST_ASSERT_EQUAL_INT(static_cast<int>(CalibrationEnd::None),
                              static_cast<int>(detector.sample(sample.e, sample.i, sample.r != 0)));
    }
    TEST_ASSERT_EQUAL_UINT16(170, detector.lastSampleCurrent());
    TEST_ASSERT_TRUE(170U * 100U > static_cast<uint32_t>(CAL_END_SWITCH_SLOW_IREF_PCT) * detector.currentReference());
}

// F2: a curl that starts 50-130 edges from its end arms in its own end
// approach. Every real curl end at PWM 120-150 started later in its approach,
// from the middle of the band the design's rule called stuck, ends Still (the
// plain stall) after a running phase too short to judge a signature against;
// none is stuck mid-stroke.
static void test_short_curls_into_their_end_are_not_stuck()
{
    int shortened = 0;
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        if (move.group != RobotEndGroup::Short || std::string(move.name).find("FR-115750#7") != std::string::npos)
            continue;
        ++shortened;
        TEST_ASSERT_TRUE_MESSAGE(move.curl, move.name);
        CalibrationEndDetector detector;
        const Outcome got = replay(move, detector);
        assertKind(CalibrationEnd::Still, got, move.name);
        TEST_ASSERT_TRUE_MESSAGE(detector.isArmed(), move.name);
        TEST_ASSERT_TRUE_MESSAGE(detector.normalSamples() < CAL_END_SHORT_NORMAL_SAMPLES, move.name);
        // Short: 44-112 drive edges.
        TEST_ASSERT_TRUE_MESSAGE(got.edges >= 40 && got.edges <= 115, move.name);
    }
    TEST_ASSERT_EQUAL_INT(11, shortened);
}

// The PWM 128/150 runs by end kind: every hip curl end and knee uncurl end that
// reached its end is a switch, every knee curl end a stop, and the two hip
// uncurls against the body (FRHL from 889, FLHL from 0) stuck.
static void test_pwm_128_runs_by_end_kind()
{
    int switches = 0;
    int stops = 0;
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        if (move.group != RobotEndGroup::Today || move.pwm < 120) continue;
        if (move.retract)
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Stop, move.name);
        else
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Switch, move.name);
        const bool hip = std::string(move.joint).substr(2) == "HL";
        if (move.expected == CalibrationEnd::Switch) ++switches;
        if (move.expected == CalibrationEnd::Stop) ++stops;
        if (move.expected == CalibrationEnd::Stop) TEST_ASSERT_TRUE_MESSAGE(!hip ? move.curl : !move.curl, move.name);
        if (move.expected == CalibrationEnd::Stuck)
            TEST_ASSERT_TRUE_MESSAGE(hip && (!move.curl || std::string(move.name) == "clearance-FR-115750#7"),
                                     move.name);
    }
    TEST_ASSERT_EQUAL_INT(5, switches);
    TEST_ASSERT_EQUAL_INT(6, stops);
}

// The scripted end-current jogs by end kind: every hip curl end and knee
// uncurl end at PWM 120/150 is a switch, recorded exactly where the count
// stopped; every knee curl end at 120/150 is a stop; no decision on a retract
// move is a stop and none on an extend move is a switch.
static void test_robot_traces_by_end_kind()
{
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        if (move.group != RobotEndGroup::EndCurrent) continue;
        if (move.retract)
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Stop, move.name);
        else
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Switch, move.name);
        if (move.pwm < 120 || move.expected == CalibrationEnd::None) continue;
        if (move.retract)
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(CalibrationEnd::Switch), static_cast<int>(move.expected),
                                          move.name);
            // The count at the decision is where motion stopped: the deepest
            // count of the drive, or one edge of bounce back from it.
            int deepest = 0;
            for (int n = 0; n <= move.k; ++n)
                if (move.samples[n].e > deepest) deepest = move.samples[n].e;
            TEST_ASSERT_TRUE_MESSAGE(deepest - move.edges >= 0 && deepest - move.edges <= 1, move.name);
        }
        else
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(CalibrationEnd::Stop), static_cast<int>(move.expected),
                                          move.name);
        }
    }
}

// Switch-end plant, fast arrival: 20 edges per 200 ms at 120 +- 30 tenths, then
// the count stops dead and the current steps to the switch-open level.
static void test_switch_fast_arrival_fires_at_three_still_samples()
{
    for (uint32_t seed = 1; seed <= 50; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 12, 0, 60);  // ramp
        run(samples, e, 40, 5, 120, &noise, 30);
        const int stopK = static_cast<int>(samples.size());
        run(samples, e, 20, 0, 80);
        CalibrationEndDetector detector = started(true, false);
        const Outcome got = feed(detector, samples, 11);
        assertKind(CalibrationEnd::Switch, got, "fast switch");
        TEST_ASSERT_EQUAL_INT(stopK + 2, got.k);  // still = 3 at the third stopped sample
        TEST_ASSERT_EQUAL_INT(200, got.edges);
        TEST_ASSERT_EQUAL_UINT16(80, detector.endCurrent());
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
    TEST_ASSERT_EQUAL_INT(stopK + 3, got.k);  // one sample later: worst of 3 still holds 110 at +2
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
            TEST_ASSERT_EQUAL_INT(lastEdgeK + 3, got.k);
        }
        else
        {
            assertKind(CalibrationEnd::Stuck, got, "crawl stall without dip");
        }
    }
}

// A one-edge bounce back after the stop neither restarts the stop nor turns the
// fast arrival into a slow one (a slow arrival would need <= 70%).
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
    TEST_ASSERT_EQUAL_INT(stopK + 2, got.k);
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

// Noisy ends (+-30 tenths): switch and stop are still found within 0.3 s.
static void test_noisy_ends_are_found_within_300_ms()
{
    for (uint32_t seed = 1; seed <= 100; ++seed)
    {
        Noise noise(seed);
        std::vector<Sample> sw;
        int e = 0;
        run(sw, e, 40, 5, 130, &noise, 30);
        run(sw, e, 30, 0, 70, &noise, 30);
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
// continuous switch-end plant (100 edges/s, current 120 then 75 at t = 3000).
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
        TEST_ASSERT_TRUE(decidedAt >= 3100 && decidedAt <= 3300);
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
// curl runs at 160, crawls in at 250 +- 50 (one edge every 2 samples), then
// stops with its current swinging 170 / 310 five samples at a time. Its
// troughs are 68% of the crawl reference and below its minimum, as a slow
// switch arrival would be, but above 90% of the running level: Stuck, never a
// switch. The same stall with its troughs at a switch-open 130 is a switch.
static void test_retract_stall_with_rippling_current_is_stuck()
{
    for (int switchOpen = 0; switchOpen < 2; ++switchOpen)
        for (uint32_t seed = 1; seed <= 50; ++seed)
        {
            Noise noise(seed);
            std::vector<Sample> samples;
            int e = 0;
            run(samples, e, 40, 5, 160, &noise, 10);
            for (int n = 0; n < 20; ++n)
            {
                if (n % 2 == 1) ++e;
                samples.push_back({e, 250 + noise.next(50)});
            }
            for (int n = 0; n < 60; ++n)
                samples.push_back({e, (n / 5) % 2 == 0 ? (switchOpen != 0 ? 130 : 170) : 310});
            CalibrationEndDetector detector = started(true, false);
            const Outcome got = feed(detector, samples);
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

// F2's retract guard: a short hip curl that stops with its current at or below
// the running level ends Still; one whose current rose above it (a motor
// stalled mid-stroke) stays stuck. A short curl whose last edge came 40
// samples after arming (a long crawl) is stuck too.
static void test_short_retract_curl_needs_the_current_not_above_running()
{
    for (int rise = 0; rise < 2; ++rise)
    {
        std::vector<Sample> samples;
        int e = 0;
        run(samples, e, 10, 1, 60);  // ramp
        static const int steps[] = {5, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2};
        for (int step : steps)
        {
            e += step;
            samples.push_back({e, 100});
        }
        run(samples, e, 40, 0, rise != 0 ? 101 : 95);
        CalibrationEndDetector detector = started(true, false);
        const Outcome got = feed(detector, samples, 10);
        TEST_ASSERT_TRUE(detector.isArmed());
        assertKind(rise != 0 ? CalibrationEnd::Stuck : CalibrationEnd::Still, got, "short retract curl");
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
    RUN_TEST(test_robot_trace_parity_with_the_python_reference);
    RUN_TEST(test_robot_traces_by_end_kind);
    RUN_TEST(test_frhl_mid_stroke_stall_at_128_is_stuck_not_a_switch);
    RUN_TEST(test_short_curls_into_their_end_are_not_stuck);
    RUN_TEST(test_pwm_128_runs_by_end_kind);
    RUN_TEST(test_retract_stall_with_rippling_current_is_stuck);
    RUN_TEST(test_short_curl_stall_ends_still_long_one_stuck);
    RUN_TEST(test_short_retract_curl_needs_the_current_not_above_running);
    RUN_TEST(test_unarmed_slow_curl_pause_is_not_an_end);
    RUN_TEST(test_switch_fast_arrival_fires_at_three_still_samples);
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

import sys

p = sys.argv[1]
s = open(p, newline='').read()
assert '\r' not in s


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


rep("""    TEST_ASSERT_EQUAL_INT(70, CAL_END_SWITCH_SLOW_PCT);
""", """    TEST_ASSERT_EQUAL_INT(70, CAL_END_SWITCH_SLOW_PCT);
    TEST_ASSERT_EQUAL_INT(90, CAL_END_SWITCH_SLOW_IREF_PCT);
    TEST_ASSERT_EQUAL_INT(12, CAL_END_SHORT_NORMAL_SAMPLES);
    TEST_ASSERT_EQUAL_INT(40, CAL_END_SHORT_RUN_SAMPLES);
""")

rep("""// Every 2026-11-24 robot move (40 moves, 3557 samples): the same verdict at the
// same sample as the Python reference, at the same drive edges. Includes the
// FRKL curl at PWM 80 called stuck (its current rose only about 11%) and the
// nine moves the host stopped while running (no decision).
static void test_robot_trace_parity_with_the_python_reference()
{
    int decided[5] = {};
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        CalibrationEndDetector detector;
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
        std::string label = std::string(move.name) + " " + move.joint + " PWM " + std::to_string(move.pwm) +
                            ": got " + kindName(got.kind) + " k=" + std::to_string(got.k);
        TEST_ASSERT_EQUAL_INT_MESSAGE(static_cast<int>(move.expected), static_cast<int>(got.kind), label.c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(move.k, got.k, label.c_str());
        if (move.k >= 0) TEST_ASSERT_EQUAL_INT_MESSAGE(move.edges, got.edges, label.c_str());
        ++decided[static_cast<int>(got.kind)];
    }
    TEST_ASSERT_EQUAL_INT(40, static_cast<int>(kRobotEndMoveCount));
    TEST_ASSERT_EQUAL_INT(9, decided[static_cast<int>(CalibrationEnd::None)]);
    TEST_ASSERT_EQUAL_INT(10, decided[static_cast<int>(CalibrationEnd::Switch)]);
    TEST_ASSERT_EQUAL_INT(11, decided[static_cast<int>(CalibrationEnd::Stop)]);
    TEST_ASSERT_EQUAL_INT(6, decided[static_cast<int>(CalibrationEnd::Stuck)]);
    TEST_ASSERT_EQUAL_INT(4, decided[static_cast<int>(CalibrationEnd::Still)]);
}

// The robot traces by end kind: every hip curl end and knee uncurl end at PWM
// 120/150 is a switch, recorded exactly where the count stopped; every knee
// curl end at 120/150 is a stop; no decision on a retract move is a stop and
// none on an extend move is a switch.
static void test_robot_traces_by_end_kind()
{
    for (size_t m = 0; m < kRobotEndMoveCount; ++m)
    {
        const RobotEndMove &move = kRobotEndMoves[m];
        if (move.retract)
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Stop, move.name);
        else
            TEST_ASSERT_TRUE_MESSAGE(move.expected != CalibrationEnd::Switch, move.name);
        if (move.pwm < 120 || move.expected == CalibrationEnd::None) continue;
""", """// Replays one robot move to its first decision.
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
    TEST_ASSERT_EQUAL_INT(7, stops);
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
""")

rep("""// calibrationCurrentTenths: round(10 * avgIS), clamped.""", """// F7: a retract stall whose current ripples +-30% (the FRHL stall at 128): a
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

// calibrationCurrentTenths: round(10 * avgIS), clamped.""")

rep("""    RUN_TEST(test_robot_traces_by_end_kind);
""", """    RUN_TEST(test_robot_traces_by_end_kind);
    RUN_TEST(test_frhl_mid_stroke_stall_at_128_is_stuck_not_a_switch);
    RUN_TEST(test_short_curls_into_their_end_are_not_stuck);
    RUN_TEST(test_pwm_128_runs_by_end_kind);
    RUN_TEST(test_retract_stall_with_rippling_current_is_stuck);
    RUN_TEST(test_short_curl_stall_ends_still_long_one_stuck);
    RUN_TEST(test_short_retract_curl_needs_the_current_not_above_running);
    RUN_TEST(test_unarmed_slow_curl_pause_is_not_an_end);
""")
rep("""// CalibrationEndDetector: parity with the Python reference on the 2026-11-24
// robot traces, then synthetic plants for each rule (switch end, compliant
// mechanical end, stuck joint, noise), the unarmed rules and the sampling.""", """// CalibrationEndDetector: parity with the Python reference on the 2026-11-24
// robot traces (the scripted end-current jogs, the PWM 128/150 runs and real
// end approaches started later), then synthetic plants for each rule (switch
// end, compliant mechanical end, stuck joint, rippling stall, short curl,
// noise), the unarmed rules and the sampling.""")
open(p, 'w', newline='').write(s)
print('ok')

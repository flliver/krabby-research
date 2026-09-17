#include "unity.h"
#include <functional>
#include <initializer_list>
#include "src/actuator/actuator_pot_tracker.h"

constexpr int POT_BAND_LO = POT_MINIMUM_RAW;
constexpr int POT_BAND_HI = POT_MAXIMUM_RAW;
constexpr uint8_t POS_DEBOUNCE = POT_INVALID_SAMPLE_LIMIT;
constexpr int IDLE_JITTER_MAX = POT_IDLE_JITTER_MAX;

namespace
{
bool update(ActuatorPotTracker &tracker, float avgPot, bool isDriving)
{
    return tracker.update<std::function<float()>>(
        avgPot,
        isDriving,
        0,
        []() { return 0; });
}

void applyBadSamples(
    ActuatorPotTracker &tracker,
    int rawPot,
    bool isDriving,
    uint8_t count)
{
    for (uint8_t sample = 0; sample < count; ++sample)
        update(tracker, rawPot, isDriving);
}
}

void setUp() {}
void tearDown() {}

static void test_open_probe_invalidates_independently_of_samples()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    bool wasProbeRead = false;

    TEST_ASSERT_FALSE(tracker.update<std::function<float()>>(
        500,
        false,
        POT_PROBE_INTERVAL_MS,
        [&wasProbeRead]()
        {
            wasProbeRead = true;
            return 520.0f;
        }));
    TEST_ASSERT_TRUE(wasProbeRead);
    TEST_ASSERT_TRUE(tracker.isPositionOpen());
    TEST_ASSERT_FALSE(tracker.isValid());

    TEST_ASSERT_FALSE(tracker.update<std::function<float()>>(
        500,
        false,
        POT_PROBE_INTERVAL_MS + 1,
        []() { return 0; }));
    TEST_ASSERT_FALSE(tracker.isValid());
}

static void test_open_probe_boundary_is_inclusive_and_clears()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    float probeRise = 9.75f;
    const auto readProbe = [&probeRise]() { return probeRise; };

    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS,
        readProbe);
    TEST_ASSERT_FALSE(tracker.isPositionOpen());

    probeRise = 10.0f;
    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS * 2,
        readProbe);
    TEST_ASSERT_TRUE(tracker.isPositionOpen());

    probeRise = 10.25f;
    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS * 3,
        readProbe);
    TEST_ASSERT_TRUE(tracker.isPositionOpen());

    probeRise = 0.0f;
    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS * 4,
        readProbe);
    TEST_ASSERT_FALSE(tracker.isPositionOpen());
    TEST_ASSERT_TRUE(tracker.isValid());
}

static void test_reset_clears_a_latched_open_probe()
{
    ActuatorPotTracker tracker;
    tracker.update<std::function<float()>>(
        500,
        false,
        POT_PROBE_INTERVAL_MS,
        []() { return 523.0f; });
    TEST_ASSERT_FALSE(tracker.isValid());

    tracker.reset(500);

    TEST_ASSERT_TRUE(tracker.isValid());
    TEST_ASSERT_FALSE(tracker.isPositionOpen());
}

static void test_probe_uses_rise_at_low_and_high_positions()
{
    const int positions[] = {100, 500, 990};
    for (int rawPot : positions)
    {
        ActuatorPotTracker tracker;
        tracker.reset(rawPot);
        TEST_ASSERT_TRUE(tracker.update<std::function<float()>>(
            rawPot, false, POT_PROBE_INTERVAL_MS,
            []() { return 2.0f; }));
        TEST_ASSERT_FALSE(tracker.update<std::function<float()>>(
            rawPot, false, POT_PROBE_INTERVAL_MS * 2,
            []() { return 10.0f; }));
    }
}

static void test_falling_probe_reading_does_not_indicate_open()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    TEST_ASSERT_TRUE(tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS,
        []() { return -20.0f; }));
    TEST_ASSERT_FALSE(tracker.isPositionOpen());
}

static void test_railed_samples_invalidate_even_without_probe_rise()
{
    ActuatorPotTracker tracker;
    tracker.reset(1023);
    for (uint8_t sample = 0; sample < POS_DEBOUNCE; ++sample)
        tracker.update<std::function<float()>>(
            1023, false, POT_PROBE_INTERVAL_MS,
            []() { return 0.0f; });
    TEST_ASSERT_FALSE(tracker.isPositionOpen());
    TEST_ASSERT_FALSE(tracker.isValid());
}

static void test_probe_only_runs_when_idle_and_due()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    uint8_t probeReadCount = 0;
    const auto readProbe = [&probeReadCount]()
    {
        ++probeReadCount;
        return 0.0f;
    };

    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS - 1,
        readProbe);
    tracker.update<std::function<float()>>(
        500, true, POT_PROBE_INTERVAL_MS,
        readProbe);
    TEST_ASSERT_EQUAL_UINT8(0, probeReadCount);

    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS,
        readProbe);
    TEST_ASSERT_EQUAL_UINT8(1, probeReadCount);
}

static void test_probe_interval_survives_millisecond_rollover()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    uint8_t probeReadCount = 0;
    const auto readProbe = [&probeReadCount]()
    {
        ++probeReadCount;
        return 0.0f;
    };

    tracker.update<std::function<float()>>(
        500, false, UINT32_MAX - 5,
        readProbe);
    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS - 7,
        readProbe);
    TEST_ASSERT_EQUAL_UINT8(1, probeReadCount);

    tracker.update<std::function<float()>>(
        500, false, POT_PROBE_INTERVAL_MS - 6,
        readProbe);
    TEST_ASSERT_EQUAL_UINT8(2, probeReadCount);
}

static void test_reset_seeds_previous_sample_and_valid_state()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);

    TEST_ASSERT_TRUE(tracker.isValid());
    TEST_ASSERT_TRUE(update(tracker, 500, false));
    TEST_ASSERT_TRUE(update(tracker, 500, false));
}

static void test_sane_band_boundaries_are_exclusive()
{
    const int rejected[] = {0, POT_BAND_LO, POT_BAND_HI, 1023};
    for (size_t value = 0; value < 4; ++value)
    {
        ActuatorPotTracker tracker;
        tracker.reset(rejected[value]);
        applyBadSamples(tracker, rejected[value], false, POS_DEBOUNCE);
        TEST_ASSERT_FALSE(tracker.isValid());
    }

    ActuatorPotTracker lowInside;
    lowInside.reset(POT_BAND_LO + 1);
    TEST_ASSERT_TRUE(update(lowInside, POT_BAND_LO + 1, false));
    ActuatorPotTracker highInside;
    highInside.reset(POT_BAND_HI - 1);
    TEST_ASSERT_TRUE(update(highInside, POT_BAND_HI - 1, false));
}

static void test_three_consecutive_bad_idle_samples_invalidate()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);

    TEST_ASSERT_TRUE(update(tracker, 507, false));
    TEST_ASSERT_TRUE(update(tracker, 500, false));
    TEST_ASSERT_FALSE(update(tracker, 507, false));
}

static void test_good_sample_breaks_bad_run_and_immediately_recovers()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);
    update(tracker, 507, false);
    update(tracker, 500, false);
    TEST_ASSERT_TRUE(update(tracker, 503, false));
    TEST_ASSERT_TRUE(update(tracker, 600, false));
    TEST_ASSERT_TRUE(update(tracker, 500, false));

    update(tracker, 600, false);
    update(tracker, 500, false);
    update(tracker, 600, false);
    TEST_ASSERT_FALSE(tracker.isValid());
    TEST_ASSERT_TRUE(update(tracker, 602, false));
}

static void test_idle_slew_boundary_is_inclusive()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);

    TEST_ASSERT_TRUE(update(tracker, 500 + IDLE_JITTER_MAX, false));

    TEST_ASSERT_TRUE(update(tracker, 1023, false));
    TEST_ASSERT_TRUE(update(tracker, 1023, false));
    TEST_ASSERT_FALSE(update(tracker, 1023, false));
}

static void test_driving_suppresses_slew_check_for_real_motion()
{
    ActuatorPotTracker tracker;
    tracker.reset(300);

    TEST_ASSERT_TRUE(update(tracker, 700, true));
    TEST_ASSERT_TRUE(update(tracker, 200, true));
}

static void test_driving_never_suppresses_rail_check()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);

    applyBadSamples(tracker, 1023, true, POS_DEBOUNCE);

    TEST_ASSERT_FALSE(tracker.isValid());
}

static void test_bad_counter_saturates_at_debounce_limit()
{
    ActuatorPotTracker tracker;
    tracker.reset(500);

    applyBadSamples(tracker, 1023, false, 20);
    TEST_ASSERT_FALSE(tracker.isValid());

    TEST_ASSERT_TRUE(update(tracker, 500, true));
}

static void test_fractional_jitter_is_not_truncated()
{
    ActuatorPotTracker tracker;
    tracker.reset(500.25f);
    TEST_ASSERT_TRUE(update(tracker, 506.5f, false));
    TEST_ASSERT_TRUE(update(tracker, 500.25f, false));
    TEST_ASSERT_FALSE(update(tracker, 506.5f, false));
    TEST_ASSERT_TRUE(update(tracker, 500.5f, false));
}

static void test_fractional_seed_and_rail_boundaries()
{
    ActuatorPotTracker tracker;
    tracker.reset(500.75f);
    for (int sample = 0; sample < 4; ++sample)
        TEST_ASSERT_TRUE(update(tracker, sample % 2 ? 500.75f : 506.75f, false));

    for (float inside : {5.25f, 1013.75f})
    {
        tracker.reset(inside);
        for (int sample = 0; sample < 4; ++sample)
            TEST_ASSERT_TRUE(update(tracker, inside, false));
    }
    for (float outside : {4.75f, 1014.25f})
    {
        tracker.reset(outside);
        TEST_ASSERT_TRUE(update(tracker, outside, false));
        TEST_ASSERT_TRUE(update(tracker, outside, false));
        TEST_ASSERT_FALSE(update(tracker, outside, false));
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_fractional_jitter_is_not_truncated);
    RUN_TEST(test_fractional_seed_and_rail_boundaries);
    RUN_TEST(test_open_probe_invalidates_independently_of_samples);
    RUN_TEST(test_open_probe_boundary_is_inclusive_and_clears);
    RUN_TEST(test_reset_clears_a_latched_open_probe);
    RUN_TEST(test_probe_uses_rise_at_low_and_high_positions);
    RUN_TEST(test_falling_probe_reading_does_not_indicate_open);
    RUN_TEST(test_railed_samples_invalidate_even_without_probe_rise);
    RUN_TEST(test_probe_only_runs_when_idle_and_due);
    RUN_TEST(test_probe_interval_survives_millisecond_rollover);
    RUN_TEST(test_reset_seeds_previous_sample_and_valid_state);
    RUN_TEST(test_sane_band_boundaries_are_exclusive);
    RUN_TEST(test_three_consecutive_bad_idle_samples_invalidate);
    RUN_TEST(test_good_sample_breaks_bad_run_and_immediately_recovers);
    RUN_TEST(test_idle_slew_boundary_is_inclusive);
    RUN_TEST(test_driving_suppresses_slew_check_for_real_motion);
    RUN_TEST(test_driving_never_suppresses_rail_check);
    RUN_TEST(test_bad_counter_saturates_at_debounce_limit);
    return UNITY_END();
}

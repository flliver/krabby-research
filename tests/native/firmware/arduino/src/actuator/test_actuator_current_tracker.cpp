#include "src/actuator/actuator_current_tracker.h"
#include "unity.h"

namespace
{
constexpr int CURRENT_PRESENT_FLOOR = ACTUATOR_CURRENT_PRESENT_FLOOR;
constexpr int QUALIFYING_PWM = ACTUATOR_CURRENT_PROBE_PWM;
constexpr uint8_t REQUIRED_SAMPLES = ACTUATOR_CURRENT_DEBOUNCE_SAMPLES;

void update(
    ActuatorCurrentTracker &tracker,
    bool isDriving,
    int commandedPwm,
    float current)
{
    tracker.update(isDriving, commandedPwm, current);
}
}

void setUp() {}
void tearDown() {}

static void test_starts_unknown()
{
    ActuatorCurrentTracker tracker;

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());
}

static void test_idle_current_is_not_evidence()
{
    ActuatorCurrentTracker tracker;

    for (int sample = 0; sample < 10; ++sample)
        update(tracker, false, QUALIFYING_PWM, 0);

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());
}

static void test_pwm_below_probe_threshold_is_not_evidence_in_either_direction()
{
    ActuatorCurrentTracker tracker;

    for (int sample = 0; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, QUALIFYING_PWM - 1, 0);
    for (int sample = 0; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, -QUALIFYING_PWM + 1, 0);

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());
}

static void test_absent_current_is_inclusive_and_debounced()
{
    ActuatorCurrentTracker tracker;

    for (int sample = 1; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());

    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentAbsent,
        (int)tracker.evidence());
}

static void test_present_current_threshold_is_inclusive_and_debounced()
{
    ActuatorCurrentTracker tracker;

    for (int sample = 1; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, -QUALIFYING_PWM, CURRENT_PRESENT_FLOOR);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());

    update(tracker, true, -QUALIFYING_PWM, CURRENT_PRESENT_FLOOR);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentPresent,
        (int)tracker.evidence());
}

static void test_opposite_evidence_breaks_a_pending_run()
{
    ActuatorCurrentTracker tracker;

    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR);
    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());
    update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentAbsent,
        (int)tracker.evidence());
}

static void test_unusable_sample_clears_pending_counts_but_retains_evidence()
{
    ActuatorCurrentTracker tracker;

    update(tracker, true, QUALIFYING_PWM, 0);
    update(tracker, true, QUALIFYING_PWM, 0);
    update(tracker, false, QUALIFYING_PWM, 1023);
    update(tracker, true, QUALIFYING_PWM, 0);
    update(tracker, true, QUALIFYING_PWM, 0);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::Unknown,
        (int)tracker.evidence());

    update(tracker, true, QUALIFYING_PWM, 0);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentAbsent,
        (int)tracker.evidence());
    for (int sample = 0; sample < 10; ++sample)
        update(tracker, false, QUALIFYING_PWM, 1023);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentAbsent,
        (int)tracker.evidence());
}

static void test_present_current_replaces_absent_evidence()
{
    ActuatorCurrentTracker tracker;
    for (int sample = 0; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR - 1);

    for (int sample = 0; sample < REQUIRED_SAMPLES; ++sample)
        update(tracker, true, QUALIFYING_PWM, CURRENT_PRESENT_FLOOR);

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorCurrentEvidence::CurrentPresent,
        (int)tracker.evidence());
}

static void test_evidence_counters_saturate_without_wrapping()
{
    ActuatorCurrentTracker tracker;
    for (int sample = 0; sample < 300; ++sample)
    {
        tracker.update(true, QUALIFYING_PWM, 1.25f);
        if (sample >= REQUIRED_SAMPLES - 1)
            TEST_ASSERT_EQUAL_INT((int)ActuatorCurrentEvidence::CurrentPresent, (int)tracker.evidence());
    }
    for (int sample = 0; sample < 300; ++sample)
    {
        tracker.update(true, -QUALIFYING_PWM, 0.75f);
        if (sample >= REQUIRED_SAMPLES - 1)
            TEST_ASSERT_EQUAL_INT((int)ActuatorCurrentEvidence::CurrentAbsent, (int)tracker.evidence());
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_evidence_counters_saturate_without_wrapping);
    RUN_TEST(test_starts_unknown);
    RUN_TEST(test_idle_current_is_not_evidence);
    RUN_TEST(test_pwm_below_probe_threshold_is_not_evidence_in_either_direction);
    RUN_TEST(test_absent_current_is_inclusive_and_debounced);
    RUN_TEST(test_present_current_threshold_is_inclusive_and_debounced);
    RUN_TEST(test_opposite_evidence_breaks_a_pending_run);
    RUN_TEST(test_unusable_sample_clears_pending_counts_but_retains_evidence);
    RUN_TEST(test_present_current_replaces_absent_evidence);
    return UNITY_END();
}

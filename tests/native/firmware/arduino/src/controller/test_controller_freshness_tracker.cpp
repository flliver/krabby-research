#include <stdint.h>

#include "src/controller/controller_freshness_tracker.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

static void test_unseen_controller_is_never_fresh()
{
    ControllerFreshnessTracker tracker;

    TEST_ASSERT_FALSE(tracker.isFresh(0, 1000));
    TEST_ASSERT_FALSE(tracker.isFresh(UINT32_MAX, 1000));
}

static void test_seen_controller_is_fresh_until_but_not_at_timeout()
{
    const ControllerFreshnessTracker tracker =
        ControllerFreshnessTracker::seenAt(1000);

    TEST_ASSERT_TRUE(tracker.isFresh(1000, 500));
    TEST_ASSERT_TRUE(tracker.isFresh(1499, 500));
    TEST_ASSERT_FALSE(tracker.isFresh(1500, 500));
}

static void test_zero_timeout_is_always_stale()
{
    const ControllerFreshnessTracker tracker =
        ControllerFreshnessTracker::seenAt(1000);

    TEST_ASSERT_FALSE(tracker.isFresh(1000, 0));
}

static void test_freshness_elapsed_time_survives_millisecond_rollover()
{
    const ControllerFreshnessTracker tracker =
        ControllerFreshnessTracker::seenAt(UINT32_MAX - 20);

    TEST_ASSERT_TRUE(tracker.isFresh(10, 32));
    TEST_ASSERT_FALSE(tracker.isFresh(11, 32));
}

static void test_new_seen_snapshot_restarts_the_freshness_window()
{
    ControllerFreshnessTracker tracker =
        ControllerFreshnessTracker::seenAt(1000);
    TEST_ASSERT_FALSE(tracker.isFresh(1500, 500));

    tracker = ControllerFreshnessTracker::seenAt(1500);

    TEST_ASSERT_TRUE(tracker.isFresh(1500, 500));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_unseen_controller_is_never_fresh);
    RUN_TEST(test_seen_controller_is_fresh_until_but_not_at_timeout);
    RUN_TEST(test_zero_timeout_is_always_stale);
    RUN_TEST(test_freshness_elapsed_time_survives_millisecond_rollover);
    RUN_TEST(test_new_seen_snapshot_restarts_the_freshness_window);
    return UNITY_END();
}

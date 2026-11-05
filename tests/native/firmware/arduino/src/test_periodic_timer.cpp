#include <stdint.h>

#include <unity.h>

#include "src/periodic_timer.h"

void setUp() {}
void tearDown() {}

static void test_timer_is_inactive_until_restarted()
{
    PeriodicTimer timer(100);

    TEST_ASSERT_FALSE(timer.isExpired(100));
    TEST_ASSERT_FALSE(timer.isExpired(UINT32_MAX));
}

static void test_expiration_check_does_not_restart_the_timer()
{
    PeriodicTimer timer(100);
    timer.restart(1000);

    TEST_ASSERT_FALSE(timer.isExpired(1099));
    TEST_ASSERT_TRUE(timer.isExpired(1100));
    TEST_ASSERT_TRUE(timer.isExpired(1100));
    timer.restart(1100);
    TEST_ASSERT_FALSE(timer.isExpired(1100));
}

static void test_late_expiration_restarts_from_observation_time()
{
    PeriodicTimer timer(100);
    timer.restart(1000);

    TEST_ASSERT_TRUE(timer.isExpired(1150));
    timer.restart(1150);
    TEST_ASSERT_FALSE(timer.isExpired(1249));
    TEST_ASSERT_TRUE(timer.isExpired(1250));
}

static void test_manual_restart_replaces_the_timing_origin()
{
    PeriodicTimer timer(100);
    timer.restart(1000);
    timer.restart(1050);

    TEST_ASSERT_FALSE(timer.isExpired(1149));
    TEST_ASSERT_TRUE(timer.isExpired(1150));
}

static void test_timer_handles_millis_rollover()
{
    PeriodicTimer timer(10);
    timer.restart(UINT32_MAX - 5);

    TEST_ASSERT_FALSE(timer.isExpired(3));
    TEST_ASSERT_TRUE(timer.isExpired(4));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_timer_is_inactive_until_restarted);
    RUN_TEST(test_expiration_check_does_not_restart_the_timer);
    RUN_TEST(test_late_expiration_restarts_from_observation_time);
    RUN_TEST(test_manual_restart_replaces_the_timing_origin);
    RUN_TEST(test_timer_handles_millis_rollover);
    return UNITY_END();
}

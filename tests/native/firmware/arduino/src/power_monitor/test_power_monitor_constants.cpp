#include <type_traits>

#include "unity.h"

#include "src/power_monitor/power_monitor_constants.h"

static_assert(
    std::is_same<decltype(BATTERY_DIVERGENCE_THRESHOLD), const Volts>::value,
    "divergence threshold must retain its unit");
static_assert(
    std::is_same<decltype(EEPROM_POWER_CAL_ADDR), const uint16_t>::value,
    "EEPROM address must be explicitly typed");

void setUp() {}
void tearDown() {}

static void test_power_monitor_policy_matches_the_build()
{
    TEST_ASSERT_FLOAT_WITHIN(
        0.000001f, 0.5f, BATTERY_DIVERGENCE_THRESHOLD.value());
    TEST_ASSERT_EQUAL_UINT8(3, POWER_MONITOR_REINIT_AFTER_BAD_TICKS);
    TEST_ASSERT_EQUAL_UINT32(
        2000, POWER_MONITOR_REINIT_INTERVAL_MILLISECONDS);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_power_monitor_policy_matches_the_build);
    return UNITY_END();
}

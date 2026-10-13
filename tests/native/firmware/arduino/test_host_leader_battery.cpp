#include "unity.h"

#include <iomanip>
#include <sstream>
#include <string>

#include "host_battery_telemetry.hpp"
#include "sim_battery_model.hpp"
#include "sim_clock.hpp"
#include "sim_i2c.hpp"

using krabby::sim::SimBatteryTelemetry;
using krabby::sim::host::appendBatteryTelemetry;
using krabby::sim::host::leaderBattery;
using krabby::sim::simI2cBus;

void setUp() { krabby::sim::resetHostSim(); }

void tearDown() {}

struct FakeOutput
{
    std::ostringstream text;
    void print(char value) { text << value; }
    void print(const char *value) { text << value; }
    void print(int value) { text << value; }
    void print(float value, int digits)
    {
        text << std::fixed << std::setprecision(digits) << value;
    }
};

static void test_setup_online_when_ina228_addresses_present()
{
    leaderBattery().setup();
    TEST_ASSERT_TRUE(leaderBattery().isOnline());
}

static void test_sample_nominal_pack_and_cells()
{
    leaderBattery().setup();
    const SimBatteryTelemetry sample = leaderBattery().sample();
    TEST_ASSERT_TRUE(sample.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 26.40f, sample.packVolts);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.250f, sample.packAmps);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 13.20f, sample.battAVolts);
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 13.20f, sample.battBVolts);
    TEST_ASSERT_EQUAL_UINT8(0, sample.divergenceFlag);
}

static void test_append_matches_golden_segment()
{
    leaderBattery().setup();
    FakeOutput output;
    appendBatteryTelemetry(output, leaderBattery().sample());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.40 1.250 33.00 100000.0 13.20 13.20 0 0",
        output.text.str().c_str());
}

static void test_offline_when_pack_monitor_absent()
{
    simI2cBus().setDevicePresent(krabby::sim::kI2cAddrIna228Pack, false);
    leaderBattery().setup();
    TEST_ASSERT_FALSE(leaderBattery().isOnline());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_setup_online_when_ina228_addresses_present);
    RUN_TEST(test_sample_nominal_pack_and_cells);
    RUN_TEST(test_append_matches_golden_segment);
    RUN_TEST(test_offline_when_pack_monitor_absent);
    return UNITY_END();
}

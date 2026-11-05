#include "unity.h"

#include <iomanip>
#include <sstream>
#include <string>

#include "EEPROM.h"
#include "host_battery_telemetry.hpp"
#include "sim_battery_model.hpp"
#include "sim_clock.hpp"
#include "sim_i2c.hpp"

using krabby::sim::host::appendBatteryTelemetry;
using krabby::sim::host::leaderBattery;
using krabby::sim::simI2cBus;
using krabby::sim::kI2cAddrIna228Pack;
using krabby::sim::kI2cAddrIna228Midpoint;

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

static void test_setup_uses_real_driver_configuration_and_resets_charge()
{
    simI2cBus().packIna228().coulombs = 1000.0;
    leaderBattery().setup();
    TEST_ASSERT_EQUAL_UINT16(1875, simI2cBus().packIna228().registerValue(ina228_native::SHUNT_CAL));
    TEST_ASSERT_EQUAL_UINT(1, simI2cBus().packIna228().resetCount);
    TEST_ASSERT_EQUAL_UINT(1, simI2cBus().packIna228().accumulatorResetCount);
    const auto &sample = leaderBattery().sample();
    TEST_ASSERT_TRUE(sample.pack.isValid);
    TEST_ASSERT_TRUE(sample.midpoint.isValid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 26.40f, sample.pack.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.250f, sample.pack.current.value());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, sample.pack.charge.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 13.20f, sample.midpoint.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 13.20f, sample.batteryB.value());
    TEST_ASSERT_FALSE(sample.isDiverged);
    TEST_ASSERT_TRUE(simI2cBus().packIna228().operations.empty());
}

static void test_telemetry_uses_hardware_format_and_per_monitor_validity()
{
    leaderBattery().setup();
    FakeOutput output;
    appendBatteryTelemetry(output, leaderBattery().sample());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.40 1.25 33.0 0.0 13.20 13.20 0 0 1 1",
        output.text.str().c_str());
}

static void test_reads_are_decoded_from_registers_not_synthetic_values()
{
    leaderBattery().setup();
    // Corrupt SHUNT_CAL over runtime Wire: the driver must decode a different
    // current even though the physical input remains 1.25 A.
    Wire.beginTransmission(kI2cAddrIna228Pack);
    Wire.write(ina228_native::SHUNT_CAL);
    Wire.write(static_cast<uint8_t>(3750 >> 8));
    Wire.write(static_cast<uint8_t>(3750));
    TEST_ASSERT_EQUAL_UINT8(0, Wire.endTransmission());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.625f, leaderBattery().sample().pack.current.value());
}

static void test_physical_input_updates_reach_registers_and_signed_charge()
{
    leaderBattery().setup();
    krabby::sim::SimBatteryState inputs;
    inputs.packVolts = 26.25f;
    inputs.battAVolts = 12.75f;
    inputs.packAmps = -3.125f;
    inputs.packChargeCoulombs = -1000.0f;
    krabby::sim::setSimBatteryState(inputs);
    const auto &sample = leaderBattery().sample();
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 26.25f, sample.pack.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -3.125f, sample.pack.current.value());
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 82.03125f, sample.pack.power.value());
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -1000.0f, sample.pack.charge.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 13.5f, sample.batteryB.value());
    TEST_ASSERT_TRUE(sample.isDiverged);
}

static void test_pack_absent_at_boot_does_not_suppress_midpoint_and_recovers()
{
    simI2cBus().setDevicePresent(kI2cAddrIna228Pack, false);
    leaderBattery().setup();
    auto sample = leaderBattery().sample();
    TEST_ASSERT_FALSE(sample.pack.isValid);
    TEST_ASSERT_TRUE(isnan(sample.pack.voltage.value()));
    TEST_ASSERT_TRUE(sample.midpoint.isValid);
    TEST_ASSERT_TRUE(sample.isDiverged);
    simI2cBus().setDevicePresent(kI2cAddrIna228Pack, true);
    for (int i = 0; i < 3; ++i) sample = leaderBattery().sample();
    TEST_ASSERT_TRUE(sample.pack.isValid);
    TEST_ASSERT_TRUE(sample.midpoint.isValid);
}

static void test_midpoint_disconnect_reports_invalid_then_recovers()
{
    leaderBattery().setup();
    simI2cBus().setDevicePresent(kI2cAddrIna228Midpoint, false);
    auto sample = leaderBattery().sample();
    TEST_ASSERT_TRUE(sample.pack.isValid);
    TEST_ASSERT_FALSE(sample.midpoint.isValid);
    TEST_ASSERT_TRUE(isnan(sample.batteryB.value()));
    TEST_ASSERT_TRUE(sample.isDiverged);
    simI2cBus().setDevicePresent(kI2cAddrIna228Midpoint, true);
    sample = leaderBattery().sample();
    TEST_ASSERT_TRUE(sample.midpoint.isValid);
}

static void test_partial_register_read_failure_keeps_other_readings()
{
    leaderBattery().setup();
    simI2cBus().packIna228().nack = [](uint8_t reg, bool isWrite, uint16_t) {
        return !isWrite && reg == ina228_native::CURRENT;
    };
    const auto &sample = leaderBattery().sample();
    TEST_ASSERT_FALSE(sample.pack.isValid);
    TEST_ASSERT_TRUE(isnan(sample.pack.current.value()));
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 26.4f, sample.pack.voltage.value());
    TEST_ASSERT_TRUE(sample.midpoint.isValid);
}

static void test_wrong_identity_is_rejected_and_recovery_preserves_charge()
{
    simI2cBus().packIna228().deviceId = 0x2380;
    leaderBattery().setup();
    TEST_ASSERT_FALSE(leaderBattery().sample().pack.isValid);
    simI2cBus().packIna228().deviceId = ina228_native::INA228_DEVICE_ID;
    for (int i = 0; i < 3; ++i) leaderBattery().sample();
    TEST_ASSERT_TRUE(leaderBattery().latest().pack.isValid);
    simI2cBus().packIna228().coulombs = 123.0;
    const auto resets = simI2cBus().packIna228().resetCount;
    simI2cBus().setDevicePresent(kI2cAddrIna228Pack, false);
    for (int i = 0; i < 3; ++i) leaderBattery().sample();
    simI2cBus().setDevicePresent(kI2cAddrIna228Pack, true);
    krabby::sim::simClock().advance(2000);
    for (int i = 0; i < 3; ++i) leaderBattery().sample();
    TEST_ASSERT_TRUE(leaderBattery().latest().pack.isValid);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 123.0f, leaderBattery().latest().pack.charge.value());
    TEST_ASSERT_EQUAL_UINT(resets, simI2cBus().packIna228().resetCount);
}

static void test_front_eeprom_calibration_applies_after_driver_decode()
{
    const PowerCalibrationRecord record = {
        EEPROM_POWER_CAL_MAGIC, EEPROM_POWER_CAL_SCHEMA, 0.25f, -0.25f, 1.5f};
    EEPROM.boards[0].put(EEPROM_POWER_CAL_ADDR, record);
    EEPROM.setActiveBoard(krabby::sim::BoardId::kLeft);
    leaderBattery().setup();
    const auto &sample = leaderBattery().sample();
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 26.65f, sample.pack.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 12.95f, sample.midpoint.voltage.value());
    TEST_ASSERT_FLOAT_WITHIN(0.002f, 1.875f, sample.pack.current.value());
    TEST_ASSERT_FLOAT_WITHIN(0.002f, 13.7f, sample.batteryB.value());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_setup_uses_real_driver_configuration_and_resets_charge);
    RUN_TEST(test_telemetry_uses_hardware_format_and_per_monitor_validity);
    RUN_TEST(test_reads_are_decoded_from_registers_not_synthetic_values);
    RUN_TEST(test_physical_input_updates_reach_registers_and_signed_charge);
    RUN_TEST(test_pack_absent_at_boot_does_not_suppress_midpoint_and_recovers);
    RUN_TEST(test_midpoint_disconnect_reports_invalid_then_recovers);
    RUN_TEST(test_partial_register_read_failure_keeps_other_readings);
    RUN_TEST(test_wrong_identity_is_rejected_and_recovery_preserves_charge);
    RUN_TEST(test_front_eeprom_calibration_applies_after_driver_decode);
    return UNITY_END();
}

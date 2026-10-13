#include "unity.h"

#include "Arduino.h"
#include "sim_i2c.hpp"

using krabby::sim::kI2cAddrIna228Midpoint;
using krabby::sim::kI2cAddrIna228Pack;
using krabby::sim::kI2cAddrLsm6dsoAlternate;
using krabby::sim::kI2cAddrLsm6dsoPrimary;
using krabby::sim::kI2cAddrSsd1306Alternate;
using krabby::sim::kI2cAddrSsd1306Primary;
using krabby::sim::kI2cNackAddress;
using krabby::sim::simI2cBus;

void setUp() { krabby::sim::resetHostSim(); }

void tearDown() {}

static bool i2cAddressResponds(uint8_t address)
{
    Wire.clearWireTimeoutFlag();
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

static void test_shipping_imu_and_oled_addresses_ack_probe()
{
    Wire.begin();
    TEST_ASSERT_TRUE(i2cAddressResponds(kI2cAddrLsm6dsoPrimary));
    TEST_ASSERT_TRUE(i2cAddressResponds(kI2cAddrLsm6dsoAlternate));
    TEST_ASSERT_TRUE(i2cAddressResponds(kI2cAddrSsd1306Primary));
    TEST_ASSERT_TRUE(i2cAddressResponds(kI2cAddrSsd1306Alternate));
}

static void test_ina228_and_legacy_imu_addresses_nack()
{
    Wire.begin();
    TEST_ASSERT_FALSE(i2cAddressResponds(kI2cAddrIna228Pack));
    TEST_ASSERT_FALSE(i2cAddressResponds(kI2cAddrIna228Midpoint));
    TEST_ASSERT_FALSE(i2cAddressResponds(0x68));
    TEST_ASSERT_FALSE(i2cAddressResponds(0x69));
}

static void test_missing_device_end_transmission_returns_nack_code()
{
    Wire.beginTransmission(0x55);
    TEST_ASSERT_EQUAL_UINT8(kI2cNackAddress, Wire.endTransmission());
}

static void test_lsm6dso_register_read_burst_matches_adapter_pattern()
{
    constexpr uint8_t kOutputStart = 0x20;
    constexpr uint8_t kNumBytes = 14;

    Wire.begin();
    Wire.beginTransmission(kI2cAddrLsm6dsoPrimary);
    TEST_ASSERT_EQUAL_INT(1, Wire.write(kOutputStart));
    TEST_ASSERT_EQUAL_UINT8(0, Wire.endTransmission(false));

    TEST_ASSERT_EQUAL_UINT8(
        kNumBytes,
        Wire.requestFrom(kI2cAddrLsm6dsoPrimary, kNumBytes, static_cast<uint8_t>(true)));

    uint8_t bytes[kNumBytes];
    for (uint8_t i = 0; i < kNumBytes; ++i)
    {
        TEST_ASSERT_TRUE(Wire.available() > 0);
        bytes[i] = static_cast<uint8_t>(Wire.read());
    }
    TEST_ASSERT_FALSE(Wire.available());

    const int16_t rawAccelZ =
        static_cast<int16_t>(bytes[12] | (static_cast<uint16_t>(bytes[13]) << 8));
    TEST_ASSERT_INT16_WITHIN(2, 4098, rawAccelZ);
}

static void test_request_from_absent_device_returns_zero()
{
    simI2cBus().setDevicePresent(kI2cAddrIna228Pack, false);
    Wire.begin();
    TEST_ASSERT_EQUAL_UINT8(0, Wire.requestFrom(kI2cAddrIna228Pack, 4, static_cast<uint8_t>(true)));
    TEST_ASSERT_FALSE(Wire.available());
}

static void test_reset_restores_default_device_map()
{
    simI2cBus().setDevicePresent(kI2cAddrLsm6dsoPrimary, false);
    krabby::sim::resetHostSim();
    Wire.begin();
    TEST_ASSERT_TRUE(i2cAddressResponds(kI2cAddrLsm6dsoPrimary));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_shipping_imu_and_oled_addresses_ack_probe);
    RUN_TEST(test_ina228_and_legacy_imu_addresses_nack);
    RUN_TEST(test_missing_device_end_transmission_returns_nack_code);
    RUN_TEST(test_lsm6dso_register_read_burst_matches_adapter_pattern);
    RUN_TEST(test_request_from_absent_device_returns_zero);
    RUN_TEST(test_reset_restores_default_device_map);
    return UNITY_END();
}

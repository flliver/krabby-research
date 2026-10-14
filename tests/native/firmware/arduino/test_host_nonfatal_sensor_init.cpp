#include "unity.h"

#include <cstring>

#include "host_api.h"
#include "sim_i2c.hpp"
#include "sim_serial.hpp"

using krabby::sim::SerialPortId;
using krabby::sim::kI2cAddrLsm6dsoAlternate;
using krabby::sim::kI2cAddrLsm6dsoPrimary;
using krabby::sim::simI2cBus;
using krabby::sim::serialPort;

void setUp() { krabby_host_reset(); }

void tearDown() {}

static void drainUsbTxLines()
{
    while (serialPort(SerialPortId::kUsb).txLineAvailable())
    {
        (void)serialPort(SerialPortId::kUsb).popTxLine();
    }
}

static std::string popFrontTelemetryLine()
{
    krabby_host_firmware_setup();
    drainUsbTxLines();
    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    return serialPort(SerialPortId::kUsb).popTxLine();
}

static void test_imu_absent_emits_valid_zero_and_control_loop_continues()
{
    simI2cBus().setDevicePresent(kI2cAddrLsm6dsoPrimary, false);
    simI2cBus().setDevicePresent(kI2cAddrLsm6dsoAlternate, false);

    const std::string line = popFrontTelemetryLine();
    TEST_ASSERT_TRUE(line.rfind("FRONT;", 0) == 0);
    TEST_ASSERT_NOT_NULL(std::strstr(line.c_str(), ";IMU 0.000 0.000 0.000 0.0000 0.0000 0.0000 0.0 0"));

    krabby_host_handle_jog("FLHY", 80);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_EQUAL_INT(1, krabby_host_pin_digital(22));
}

static void test_one_disabled_imu_address_falls_back_to_alternate()
{
    simI2cBus().setDevicePresent(kI2cAddrLsm6dsoPrimary, false);
    simI2cBus().setDevicePresent(kI2cAddrLsm6dsoAlternate, true);

    const std::string line = popFrontTelemetryLine();
    TEST_ASSERT_TRUE(line.rfind("FRONT;", 0) == 0);
    TEST_ASSERT_NOT_NULL(std::strstr(line.c_str(), ";IMU "));
    TEST_ASSERT_NOT_NULL(std::strstr(line.c_str(), " 1"));
    TEST_ASSERT_NULL(std::strstr(line.c_str(), ";IMU 0.000 0.000 0.000 0.0000 0.0000 0.0000 0.0 0"));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_imu_absent_emits_valid_zero_and_control_loop_continues);
    RUN_TEST(test_one_disabled_imu_address_falls_back_to_alternate);
    return UNITY_END();
}

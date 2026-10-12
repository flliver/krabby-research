#include "unity.h"

#include "host_api.h"
#include "sim_serial.hpp"

using krabby::sim::SerialPortId;
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

static void test_telemetry_emits_on_fifty_ms_interval()
{
    krabby_host_firmware_setup();
    drainUsbTxLines();
    krabby_host_advance_millis(10);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_FALSE(serialPort(SerialPortId::kUsb).txLineAvailable());

    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    const std::string line = serialPort(SerialPortId::kUsb).popTxLine();
    TEST_ASSERT_TRUE(line.rfind("FRONT;", 0) == 0);
}

static void test_jog_command_reaches_actuator_via_loop()
{
    const char cmd[] = "JFLHY 100\n";
    krabby_host_serial_usb_rx_write(
        reinterpret_cast<const uint8_t *>(cmd),
        static_cast<int>(sizeof(cmd) - 1));

    krabby_host_firmware_setup();
    krabby_host_firmware_loop_once();

    TEST_ASSERT_EQUAL_INT(1, krabby_host_pin_digital(22));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_telemetry_emits_on_fifty_ms_interval);
    RUN_TEST(test_jog_command_reaches_actuator_via_loop);
    return UNITY_END();
}

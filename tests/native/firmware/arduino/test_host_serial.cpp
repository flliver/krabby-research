#include "unity.h"

#include "Arduino.h"
#include "sim_serial.hpp"

using krabby::sim::serialPort;
using krabby::sim::SerialPortId;

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_println_visible_as_newline_delimited_line()
{
    Serial.println("Krabby Ready");
    TEST_ASSERT_TRUE(serialPort(SerialPortId::kUsb).txLineAvailable());
    TEST_ASSERT_EQUAL_STRING("Krabby Ready", serialPort(SerialPortId::kUsb).popTxLine().c_str());
}

static void test_each_hardware_serial_port_has_separate_tx_lines()
{
    Serial.println("usb");
    Serial1.println("left");
    Serial2.println("right");

    TEST_ASSERT_EQUAL_STRING("usb", serialPort(SerialPortId::kUsb).popTxLine().c_str());
    TEST_ASSERT_EQUAL_STRING("left", serialPort(SerialPortId::kSerial1).popTxLine().c_str());
    TEST_ASSERT_EQUAL_STRING("right", serialPort(SerialPortId::kSerial2).popTxLine().c_str());
}

static void test_rx_read_string_until_newline()
{
    const char payload[] = "SYNC\n";
    serialPort(SerialPortId::kSerial1).rxPush(
        reinterpret_cast<const uint8_t *>(payload),
        sizeof(payload) - 1);

    TEST_ASSERT_TRUE(Serial1.available());
    String line = Serial1.readStringUntil('\n');
    TEST_ASSERT_EQUAL_STRING("SYNC", line.c_str());
    TEST_ASSERT_FALSE(Serial1.available());
}

static void test_reset_clears_tx_and_rx()
{
    Serial.println("boot");
    serialPort(SerialPortId::kUsb).rxPushByte('H');
    krabby::sim::resetHostSim();
    TEST_ASSERT_FALSE(serialPort(SerialPortId::kUsb).txLineAvailable());
    TEST_ASSERT_FALSE(Serial.available());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_println_visible_as_newline_delimited_line);
    RUN_TEST(test_each_hardware_serial_port_has_separate_tx_lines);
    RUN_TEST(test_rx_read_string_until_newline);
    RUN_TEST(test_reset_clears_tx_and_rx);
    return UNITY_END();
}

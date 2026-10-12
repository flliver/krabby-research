#include "unity.h"

#include "Arduino.h"
#include "sim_serial.hpp"

using krabby::sim::BoardHardwareSerial;
using krabby::sim::BoardId;
using krabby::sim::SerialPortId;
using krabby::sim::serialPort;

void setUp()
{
    krabby::sim::resetHostSim();
    krabby::sim::wireDefaultThreeBoardSerialLinks();
}

void tearDown() {}

static void test_front_serial1_tx_received_on_left_serial1()
{
    BoardHardwareSerial front_serial1(BoardId::kFront, SerialPortId::kSerial1);
    BoardHardwareSerial left_serial1(BoardId::kLeft, SerialPortId::kSerial1);

    front_serial1.println("SYNC");

    TEST_ASSERT_TRUE(left_serial1.available());
    String line = left_serial1.readStringUntil('\n');
    TEST_ASSERT_EQUAL_STRING("SYNC", line.c_str());
    TEST_ASSERT_FALSE(left_serial1.available());
}

static void test_left_serial1_tx_received_on_front_serial1()
{
    BoardHardwareSerial front_serial1(BoardId::kFront, SerialPortId::kSerial1);
    BoardHardwareSerial left_serial1(BoardId::kLeft, SerialPortId::kSerial1);

    left_serial1.print("PING");
    left_serial1.println("");

    TEST_ASSERT_TRUE(front_serial1.available());
    String line = front_serial1.readStringUntil('\n');
    TEST_ASSERT_EQUAL_STRING("PING", line.c_str());
}

static void test_front_serial2_tx_received_on_right_serial2()
{
    BoardHardwareSerial front_serial2(BoardId::kFront, SerialPortId::kSerial2);
    BoardHardwareSerial right_serial2(BoardId::kRight, SerialPortId::kSerial2);

    front_serial2.println("RIGHT-LINK");

    TEST_ASSERT_TRUE(right_serial2.available());
    String line = right_serial2.readStringUntil('\n');
    TEST_ASSERT_EQUAL_STRING("RIGHT-LINK", line.c_str());
}

static void test_unlinked_ports_do_not_cross_boards()
{
    BoardHardwareSerial front_usb(BoardId::kFront, SerialPortId::kUsb);
    BoardHardwareSerial left_serial2(BoardId::kLeft, SerialPortId::kSerial2);

    front_usb.println("usb-only");

    TEST_ASSERT_FALSE(left_serial2.available());
    TEST_ASSERT_TRUE(serialPort(BoardId::kFront, SerialPortId::kUsb).txLineAvailable());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_front_serial1_tx_received_on_left_serial1);
    RUN_TEST(test_left_serial1_tx_received_on_front_serial1);
    RUN_TEST(test_front_serial2_tx_received_on_right_serial2);
    RUN_TEST(test_unlinked_ports_do_not_cross_boards);
    return UNITY_END();
}

#include "unity.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "host_api.h"
#include "sim_serial.hpp"

using krabby::sim::BoardId;
using krabby::sim::SerialPortId;
using krabby::sim::SimSerialPort;
using krabby::sim::serialPort;

void setUp() { krabby_host_reset(); }

void tearDown() {}

static std::set<std::string> drainUsbRolePrefixes()
{
    std::set<std::string> roles;
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("FRONT;", 0) == 0) roles.insert("FRONT");
        if (line.rfind("LEFT ;", 0) == 0) roles.insert("LEFT");
        if (line.rfind("RIGHT;", 0) == 0) roles.insert("RIGHT");
    }
    return roles;
}

static std::vector<std::string> drainInterBoardRelayLines(const SerialPortId link)
{
    std::vector<std::string> lines;
    SimSerialPort &port = serialPort(BoardId::kFront, link);
    while (port.txLineAvailable()) lines.push_back(port.popTxLine());
    return lines;
}

static void bootThreeBoardLeaderLoop()
{
    int roles[3] = {0, 0, 0};
    krabby_host_run_three_board_role_election(roles);
    krabby_host_enable_three_board_firmware_loop(1);
    krabby_host_firmware_setup();
}

static void injectUsbCommand(const char *wire)
{
    krabby_host_serial_usb_rx_write(
        reinterpret_cast<const uint8_t *>(wire),
        static_cast<int>(std::strlen(wire)));
    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();
}

static void assertBothFollowerLinksRelayedSameLine(
    const char *expectedSubstring,
    const std::vector<std::string> &leftLines,
    const std::vector<std::string> &rightLines)
{
    TEST_ASSERT_FALSE(leftLines.empty());
    TEST_ASSERT_FALSE(rightLines.empty());
    const std::string &left = leftLines.back();
    const std::string &right = rightLines.back();
    TEST_ASSERT_EQUAL_STRING(left.c_str(), right.c_str());
    TEST_ASSERT_NOT_NULL(std::strstr(left.c_str(), expectedSubstring));
}

static bool lineHasJointPwm(const std::string &line, const char *joint, int minPwm)
{
    const std::size_t pos = line.find(joint);
    if (pos == std::string::npos) return false;
    const std::size_t fieldStart = line.find(' ', pos + std::strlen(joint));
    if (fieldStart == std::string::npos) return false;
    int pwmL = 0;
    int pwmR = 0;
    if (std::sscanf(
            line.c_str() + fieldStart,
            " %*f %*d %*d %*d %*d %d %d",
            &pwmL,
            &pwmR) < 2)
    {
        return false;
    }
    return pwmL >= minPwm || pwmR >= minPwm;
}

static void test_usb_telemetry_includes_three_roles_each_cycle()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbRolePrefixes();

    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();

    const std::set<std::string> roles = drainUsbRolePrefixes();
    TEST_ASSERT_TRUE(roles.count("FRONT") > 0);
    TEST_ASSERT_TRUE(roles.count("LEFT") > 0);
    TEST_ASSERT_TRUE(roles.count("RIGHT") > 0);
}

static void test_jog_relay_targets_left_joint_not_front()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbRolePrefixes();

    injectUsbCommand("JRLHY 200\n");

    std::string frontLine;
    std::string leftLine;
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("FRONT;", 0) == 0) frontLine = line;
        if (line.rfind("LEFT ;", 0) == 0) leftLine = line;
    }

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("JRLHY", leftRelay, rightRelay);

    TEST_ASSERT_TRUE(leftLine.find("RLHY") != std::string::npos);
    TEST_ASSERT_TRUE(lineHasJointPwm(leftLine, "RLHY", 100));
    if (!frontLine.empty())
    {
        TEST_ASSERT_FALSE(lineHasJointPwm(frontLine, "FLHY", 100));
    }
}

static void test_jog_relay_targets_right_joint_not_front()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbRolePrefixes();

    injectUsbCommand("JRRHY 200\n");

    std::string frontLine;
    std::string rightLine;
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("FRONT;", 0) == 0) frontLine = line;
        if (line.rfind("RIGHT;", 0) == 0) rightLine = line;
    }

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("JRRHY", leftRelay, rightRelay);

    TEST_ASSERT_TRUE(rightLine.find("RRHY") != std::string::npos);
    TEST_ASSERT_TRUE(lineHasJointPwm(rightLine, "RRHY", 100));
    if (!frontLine.empty())
    {
        TEST_ASSERT_FALSE(lineHasJointPwm(frontLine, "FRHY", 100));
    }
}

static void test_batch_jog_relay_reaches_both_followers()
{
    bootThreeBoardLeaderLoop();
    injectUsbCommand("B RLHY 180 RRHY 180\n");

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("RLHY", leftRelay, rightRelay);
    assertBothFollowerLinksRelayedSameLine("RRHY", leftRelay, rightRelay);

    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RLHY") >= 100);
    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RRHY") >= 100);
}

static void test_trajectory_relay_reaches_both_followers()
{
    bootThreeBoardLeaderLoop();
    injectUsbCommand("T RLHY 0.800 RRHY 0.800\n");

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("T", leftRelay, rightRelay);
    assertBothFollowerLinksRelayedSameLine("RLHY", leftRelay, rightRelay);
    assertBothFollowerLinksRelayedSameLine("RRHY", leftRelay, rightRelay);

    for (int i = 0; i < 20; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
    }
    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RLHY") > 0);
    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RRHY") > 0);
}

static void test_hold_relay_reaches_both_followers_and_coasts_joints()
{
    bootThreeBoardLeaderLoop();
    injectUsbCommand("B RLHY 200 RRHY 200\n");
    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RLHY") >= 100);
    TEST_ASSERT_TRUE(krabby_host_three_board_actuator_command_pwm("RRHY") >= 100);

    injectUsbCommand("H\n");

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("H", leftRelay, rightRelay);

    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_command_pwm("RLHY"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_command_pwm("RRHY"));
}

static void test_calibrate_relay_reaches_both_followers()
{
    bootThreeBoardLeaderLoop();
    injectUsbCommand("C\n");

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("C", leftRelay, rightRelay);

    int calStarts = 0;
    for (int i = 0; i < 10; ++i)
    {
        krabby_host_advance_millis(20);
        krabby_host_firmware_loop_once();
        SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
        while (usb.txLineAvailable())
        {
            const std::string line = usb.popTxLine();
            if (line.find("Starting Auto-Calibration") != std::string::npos) calStarts++;
        }
    }
    TEST_ASSERT_TRUE(calStarts >= 3);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_usb_telemetry_includes_three_roles_each_cycle);
    RUN_TEST(test_jog_relay_targets_left_joint_not_front);
    RUN_TEST(test_jog_relay_targets_right_joint_not_front);
    RUN_TEST(test_batch_jog_relay_reaches_both_followers);
    RUN_TEST(test_trajectory_relay_reaches_both_followers);
    RUN_TEST(test_hold_relay_reaches_both_followers_and_coasts_joints);
    RUN_TEST(test_calibrate_relay_reaches_both_followers);
    return UNITY_END();
}

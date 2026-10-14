#include "unity.h"

#include <cstdio>
#include <cstring>
#include <set>
#include <string>

#include "host_api.h"
#include "sim_serial.hpp"

using krabby::sim::SerialPortId;
using krabby::sim::SimSerialPort;
using krabby::sim::serialPort;

void setUp() { krabby_host_reset(); }

void tearDown() {}

static std::set<std::string> drainUsbRolePrefixes()
{
    std::set<std::string> roles;
    SimSerialPort &usb = serialPort(krabby::sim::BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("FRONT;", 0) == 0) roles.insert("FRONT");
        if (line.rfind("LEFT ;", 0) == 0) roles.insert("LEFT");
        if (line.rfind("RIGHT;", 0) == 0) roles.insert("RIGHT");
    }
    return roles;
}

static void bootThreeBoardLeaderLoop()
{
    int roles[3] = {0, 0, 0};
    krabby_host_run_three_board_role_election(roles);
    krabby_host_enable_three_board_firmware_loop(1);
    krabby_host_firmware_setup();
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

static void test_jog_relay_targets_left_joint_not_front()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbRolePrefixes();

    const char wire[] = "JRLHY 200\n";
    krabby_host_serial_usb_rx_write(
        reinterpret_cast<const uint8_t *>(wire),
        static_cast<int>(sizeof(wire) - 1));

    krabby_host_advance_millis(50);
    krabby_host_firmware_loop_once();

    std::string frontLine;
    std::string leftLine;
    SimSerialPort &usb = serialPort(krabby::sim::BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("FRONT;", 0) == 0) frontLine = line;
        if (line.rfind("LEFT ;", 0) == 0) leftLine = line;
    }

    TEST_ASSERT_TRUE(leftLine.find("RLHY") != std::string::npos);
    TEST_ASSERT_TRUE(lineHasJointPwm(leftLine, "RLHY", 100));
    if (!frontLine.empty())
    {
        TEST_ASSERT_FALSE(lineHasJointPwm(frontLine, "FLHY", 100));
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_usb_telemetry_includes_three_roles_each_cycle);
    RUN_TEST(test_jog_relay_targets_left_joint_not_front);
    return UNITY_END();
}

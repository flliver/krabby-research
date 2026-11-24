#include "unity.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "host_api.h"
#include "pin_state.hpp"
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

// Every board's sim actuators share one pin map: PWM_R, PWM_L, EN, POT per slot.
struct SimJointPins
{
    int pwmR;
    int pwmL;
    int en;
    int pot;
};
static const SimJointPins kSimJointPins[6] = {
    {2, 3, 22, 0}, {4, 5, 24, 1}, {6, 7, 26, 2},
    {8, 9, 23, 3}, {10, 11, 25, 4}, {12, 13, 27, 5},
};

// Each board's signed drive per slot, from the pin writes its own firmware makes
// (the boards share one pin map in the sim; PinStateModel tags each write).
struct BoardDrives
{
    int digital[3][krabby::sim::PinStateModel::kMaxMegaPin + 1];
    int duty[3][krabby::sim::PinStateModel::kMaxMegaPin + 1];

    int drive(int board, int joint) const
    {
        const SimJointPins &pins = kSimJointPins[joint];
        return digital[board][pins.en] ? duty[board][pins.pwmR] - duty[board][pins.pwmL] : 0;
    }
};
static BoardDrives g_drives;

// Records the drives; krabby_host_reset() drops the observer.
static void watchBoardDrives()
{
    std::memset(&g_drives, 0, sizeof(g_drives));
    krabby::sim::pinState().setWriteObserver([](const krabby::sim::PinWriteEvent &event) {
        if (event.board < 0 || event.board > 2 || event.pin < 0 ||
            event.pin > krabby::sim::PinStateModel::kMaxMegaPin)
            return;
        (event.isPwm ? g_drives.duty : g_drives.digital)[event.board][event.pin] = event.value;
    });
}

static std::vector<std::string> drainUsbCalibrationLines()
{
    std::vector<std::string> lines;
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("CAL ", 0) == 0) lines.push_back(line);
    }
    return lines;
}

static bool containsLine(const std::vector<std::string> &lines, const char *expected)
{
    return std::find(lines.begin(), lines.end(), std::string(expected)) != lines.end();
}

static void test_calibrate_starts_on_front_quiets_followers_and_hold_aborts()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbCalibrationLines();
    injectUsbCommand("C\n");

    const auto leftRelay = drainInterBoardRelayLines(SerialPortId::kSerial1);
    const auto rightRelay = drainInterBoardRelayLines(SerialPortId::kSerial2);
    assertBothFollowerLinksRelayedSameLine("H", leftRelay, rightRelay);
    const auto started = drainUsbCalibrationLines();
    TEST_ASSERT_TRUE(containsLine(started, "CAL ROBOT START pwm=128 passes=SEEK,SWEEP"));
    TEST_ASSERT_TRUE(containsLine(started, "CAL FRONT PASS SEEK pwm=128"));
    TEST_ASSERT_TRUE(containsLine(started, "CAL FRONT MOVE FLKL curl"));
    TEST_ASSERT_EQUAL_INT(1, krabby_host_firmware_calibration_active());

    injectUsbCommand("H\n");
    const auto aborted = drainUsbCalibrationLines();
    TEST_ASSERT_TRUE(containsLine(aborted, "CAL FRONT ABORTED H"));
    TEST_ASSERT_TRUE(containsLine(aborted, "CAL ROBOT ABORTED H"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_command_pwm("FLKL"));
}

static void test_robot_calibration_runs_boards_in_order_one_joint_at_a_time()
{
    // As on the robot, the pots read near the ADC rails at the stroke ends (no
    // band applies without quadrature halls). Each board has its own pots.
    constexpr int kLow = 3;
    constexpr int kHigh = 1020;
    int pot[3][6];
    for (int board = 0; board < 3; ++board)
        for (int joint = 0; joint < 6; ++joint)
        {
            pot[board][joint] = 500;
            krabby_host_set_board_analog_read(board, kSimJointPins[joint].pot, pot[board][joint], 0);
        }
    watchBoardDrives();
    bootThreeBoardLeaderLoop();
    (void)drainUsbCalibrationLines();
    krabby_host_serial_usb_rx_write(reinterpret_cast<const uint8_t *>("C 150\n"), 6);

    std::vector<std::string> log;
    for (int tick = 0; tick < 40000; ++tick)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
        for (int board = 0; board < 3; ++board)
            for (int joint = 0; joint < 6; ++joint)
            {
                const int drive = g_drives.drive(board, joint);
                int &reading = pot[board][joint];
                if (drive > 0) reading = std::min(kHigh, reading + 3);
                if (drive < 0) reading = std::max(kLow, reading - 3);
                krabby_host_set_board_analog_read(board, kSimJointPins[joint].pot, reading, 0);
            }
        const auto lines = drainUsbCalibrationLines();
        log.insert(log.end(), lines.begin(), lines.end());
        if (containsLine(lines, "CAL ROBOT DONE")) break;
    }

    std::string joined;
    for (const std::string &line : log) joined += line + "\n";
    TEST_ASSERT_TRUE_MESSAGE(containsLine(log, "CAL ROBOT DONE"), joined.c_str());
    static const char *kOrder[] = {
        "CAL ROBOT START pwm=150 passes=SEEK,SWEEP",
        "CAL FRONT PASS SEEK pwm=150",
        "CAL FRONT PASS_DONE SEEK",
        "CAL LEFT PASS SEEK pwm=150",
        "CAL LEFT PASS_DONE SEEK",
        "CAL RIGHT PASS SEEK pwm=150",
        "CAL RIGHT PASS_DONE SEEK",
        "CAL FRONT PASS SWEEP pwm=150",
        "CAL FRONT PASS_DONE SWEEP saved",
        "CAL LEFT PASS SWEEP pwm=150",
        "CAL LEFT PASS_DONE SWEEP saved",
        "CAL RIGHT PASS SWEEP pwm=150",
        "CAL RIGHT PASS_DONE SWEEP saved",
        "CAL ROBOT DONE",
    };
    auto cursor = log.begin();
    for (const char *expected : kOrder)
    {
        cursor = std::find(cursor, log.end(), std::string(expected));
        TEST_ASSERT_TRUE_MESSAGE(cursor != log.end(), expected);
    }
    TEST_ASSERT_TRUE(containsLine(log, "CAL LEFT MOVE RLKL curl"));
    TEST_ASSERT_TRUE(containsLine(log, "CAL RIGHT MOVE MRHL uncurl"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
    // Each board saved its own pot ends.
    TEST_ASSERT_EQUAL_INT(kLow, krabby_host_three_board_actuator_min_stop("RLHL"));
    TEST_ASSERT_EQUAL_INT(kHigh, krabby_host_three_board_actuator_max_stop("MRKL"));
}

// Fields of one joint's telemetry segment in a role's line: pos text, hall count, connection.
struct JointFields
{
    bool isFound = false;
    std::string pos;
    long hall = 0;
    int connection = -1;
};

static JointFields jointFields(const std::string &line, const char *name)
{
    JointFields fields;
    const std::size_t start = line.find(std::string(name) + " ");
    if (start == std::string::npos) return fields;
    char pos[32] = {0};
    fields.isFound = std::sscanf(line.c_str() + start, "%*s %31s %*d %*d %*d %*d %*d %*d %ld %d",
                                 pos, &fields.hall, &fields.connection) == 3;
    fields.pos = pos;
    return fields;
}

// Hall counts, the quadrature flag and SENS state belong to each board: only
// LEFT counts here, and only LEFT reports SENS and refuses T for lack of a position.
static void test_hall_counts_and_quadrature_stay_on_their_own_board()
{
    krabby_host_board_set_hall_quadrature(1, 1);  // LEFT only
    bootThreeBoardLeaderLoop();
    // The boards boot (hallHwInit zeroes each one's counts) on the first loop.
    krabby_host_advance_millis(10);
    krabby_host_firmware_loop_once();
    krabby_host_board_set_hall_edge_count(1, 1, -37);  // RLHL
    krabby_host_board_set_hall_edge_count(2, 2, 5);    // RRKL
    TEST_ASSERT_EQUAL_INT(0, krabby_host_board_hall_edge_count(0, 1));
    TEST_ASSERT_EQUAL_INT(-37, krabby_host_board_hall_edge_count(1, 1));

    std::string front;
    std::string left;
    std::string right;
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    for (int i = 0; i < 3; ++i)
    {
        krabby_host_advance_millis(50);
        krabby_host_firmware_loop_once();
        while (usb.txLineAvailable())
        {
            const std::string line = usb.popTxLine();
            if (line.rfind("FRONT;", 0) == 0) front = line;
            if (line.rfind("LEFT ;", 0) == 0) left = line;
            if (line.rfind("RIGHT;", 0) == 0) right = line;
        }
    }
    const JointFields flhl = jointFields(front, "FLHL");
    const JointFields rlhl = jointFields(left, "RLHL");
    const JointFields rrhl = jointFields(right, "RRHL");
    const JointFields rrkl = jointFields(right, "RRKL");
    TEST_ASSERT_TRUE(flhl.isFound && rlhl.isFound && rrhl.isFound && rrkl.isFound);
    TEST_ASSERT_EQUAL_INT(0, flhl.hall);
    TEST_ASSERT_EQUAL_INT(-37, rlhl.hall);
    TEST_ASSERT_EQUAL_INT(0, rrhl.hall);
    TEST_ASSERT_EQUAL_INT(5, rrkl.hall);
    // LEFT's joints are Unknown (no position); FRONT's and RIGHT's are pots.
    TEST_ASSERT_EQUAL_STRING("nan", rlhl.pos.c_str());
    TEST_ASSERT_EQUAL_INT(2, rlhl.connection);
    TEST_ASSERT_EQUAL_STRING("0.500", flhl.pos.c_str());
    TEST_ASSERT_EQUAL_STRING("0.500", rrkl.pos.c_str());
    TEST_ASSERT_EQUAL_INT('?', krabby_host_three_board_sensor_state("RLHL"));
    TEST_ASSERT_EQUAL_INT('?', krabby_host_three_board_sensor_state("MLKL"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_sensor_state("FLHL"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_sensor_state("RRHL"));

    // T reaches every board: FRONT holds FLHL, LEFT refuses RLHL on its own line.
    injectUsbCommand("T FLHL 0.500 RLHL 0.500\n");
    std::vector<std::string> lines;
    for (int i = 0; i < 3; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
        while (usb.txLineAvailable()) lines.push_back(usb.popTxLine());
    }
    TEST_ASSERT_TRUE(containsLine(lines, "ERR T refused RLHL (no position)"));
    TEST_ASSERT_EQUAL_INT(1, krabby_host_three_board_actuator_has_target("FLHL"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_has_target("RLHL"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_command_pwm("RLHL"));
}

static void pushToLeftFollower(const char *wire)
{
    serialPort(BoardId::kLeft, SerialPortId::kSerial1)
        .rxPush(reinterpret_cast<const uint8_t *>(wire), std::strlen(wire));
}

static std::vector<std::string> runLoops(int count)
{
    std::vector<std::string> lines;
    for (int i = 0; i < count; ++i)
    {
        krabby_host_advance_millis(10);
        krabby_host_firmware_loop_once();
        const auto more = drainUsbCalibrationLines();
        lines.insert(lines.end(), more.begin(), more.end());
    }
    return lines;
}

// A follower handles its own C lines through the sketch's dispatch: a bad one
// is answered with its usage, a second one while it runs with BUSY. With no
// robot run on the leader, that follower's MOVE is a pass outside its turn (as
// after a leader reset mid-run): the leader sends it H and it stops.
static void test_follower_calibration_outside_a_robot_run_is_held()
{
    bootThreeBoardLeaderLoop();
    (void)drainUsbCalibrationLines();

    pushToLeftFollower("C BOGUS\n");
    const auto rejected = runLoops(3);
    TEST_ASSERT_TRUE(containsLine(rejected, "CAL LEFT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150"));
    TEST_ASSERT_FALSE(containsLine(rejected, "CAL ROBOT FAIL LEFT: MOVE outside its turn"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());

    pushToLeftFollower("C SEEK 80\nC SEEK 80\n");
    const auto lines = runLoops(5);
    std::string joined;
    for (const std::string &line : lines) joined += line + "\n";
    static const char *kOrder[] = {
        "CAL LEFT PASS SEEK pwm=80",
        "CAL LEFT MOVE RLKL curl",
        "CAL ROBOT FAIL LEFT: MOVE outside its turn",  // as soon as the MOVE is forwarded
        "CAL LEFT ABORTED H",
    };
    auto cursor = lines.begin();
    for (const char *expected : kOrder)
    {
        cursor = std::find(cursor, lines.end(), std::string(expected));
        TEST_ASSERT_TRUE_MESSAGE(cursor != lines.end(), (std::string(expected) + "\n" + joined).c_str());
    }
    TEST_ASSERT_TRUE_MESSAGE(containsLine(lines, "CAL LEFT BUSY"), joined.c_str());
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
    TEST_ASSERT_EQUAL_INT(0, krabby_host_three_board_actuator_command_pwm("RLKL"));
    TEST_ASSERT_FALSE(containsLine(runLoops(300), "CAL LEFT MOVE RLHL curl"));
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
    RUN_TEST(test_calibrate_starts_on_front_quiets_followers_and_hold_aborts);
    RUN_TEST(test_robot_calibration_runs_boards_in_order_one_joint_at_a_time);
    RUN_TEST(test_follower_calibration_outside_a_robot_run_is_held);
    RUN_TEST(test_hall_counts_and_quadrature_stay_on_their_own_board);
    return UNITY_END();
}

// The calibration command glue arduino.ino runs (and the host sim shares):
// C dispatch by role, the H/J/B/T abort, and when V may query the followers.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "unity.h"

#include "src/actuator/calibration_command_dispatch.h"

namespace
{

struct FakeSink final : public CalibrationTextSink
{
    std::string text;

    void printFlash(const char *value) override { text += value; }
    void printText(const char *value) override { text += value; }
    void printNumber(long value) override { text += std::to_string(value); }
    void printChar(char value) override { text += value; }
    void endLine() override { text += '\n'; }
};

struct FakeHooks final : public RobotCalibrationHooks
{
    FakeSink usb;
    FakeSink left;
    FakeSink right;  // both followers linked: a SWEEP is refused with only one
    std::vector<std::pair<uint8_t, int>> localStarts;
    std::vector<char> localAborts;
    bool localRunning = false;

    uint32_t nowMs() override { return 0; }
    CalibrationTextSink &out() override { return usb; }
    const char *localRoleLabel() override { return "FRONT"; }
    CalibrationTextSink *followerLink(BoardRole role) override
    {
        if (role == ROLE_LEFT) return &left;
        return role == ROLE_RIGHT ? &right : nullptr;
    }
    bool startLocalPass(uint8_t pass, int pwm) override
    {
        localStarts.push_back(std::make_pair(pass, pwm));
        localRunning = true;
        return true;
    }
    bool isLocalRunning() override { return localRunning; }
    LegCalibrationOutcome localOutcome() override { return LegCalibrationOutcome::Running; }
    void printLocalFailure(CalibrationTextSink &) override {}
    void abortLocal(char cause) override
    {
        localAborts.push_back(cause);
        localRunning = false;
    }
    void stopLocalMotors() override {}
};

// Stands in for ActuatorManager.
struct FakeBoard
{
    bool calibrating = false;
    std::vector<std::pair<uint8_t, int>> starts;
    std::vector<char> aborts;

    bool isCalibrating() const { return calibrating; }
    bool startCalibration(uint8_t passes, int pwm)
    {
        starts.push_back(std::make_pair(passes, pwm));
        calibrating = true;
        return true;
    }
    bool abortCalibration(char cause)
    {
        aborts.push_back(cause);
        const bool wasCalibrating = calibrating;
        calibrating = false;
        return wasCalibrating;
    }
};

const ActuatorCalibrationRequest kSweep100 = {CAL_PASS_SWEEP, 100};

}  // namespace

// hall_hw.h, for RobotCalibrationHooks::hasQuadratureHall(): a leader without
// quadrature halls, which never asks its followers for SENS lines.
bool hallHwHasQuadrature() { return false; }

void setUp() {}
void tearDown() {}

static void test_leader_roles()
{
    TEST_ASSERT_TRUE(isCalibrationLeaderRole(ROLE_FRONT));
    TEST_ASSERT_TRUE(isCalibrationLeaderRole(ROLE_UNKNOWN));  // single-board bench
    TEST_ASSERT_FALSE(isCalibrationLeaderRole(ROLE_LEFT));
    TEST_ASSERT_FALSE(isCalibrationLeaderRole(ROLE_RIGHT));
}

static void test_leader_run_starts_the_robot_and_never_the_board_directly()
{
    for (const BoardRole role : {ROLE_FRONT, ROLE_UNKNOWN})
    {
        FakeHooks hooks;
        FakeBoard board;
        RobotCalibration robot;
        FakeSink out;
        dispatchActuatorCalibrationCommand(role, ActuatorCalibrationCommandKind::Run, kSweep100, out,
            robot, hooks, board);
        TEST_ASSERT_TRUE(robot.isActive());
        TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=100 passes=SWEEP\n", hooks.usb.text.c_str());
        TEST_ASSERT_EQUAL_STRING("H\n", hooks.left.text.c_str());
        TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
        TEST_ASSERT_TRUE(board.starts.empty());
        TEST_ASSERT_TRUE(out.text.empty());

        // A second C while it runs is the robot's BUSY.
        dispatchActuatorCalibrationCommand(role, ActuatorCalibrationCommandKind::Run, kSweep100, out,
            robot, hooks, board);
        TEST_ASSERT_EQUAL_STRING(
            "CAL ROBOT START pwm=100 passes=SWEEP\nCAL ROBOT BUSY\n", hooks.usb.text.c_str());
        TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
    }
}

static void test_follower_run_starts_its_own_passes_or_answers_busy()
{
    FakeHooks hooks;
    FakeBoard board;
    RobotCalibration robot;
    FakeSink out;
    dispatchActuatorCalibrationCommand(ROLE_LEFT, ActuatorCalibrationCommandKind::Run, kSweep100, out,
        robot, hooks, board);
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_UINT(1, board.starts.size());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SWEEP, board.starts[0].first);
    TEST_ASSERT_EQUAL_INT(100, board.starts[0].second);
    TEST_ASSERT_TRUE(out.text.empty());
    TEST_ASSERT_TRUE(hooks.usb.text.empty());
    TEST_ASSERT_TRUE(hooks.left.text.empty());  // followers never forward

    dispatchActuatorCalibrationCommand(ROLE_LEFT, ActuatorCalibrationCommandKind::Run, kSweep100, out,
        robot, hooks, board);
    TEST_ASSERT_EQUAL_STRING("CAL LEFT BUSY\n", out.text.c_str());
    TEST_ASSERT_EQUAL_UINT(1, board.starts.size());
}

static void test_invalid_command_reports_usage_and_moves_nothing()
{
    const std::pair<BoardRole, const char *> cases[] = {
        {ROLE_FRONT, "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150\n"},
        {ROLE_UNKNOWN, "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150\n"},
        {ROLE_RIGHT, "CAL RIGHT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150\n"},
    };
    for (const auto &entry : cases)
    {
        FakeHooks hooks;
        FakeBoard board;
        RobotCalibration robot;
        FakeSink out;
        dispatchActuatorCalibrationCommand(entry.first, ActuatorCalibrationCommandKind::Invalid, kSweep100,
            out, robot, hooks, board);
        TEST_ASSERT_EQUAL_STRING(entry.second, out.text.c_str());
        TEST_ASSERT_FALSE(robot.isActive());
        TEST_ASSERT_TRUE(board.starts.empty());
        TEST_ASSERT_TRUE(hooks.localStarts.empty());
        TEST_ASSERT_TRUE(hooks.left.text.empty());
    }
}

static void test_power_sensor_command_is_not_handled_here()
{
    for (const BoardRole role : {ROLE_FRONT, ROLE_LEFT})
    {
        FakeHooks hooks;
        FakeBoard board;
        RobotCalibration robot;
        FakeSink out;
        dispatchActuatorCalibrationCommand(role, ActuatorCalibrationCommandKind::PowerSensor, kSweep100,
            out, robot, hooks, board);
        TEST_ASSERT_TRUE(out.text.empty());
        TEST_ASSERT_TRUE(hooks.usb.text.empty());
        TEST_ASSERT_FALSE(robot.isActive());
        TEST_ASSERT_TRUE(board.starts.empty());
    }
}

static void test_abort_stops_the_robot_then_the_board()
{
    FakeHooks hooks;
    FakeBoard board;
    RobotCalibration robot;
    FakeSink out;
    // Idle: both are asked, nothing is printed.
    abortBoardCalibrationForCommand(robot, hooks, board, 'T');
    TEST_ASSERT_EQUAL_UINT(1, board.aborts.size());
    TEST_ASSERT_TRUE(hooks.usb.text.empty());

    dispatchActuatorCalibrationCommand(ROLE_FRONT, ActuatorCalibrationCommandKind::Run, kSweep100, out,
        robot, hooks, board);
    hooks.left.text.clear();
    abortBoardCalibrationForCommand(robot, hooks, board, 'B');
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_STRING("H\n", hooks.left.text.c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localAborts.size());
    TEST_ASSERT_EQUAL_INT('B', hooks.localAborts[0]);
    TEST_ASSERT_EQUAL_UINT(2, board.aborts.size());
    TEST_ASSERT_EQUAL_INT('B', board.aborts[1]);
    TEST_ASSERT_TRUE(hooks.usb.text.find("CAL ROBOT ABORTED B\n") != std::string::npos);

    // A follower: its robot sequence is idle, its own pass stops.
    FakeBoard follower;
    RobotCalibration followerRobot;
    follower.startCalibration(CAL_PASS_ALL, CAL_DEFAULT_PWM);
    abortBoardCalibrationForCommand(followerRobot, hooks, follower, 'H');
    TEST_ASSERT_FALSE(follower.isCalibrating());
    TEST_ASSERT_EQUAL_INT('H', follower.aborts[0]);
}

static void test_version_query_skips_followers_while_calibrating()
{
    FakeHooks hooks;
    FakeBoard board;
    RobotCalibration robot;
    FakeSink out;
    TEST_ASSERT_TRUE(shouldQueryFollowerVersions(robot));
    dispatchActuatorCalibrationCommand(ROLE_FRONT, ActuatorCalibrationCommandKind::Run, kSweep100, out,
        robot, hooks, board);
    TEST_ASSERT_FALSE(shouldQueryFollowerVersions(robot));
    robot.abort(hooks, 'H');
    TEST_ASSERT_TRUE(shouldQueryFollowerVersions(robot));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_leader_roles);
    RUN_TEST(test_leader_run_starts_the_robot_and_never_the_board_directly);
    RUN_TEST(test_follower_run_starts_its_own_passes_or_answers_busy);
    RUN_TEST(test_invalid_command_reports_usage_and_moves_nothing);
    RUN_TEST(test_power_sensor_command_is_not_handled_here);
    RUN_TEST(test_abort_stops_the_robot_then_the_board);
    RUN_TEST(test_version_query_skips_followers_while_calibrating);
    return UNITY_END();
}

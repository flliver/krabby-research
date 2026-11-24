#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "unity.h"

#include "src/actuator/robot_calibration.h"

namespace
{

// Shared record of every side effect, in order, across output, links and the leader.
std::vector<std::string> events;
// What hallHwHasQuadrature() reports for the leader (the hooks' default).
bool g_quadrature = false;

struct FakeSink final : public CalibrationTextSink
{
    FakeSink(const char *sinkName) : name(sinkName) {}

    const char *name;
    std::string text;

    void printFlash(const char *value) override { text += value; }
    void printText(const char *value) override { text += value; }
    void printNumber(long value) override { text += std::to_string(value); }
    void printChar(char value) override { text += value; }
    void endLine() override
    {
        const size_t start = text.rfind('\n', text.size() == 0 ? 0 : text.size() - 1);
        const std::string line = text.substr(start == std::string::npos ? 0 : start + 1);
        events.push_back(std::string(name) + ":" + line);
        text += '\n';
    }

    std::vector<std::string> lines() const
    {
        std::vector<std::string> result;
        std::istringstream stream(text);
        std::string line;
        while (std::getline(stream, line)) result.push_back(line);
        return result;
    }
    std::string last() const { return lines().empty() ? std::string() : lines().back(); }
};

struct FakeHooks final : public RobotCalibrationHooks
{
    uint32_t now = 0;
    FakeSink usb{"usb"};
    FakeSink left{"left"};
    FakeSink right{"right"};
    bool hasLeft = true;
    bool hasRight = true;
    const char *label = "FRONT";
    bool startResult = true;
    bool localRunning = false;
    LegCalibrationOutcome outcome = LegCalibrationOutcome::Idle;
    std::vector<std::pair<uint8_t, int>> localStarts;
    std::vector<char> aborts;
    int stops = 0;

    uint32_t nowMs() override { return now; }
    CalibrationTextSink &out() override { return usb; }
    const char *localRoleLabel() override { return label; }
    CalibrationTextSink *followerLink(BoardRole role) override
    {
        if (role == ROLE_LEFT) return hasLeft ? &left : nullptr;
        if (role == ROLE_RIGHT) return hasRight ? &right : nullptr;
        return nullptr;
    }
    bool startLocalPass(uint8_t pass, int pwm) override
    {
        localStarts.push_back(std::make_pair(pass, pwm));
        events.push_back("local:start");
        localRunning = startResult;
        outcome = startResult ? LegCalibrationOutcome::Running : outcome;
        return startResult;
    }
    bool isLocalRunning() override { return localRunning; }
    LegCalibrationOutcome localOutcome() override { return outcome; }
    void printLocalFailure(CalibrationTextSink &sink) override { sink.printText("FLHL pot not connected"); }
    void abortLocal(char cause) override
    {
        aborts.push_back(cause);
        events.push_back(std::string("local:abort ") + cause);
        localRunning = false;
    }
    void stopLocalMotors() override
    {
        ++stops;
        events.push_back("local:stop");
    }

    void finishLocal(LegCalibrationOutcome result)
    {
        localRunning = false;
        outcome = result;
    }
};

void completeLocal(RobotCalibration &robot, FakeHooks &hooks)
{
    hooks.finishLocal(LegCalibrationOutcome::Completed);
    robot.update(hooks);
}

const char *const kNoSensReason =
    "no SENS: flash all three boards with the same firmware, or wait a few seconds after connecting";

}  // namespace

// hall_hw.h, for RobotCalibrationHooks::hasQuadratureHall().
bool hallHwHasQuadrature() { return g_quadrature; }

void setUp()
{
    events.clear();
    g_quadrature = false;
}
void tearDown() {}

static void test_full_robot_run_orders_passes_and_boards()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, 100));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=100 passes=SEEK,SWEEP", hooks.usb.lines().front().c_str());
    TEST_ASSERT_EQUAL_STRING("H", hooks.left.last().c_str());
    TEST_ASSERT_EQUAL_STRING("H", hooks.right.last().c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SEEK, hooks.localStarts[0].first);
    TEST_ASSERT_EQUAL_INT(100, hooks.localStarts[0].second);
    TEST_ASSERT_EQUAL_INT(static_cast<int>(RobotCalibrationPhase::Local), static_cast<int>(robot.phase()));
    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, robot.currentBoard());
    TEST_ASSERT_EQUAL_INT(100, robot.pwm());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_ALL, robot.passes());

    robot.update(hooks);  // still running locally
    TEST_ASSERT_EQUAL_UINT(1, hooks.left.lines().size());
    hooks.now = 5000;
    completeLocal(robot, hooks);
    TEST_ASSERT_EQUAL_STRING("C SEEK 100", hooks.left.last().c_str());
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, robot.currentBoard());
    TEST_ASSERT_EQUAL_UINT32(5000, robot.lastFollowerLineMs());

    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS SEEK pwm=100");
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT PASS_DONE SEEK");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SWEEP saved");
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, robot.currentBoard());
    TEST_ASSERT_EQUAL_UINT(1, hooks.right.lines().size());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SEEK");
    TEST_ASSERT_EQUAL_STRING("C SEEK 100", hooks.right.last().c_str());
    TEST_ASSERT_EQUAL_INT(ROLE_RIGHT, robot.currentBoard());
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT PASS_DONE SEEK");

    TEST_ASSERT_EQUAL_UINT(2, hooks.localStarts.size());
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SWEEP, hooks.localStarts[1].first);
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_SWEEP, robot.currentPass());
    completeLocal(robot, hooks);
    TEST_ASSERT_EQUAL_STRING("C SWEEP 100", hooks.left.last().c_str());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SEEK");
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, robot.currentBoard());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SWEEP saved");
    TEST_ASSERT_EQUAL_STRING("C SWEEP 100", hooks.right.last().c_str());
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT PASS_DONE SWEEP saved");

    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT DONE", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_UINT(2, hooks.usb.lines().size());
    TEST_ASSERT_EQUAL_INT(0, hooks.stops);
    TEST_ASSERT_FALSE(robot.abort(hooks, 'H'));
    robot.update(hooks);
    TEST_ASSERT_EQUAL_UINT(2, hooks.usb.lines().size());
}

static void test_missing_followers_are_skipped()
{
    FakeHooks hooks;
    hooks.hasLeft = false;
    hooks.label = "UNKWN";
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=128 passes=SEEK", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_STRING("H", hooks.right.last().c_str());
    completeLocal(robot, hooks);
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT SKIP LEFT no link", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_STRING("C SEEK 128", hooks.right.last().c_str());
    TEST_ASSERT_TRUE(hooks.left.text.empty());
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT PASS_DONE SEEK");
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT DONE", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());

    FakeHooks alone;
    alone.hasLeft = false;
    alone.hasRight = false;
    RobotCalibration bench;
    TEST_ASSERT_TRUE(bench.start(alone, CAL_PASS_SWEEP, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=128 passes=SWEEP", alone.usb.last().c_str());
    completeLocal(bench, alone);
    const std::vector<std::string> lines = alone.usb.lines();
    TEST_ASSERT_EQUAL_UINT(4, lines.size());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT SKIP LEFT no link", lines[1].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT SKIP RIGHT no link", lines[2].c_str());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT DONE", lines[3].c_str());
    TEST_ASSERT_FALSE(bench.isActive());
}

// One follower linked and the other not: the robot is assembled but that
// board missed the election, so its legs were never curled. The SEEK still
// runs (curls only), but no SWEEP starts; a single board with no links (the
// bench) still sweeps.
static void test_sweep_refused_while_one_follower_has_no_link()
{
    FakeHooks hooks;
    hooks.hasLeft = false;
    hooks.label = "UNKWN";
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    completeLocal(robot, hooks);
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT SKIP LEFT no link", hooks.usb.last().c_str());
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT PASS_DONE SEEK");
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL LEFT: no link, no SWEEP until every board is linked",
        hooks.usb.last().c_str());
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
    TEST_ASSERT_EQUAL_STRING("H", hooks.right.last().c_str());
    TEST_ASSERT_EQUAL_INT(1, hooks.stops);

    FakeHooks sweepOnly;
    sweepOnly.hasRight = false;
    RobotCalibration other;
    TEST_ASSERT_TRUE(other.start(sweepOnly, CAL_PASS_SWEEP, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL RIGHT: no link, no SWEEP until every board is linked",
        sweepOnly.usb.last().c_str());
    TEST_ASSERT_TRUE(sweepOnly.localStarts.empty());
    TEST_ASSERT_FALSE(other.isActive());
}

static void test_busy_and_invalid_requests_are_rejected()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_FALSE(robot.start(hooks, CAL_PASS_ALL, CAL_PWM_MAX + 1));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT ERROR invalid request", hooks.usb.last().c_str());
    TEST_ASSERT_FALSE(robot.start(hooks, 0, CAL_DEFAULT_PWM));
    TEST_ASSERT_TRUE(hooks.localStarts.empty());
    TEST_ASSERT_TRUE(hooks.left.text.empty());

    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_FALSE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT BUSY", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
    TEST_ASSERT_TRUE(robot.isActive());
}

static void test_local_failure_fails_the_robot()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    events.clear();
    hooks.finishLocal(LegCalibrationOutcome::Failed);
    robot.update(hooks);
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL FRONT: FLHL pot not connected", hooks.usb.last().c_str());
    const std::vector<std::string> expected = {"left:H", "right:H", "local:stop",
        "usb:CAL ROBOT FAIL FRONT: FLHL pot not connected"};
    TEST_ASSERT_EQUAL_UINT(expected.size(), events.size());
    for (size_t i = 0; i < expected.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());

    FakeHooks refused;
    refused.startResult = false;
    RobotCalibration other;
    TEST_ASSERT_TRUE(other.start(refused, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_FALSE(other.isActive());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL FRONT: could not start its pass", refused.usb.last().c_str());
}

static void test_follower_failure_lines_fail_the_robot()
{
    const char *const lines[] = {
        "CAL LEFT FAIL RLHL barely moved (4 counts) at PWM 80: check pot wiring or raise speed",
        "CAL LEFT ABORTED H",
        "CAL LEFT BUSY",
        "CAL LEFT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150",
    };
    const char *const expected[] = {
        "CAL ROBOT FAIL LEFT: RLHL barely moved (4 counts) at PWM 80: check pot wiring or raise speed",
        "CAL ROBOT FAIL LEFT: ABORTED H",
        "CAL ROBOT FAIL LEFT: BUSY",
        "CAL ROBOT FAIL LEFT: ERROR usage: C [SEEK|SWEEP] [pwm], pwm 40-150",
    };
    for (size_t i = 0; i < 4; ++i)
    {
        FakeHooks hooks;
        RobotCalibration robot;
        TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
        completeLocal(robot, hooks);
        robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT MOVE RLKL curl");
        TEST_ASSERT_TRUE(robot.isActive());
        events.clear();
        robot.onFollowerLine(hooks, ROLE_LEFT, lines[i]);
        TEST_ASSERT_FALSE(robot.isActive());
        TEST_ASSERT_EQUAL_STRING(expected[i], hooks.usb.last().c_str());
        TEST_ASSERT_EQUAL_STRING("left:H", events[0].c_str());
        TEST_ASSERT_EQUAL_STRING("right:H", events[1].c_str());
        TEST_ASSERT_EQUAL_STRING("local:stop", events[2].c_str());
        TEST_ASSERT_EQUAL_INT(1, hooks.stops);
    }
}

static void test_follower_lines_that_do_not_matter_are_ignored()
{
    FakeHooks hooks;
    RobotCalibration robot;
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SEEK");  // idle
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT FAIL x");  // leader still running its own pass
    TEST_ASSERT_TRUE(robot.isActive());
    completeLocal(robot, hooks);
    hooks.now = 1000;
    robot.onFollowerLine(hooks, ROLE_LEFT, nullptr);
    robot.onFollowerLine(hooks, ROLE_LEFT, "LEFT ;RLHY 0.500 500 0 0 0 0 0 0 1");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL RIGHT FAIL x");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFTX FAIL x");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT");
    TEST_ASSERT_EQUAL_UINT32(0, robot.lastFollowerLineMs());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT WARN RLKL text");
    TEST_ASSERT_EQUAL_UINT32(1000, robot.lastFollowerLineMs());
    TEST_ASSERT_TRUE(robot.isActive());
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, robot.currentBoard());
}

static void test_silent_follower_trips_the_watchdog()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    hooks.now = 10;
    completeLocal(robot, hooks);
    hooks.now = 10 + CAL_FOLLOWER_SILENCE_MS - 1;
    robot.update(hooks);
    TEST_ASSERT_TRUE(robot.isActive());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT END RLKL curl pot=900 travel=3");
    hooks.now += CAL_FOLLOWER_SILENCE_MS - 1;
    robot.update(hooks);
    TEST_ASSERT_TRUE(robot.isActive());
    events.clear();
    hooks.now += 1;
    robot.update(hooks);
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL LEFT: no CAL report for 135 s", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_STRING("left:H", events[0].c_str());
    TEST_ASSERT_EQUAL_STRING("right:H", events[1].c_str());
}

static void test_abort_sends_hold_to_followers_first()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    events.clear();
    TEST_ASSERT_TRUE(robot.abort(hooks, 'J'));
    const std::vector<std::string> expected = {"left:H", "right:H", "local:abort J", "local:stop",
        "usb:CAL ROBOT ABORTED J"};
    TEST_ASSERT_EQUAL_UINT(expected.size(), events.size());
    for (size_t i = 0; i < expected.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
    TEST_ASSERT_FALSE(robot.isActive());
    TEST_ASSERT_FALSE(robot.abort(hooks, 'H'));

    // While a follower runs, too; its later lines no longer matter.
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    completeLocal(robot, hooks);
    TEST_ASSERT_TRUE(robot.abort(hooks, 'T'));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT ABORTED T", hooks.usb.last().c_str());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT ABORTED H");
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT ABORTED T", hooks.usb.last().c_str());

    robot.reset();
    TEST_ASSERT_EQUAL_INT(static_cast<int>(RobotCalibrationPhase::Idle), static_cast<int>(robot.phase()));
    TEST_ASSERT_EQUAL_INT(0, robot.pwm());
}

static void test_prefix_helpers()
{
    TEST_ASSERT_EQUAL_STRING("x", calibrationTextAfterPrefix("CALx", "CAL"));
    TEST_ASSERT_NULL(calibrationTextAfterPrefix("CA", "CAL"));
    TEST_ASSERT_EQUAL_STRING("", calibrationTextAfterFlashPrefix("CAL", "CAL"));
    TEST_ASSERT_NULL(calibrationTextAfterFlashPrefix("CBL", "CAL"));
    TEST_ASSERT_EQUAL_STRING("PASS_DONE SEEK", calibrationLineEvent("CAL RIGHT PASS_DONE SEEK", "RIGHT"));
    TEST_ASSERT_NULL(calibrationLineEvent("CAL RIGHT", "RIGHT"));
    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, robotCalibrationBoardRole(0));
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, robotCalibrationBoardRole(1));
    TEST_ASSERT_EQUAL_INT(ROLE_RIGHT, robotCalibrationBoardRole(2));
}

// Review finding: a follower reason is echoed into one "CAL ROBOT FAIL" line;
// a corrupted (or very long) follower line is cut so that line stays ~100 chars.
static void test_echoed_follower_reason_is_capped()
{
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    completeLocal(robot, hooks);
    completeLocal(robot, hooks);  // no-op: waiting on LEFT
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT PASS_DONE SEEK");
    const std::string reason(200, 'x');
    robot.onFollowerLine(hooks, ROLE_RIGHT, ("CAL RIGHT FAIL " + reason).c_str());
    TEST_ASSERT_FALSE(robot.isActive());
    const std::string expected = "CAL ROBOT FAIL RIGHT: " + std::string(CAL_ROBOT_REASON_MAX_CHARS, 'x');
    TEST_ASSERT_EQUAL_STRING(expected.c_str(), hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_UINT(100, expected.size());

    // The longest reason the firmware prints passes through whole.
    FakeHooks other;
    RobotCalibration second;
    TEST_ASSERT_TRUE(second.start(other, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    completeLocal(second, other);
    const char *longest = "MRKL barely moved (1023 counts) at PWM 150: check pot wiring or raise speed";
    TEST_ASSERT_TRUE(std::string(longest).size() <= CAL_ROBOT_REASON_MAX_CHARS);
    second.onFollowerLine(other, ROLE_LEFT, (std::string("CAL LEFT FAIL ") + longest).c_str());
    TEST_ASSERT_EQUAL_STRING((std::string("CAL ROBOT FAIL LEFT: ") + longest).c_str(), other.usb.last().c_str());
}

// Review finding: a follower that keeps running a pass the leader no longer
// supervises (the leader reset mid-run, so it is idle) is sent H as soon as it
// reports a move; nothing else is stopped.
static void test_follower_moving_while_the_leader_is_idle_is_held()
{
    FakeHooks hooks;
    RobotCalibration robot;
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT END RLKL curl pot=900 travel=0");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT MOVEX");
    TEST_ASSERT_TRUE(events.empty());
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT MOVE RLHL curl");
    const std::vector<std::string> expected = {"left:H", "usb:CAL ROBOT FAIL LEFT: MOVE outside its turn"};
    TEST_ASSERT_EQUAL_UINT(expected.size(), events.size());
    for (size_t i = 0; i < expected.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
    TEST_ASSERT_EQUAL_INT(0, hooks.stops);
    TEST_ASSERT_FALSE(robot.isActive());

    // A link that is gone: only the report.
    hooks.hasRight = false;
    events.clear();
    robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT MOVE RRKL uncurl");
    TEST_ASSERT_EQUAL_UINT(1, events.size());
    TEST_ASSERT_EQUAL_STRING("usb:CAL ROBOT FAIL RIGHT: MOVE outside its turn", events[0].c_str());
}

// A follower moving while the leader runs its own pass, or while it waits on
// the other follower, fails the robot: H to every follower, leader stopped.
static void test_follower_moving_outside_its_turn_fails_the_robot()
{
    {
        FakeHooks hooks;
        RobotCalibration robot;
        TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));  // Local
        events.clear();
        robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT MOVE RRHL curl");
        TEST_ASSERT_FALSE(robot.isActive());
        const std::vector<std::string> expected = {"left:H", "right:H", "local:stop",
            "usb:CAL ROBOT FAIL RIGHT: MOVE outside its turn"};
        TEST_ASSERT_EQUAL_UINT(expected.size(), events.size());
        for (size_t i = 0; i < expected.size(); ++i)
            TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
    }
    {
        FakeHooks hooks;
        RobotCalibration robot;
        TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
        completeLocal(robot, hooks);  // waiting on LEFT
        robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT MOVE RLKL curl");
        robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT END RRKL curl pot=900 travel=0");
        TEST_ASSERT_TRUE(robot.isActive());
        robot.onFollowerLine(hooks, ROLE_RIGHT, "CAL RIGHT MOVE RRKL curl");
        TEST_ASSERT_FALSE(robot.isActive());
        TEST_ASSERT_EQUAL_STRING("CAL ROBOT FAIL RIGHT: MOVE outside its turn", hooks.usb.last().c_str());
        TEST_ASSERT_EQUAL_STRING("H", hooks.left.last().c_str());
        TEST_ASSERT_EQUAL_STRING("H", hooks.right.last().c_str());
        TEST_ASSERT_EQUAL_INT(1, hooks.stops);
    }
}

// Review finding R1-8: on a quadrature leader the robot calibration never
// starts while a linked follower has sent no SENS line since the leader booted
// (an older build, a build without quadrature halls, or a board just
// connected): nothing moves, H goes to every follower and the leader stops.
static void test_quadrature_leader_refuses_followers_without_sens_lines()
{
    g_quadrature = true;
    FakeHooks hooks;
    RobotCalibration robot;
    TEST_ASSERT_FALSE(robot.hasSensLineFrom(ROLE_LEFT));
    TEST_ASSERT_FALSE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_FALSE(robot.isActive());
    const std::vector<std::string> expected = {"left:H", "right:H", "local:stop",
        std::string("usb:CAL ROBOT FAIL LEFT: ") + kNoSensReason};
    TEST_ASSERT_EQUAL_UINT(expected.size(), events.size());
    for (size_t i = 0; i < expected.size(); ++i)
        TEST_ASSERT_EQUAL_STRING(expected[i].c_str(), events[i].c_str());
    TEST_ASSERT_TRUE(hooks.localStarts.empty());
    TEST_ASSERT_EQUAL_UINT(1, hooks.usb.lines().size());  // no START

    // Only "SENS <this follower's role> " counts.
    robot.onFollowerLine(hooks, ROLE_LEFT, "SENS RIGHT RRHY:? RRHL:e RRKL:e MRHY:? MRHL:e MRKL:e");
    robot.onFollowerLine(hooks, ROLE_LEFT, "SENS LEFTX RLHY:?");
    robot.onFollowerLine(hooks, ROLE_LEFT, "SENS LEFT");
    robot.onFollowerLine(hooks, ROLE_LEFT, "LEFT ;RLHY 0.500 500 0 0 0 0 0 0 1");
    robot.onFollowerLine(hooks, ROLE_LEFT, "CAL LEFT SENSOR RLKL hall");
    TEST_ASSERT_FALSE(robot.hasSensLineFrom(ROLE_LEFT));
    TEST_ASSERT_FALSE(robot.hasSensLineFrom(ROLE_RIGHT));
    robot.onFollowerLine(hooks, ROLE_LEFT, "SENS LEFT RLHY:? RLHL:e RLKL:e MLHY:? MLHL:e MLKL:e");
    TEST_ASSERT_TRUE(robot.hasSensLineFrom(ROLE_LEFT));
    TEST_ASSERT_FALSE(robot.start(hooks, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING((std::string("CAL ROBOT FAIL RIGHT: ") + kNoSensReason).c_str(),
        hooks.usb.last().c_str());
    TEST_ASSERT_TRUE(hooks.localStarts.empty());

    robot.onFollowerLine(hooks, ROLE_RIGHT, "SENS RIGHT RRHY:? RRHL:e RRKL:e MRHY:? MRHL:e MRKL:e");
    TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING("CAL ROBOT START pwm=128 passes=SEEK,SWEEP", hooks.usb.last().c_str());
    TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());

    // A reset is a boot: the SENS lines are forgotten.
    TEST_ASSERT_TRUE(robot.abort(hooks, 'H'));
    robot.reset();
    TEST_ASSERT_FALSE(robot.hasSensLineFrom(ROLE_LEFT));
    TEST_ASSERT_FALSE(robot.hasSensLineFrom(ROLE_RIGHT));
    TEST_ASSERT_FALSE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
    TEST_ASSERT_EQUAL_STRING((std::string("CAL ROBOT FAIL LEFT: ") + kNoSensReason).c_str(),
        hooks.usb.last().c_str());
}

// The SENS check applies to linked followers of a quadrature leader only: a
// leader without quadrature halls (its own build prints no SENS lines) and a
// single board on the bench start as before; with one follower linked, only
// that one is checked.
static void test_sens_check_applies_to_linked_followers_of_a_quadrature_leader()
{
    {
        FakeHooks hooks;  // g_quadrature false
        RobotCalibration robot;
        TEST_ASSERT_TRUE(robot.start(hooks, CAL_PASS_ALL, CAL_DEFAULT_PWM));
        TEST_ASSERT_EQUAL_UINT(1, hooks.localStarts.size());
    }
    g_quadrature = true;
    {
        FakeHooks bench;
        bench.hasLeft = false;
        bench.hasRight = false;
        bench.label = "UNKWN";
        RobotCalibration robot;
        TEST_ASSERT_TRUE(robot.start(bench, CAL_PASS_ALL, CAL_DEFAULT_PWM));
        TEST_ASSERT_EQUAL_UINT(1, bench.localStarts.size());
    }
    {
        FakeHooks one;
        one.hasLeft = false;
        one.label = "UNKWN";
        RobotCalibration robot;
        TEST_ASSERT_FALSE(robot.start(one, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
        TEST_ASSERT_EQUAL_STRING((std::string("CAL ROBOT FAIL RIGHT: ") + kNoSensReason).c_str(),
            one.usb.last().c_str());
        robot.onFollowerLine(one, ROLE_RIGHT, "SENS RIGHT RRHY:? RRHL:? RRKL:? MRHY:? MRHL:? MRKL:?");
        TEST_ASSERT_TRUE(robot.start(one, CAL_PASS_SEEK, CAL_DEFAULT_PWM));
        TEST_ASSERT_EQUAL_UINT(1, one.localStarts.size());
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_full_robot_run_orders_passes_and_boards);
    RUN_TEST(test_missing_followers_are_skipped);
    RUN_TEST(test_sweep_refused_while_one_follower_has_no_link);
    RUN_TEST(test_busy_and_invalid_requests_are_rejected);
    RUN_TEST(test_local_failure_fails_the_robot);
    RUN_TEST(test_follower_failure_lines_fail_the_robot);
    RUN_TEST(test_follower_lines_that_do_not_matter_are_ignored);
    RUN_TEST(test_silent_follower_trips_the_watchdog);
    RUN_TEST(test_abort_sends_hold_to_followers_first);
    RUN_TEST(test_prefix_helpers);
    RUN_TEST(test_echoed_follower_reason_is_capped);
    RUN_TEST(test_follower_moving_while_the_leader_is_idle_is_held);
    RUN_TEST(test_follower_moving_outside_its_turn_fails_the_robot);
    RUN_TEST(test_quadrature_leader_refuses_followers_without_sens_lines);
    RUN_TEST(test_sens_check_applies_to_linked_followers_of_a_quadrature_leader);
    return UNITY_END();
}

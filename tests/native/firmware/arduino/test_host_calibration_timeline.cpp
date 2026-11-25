// Actuator calibration on the host sim, single board and three boards, with
// every drive recorded and the safety invariants checked at each pin write
// (host_calibration_timeline.hpp): one actuator at a time across the robot,
// yaws never driven, |PWM| ramped from 0 and never above the request (a nudge
// never above 128), knees uncurled only with their hip curled (with
// CAL_OPT_KNEE_NUDGE, the shipped default used by every C command run, a
// bounded knee nudge followed by its curl; strict runs pass options 0), hall
// hips inside the table-safe envelope (above 560 edges only with the knee
// parked at -264, never past the 840-edge sweep plus coast, never at the extend
// switch), SEEK on every board before any SWEEP, stop commands end everything,
// and EEPROM is written only when a board's SWEEP completes. Rods report
// through pots, quadrature hall encoders (signed counts per board, hall B on
// the pot pin) or nothing.

#include "unity.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "host_api.h"
#include "host_calibration_timeline.hpp"
#include "host_firmware_internal.hpp"
#include "sim_serial.hpp"
#include "src/controller/board_role.h"

using calibration_timeline::ExpectedMove;
using calibration_timeline::Move;
using calibration_timeline::Rod;
using calibration_timeline::RodSensor;
using calibration_timeline::Timeline;
using calibration_timeline::appendNudged;
using calibration_timeline::appendSeek;
using calibration_timeline::appendSeekLeg;
using calibration_timeline::appendSweep;
using calibration_timeline::appendSweepLeg;
using calibration_timeline::compareMoves;
using calibration_timeline::kBoards;
using calibration_timeline::kCalDataBytes;
using calibration_timeline::kJointNames;
using calibration_timeline::kJoints;
using calibration_timeline::kSpecFollowerSilenceMs;
using calibration_timeline::kSpecHipCurlSign;
using calibration_timeline::kSpecHipNudgeMaxMs;
using calibration_timeline::kSpecKneeCurlSign;
using calibration_timeline::kSpecHipNudgeBlindMs;
using calibration_timeline::kSpecHipStrokeEdges;
using calibration_timeline::kSpecHipSweepCoastEdges;
using calibration_timeline::kSpecHipSweepEdges;
using calibration_timeline::kSpecKneeParkBandMaxEdges;
using calibration_timeline::kSpecKneeParkBandMinEdges;
using calibration_timeline::kSpecNudgeMaxPwm;
using calibration_timeline::kSpecKneeNudgeMaxEdges;
using calibration_timeline::kSpecBlindKneeNudgeMaxEdges;
using calibration_timeline::kSpecKneeNudgeMaxMs;
using calibration_timeline::kSpecKneeSpringMaxEdges;
using calibration_timeline::kSpecMoveTimeoutMs;
using calibration_timeline::kSpecSettleMs;
using calibration_timeline::makeRobotLikeHip;
using calibration_timeline::makeRobotLikeKnee;
using krabby::sim::BoardId;
using krabby::sim::SerialPortId;
using krabby::sim::SimSerialPort;
using krabby::sim::serialPort;

namespace
{

// One firmware loop per tick. The robot's leader loop runs about every 3 ms
// (2026-11-24 design spec: its IS filter, alpha 0.1 per loop, settles in about
// 30 ms); at 10 ms the filter would lag about three times as long and the end
// detector's floor, taken 200-250 ms into a drive, would read about 10-15%
// low, which no robot trace shows. 5 ms keeps the filter within 5% by then and
// divides the 20 ms ramp step and the 50 ms detector sample.
constexpr uint32_t kTickMs = 5;
// The record's magic and check, written out (actuator_manager.h CalData v2).
constexpr uint16_t kSpecCalMagic = 0xCA2B;
constexpr uint16_t kSpecCalCheckXor = 0x5A5A;
// Sensor bits (2 per joint, 1 = Pot, 2 = Hall, 0 = Unknown): every joint Pot;
// hips and knees Hall with Unknown yaws; leg A Hall and leg B Pot.
constexpr uint16_t kAllPotBits = 0x0555;
constexpr uint16_t kHallHipsAndKneesBits = 0x0A28;
constexpr uint16_t kPotHipsAndKneesBits = 0x0514;
constexpr uint16_t kHallLegAPotLegBBits = 0x0528;
// Encoder T stops once Kp (1 PWM per edge) times the error falls below the
// 20-PWM drive deadband, so it settles within 20 edges of the target (the
// 12-edge error deadband is inside that). Spec numbers, as is the hip nudge bound.
constexpr int kSpecEncoderSettleEdges = 20;
constexpr double kSpecHipNudgeEdges = 24.0;

struct LoggedLine
{
    uint32_t ms;
    std::string text;
};

std::vector<LoggedLine> g_log;    // CAL lines
std::vector<LoggedLine> g_other;  // SENS, ERR and the EEPROM messages
bool g_dropLeftCommands = false;

// The CalData v2 record as the Mega stores it (int16 fields, 16-bit magic,
// sensor bits and check).
struct HostCalData
{
    int16_t minVals[6];
    int16_t maxVals[6];
    uint16_t magic;
    uint16_t sensorBits;
    uint16_t check;
};
static_assert(sizeof(HostCalData) == kCalDataBytes, "host CalData layout");

// (sum of bytes 0..27 as uint16) ^ 0x5A5A.
uint16_t calDataCheck(const HostCalData &data)
{
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(&data);
    uint16_t sum = 0;
    for (size_t i = 0; i < offsetof(HostCalData, check); ++i) sum = static_cast<uint16_t>(sum + bytes[i]);
    return static_cast<uint16_t>(sum ^ kSpecCalCheckXor);
}

HostCalData readCalData(int board)
{
    HostCalData data;
    std::memcpy(&data, EEPROM.boards[board].bytes, sizeof(data));
    return data;
}

// A saved v2 record: hall hips [0, hipStroke] and knees [-kneeStroke, 0], yaws
// Unknown (a hip record loads only with its stroke within 20 edges of 1069).
void writeHallRecord(int board, int hipStroke, int kneeStroke)
{
    HostCalData data = {};
    for (int joint = 0; joint < kJoints; ++joint)
    {
        if (calibration_timeline::isHip(joint)) data.maxVals[joint] = static_cast<int16_t>(hipStroke);
        if (calibration_timeline::isKnee(joint)) data.minVals[joint] = static_cast<int16_t>(-kneeStroke);
    }
    data.magic = kSpecCalMagic;
    data.sensorBits = kHallHipsAndKneesBits;
    data.check = calDataCheck(data);
    std::memcpy(EEPROM.boards[board].bytes, &data, sizeof(data));
}

// A saved v2 record of pots: every joint 0/1023 except `joint`, which gets lo/hi.
void writePotRecord(int board, int joint, int lo, int hi)
{
    HostCalData data = {};
    for (int j = 0; j < kJoints; ++j)
    {
        data.minVals[j] = static_cast<int16_t>(j == joint ? lo : 0);
        data.maxVals[j] = static_cast<int16_t>(j == joint ? hi : 1023);
    }
    data.magic = kSpecCalMagic;
    data.sensorBits = kAllPotBits;
    data.check = calDataCheck(data);
    std::memcpy(EEPROM.boards[board].bytes, &data, sizeof(data));
}

bool isTelemetryLine(const std::string &line)
{
    return line.rfind("FRONT;", 0) == 0 || line.rfind("LEFT ;", 0) == 0 || line.rfind("RIGHT;", 0) == 0;
}

void drainUsb()
{
    SimSerialPort &usb = serialPort(BoardId::kFront, SerialPortId::kUsb);
    while (usb.txLineAvailable())
    {
        const std::string line = usb.popTxLine();
        if (line.rfind("CAL ", 0) == 0)
            g_log.push_back(LoggedLine{Timeline::nowMs(), line});
        else if (!isTelemetryLine(line))
            g_other.push_back(LoggedLine{Timeline::nowMs(), line});
    }
}

const LoggedLine *findLine(const std::string &text)
{
    for (const LoggedLine &line : g_log)
        if (line.text == text) return &line;
    return nullptr;
}

bool hasLine(const std::string &text) { return findLine(text) != nullptr; }

int countLines(const std::string &text)
{
    int count = 0;
    for (const LoggedLine &line : g_log) count += line.text == text;
    return count;
}

int countLinesStarting(const std::string &prefix)
{
    int count = 0;
    for (const LoggedLine &line : g_log) count += line.text.rfind(prefix, 0) == 0;
    return count;
}

bool hasOtherLine(const std::string &text)
{
    for (const LoggedLine &line : g_other)
        if (line.text == text) return true;
    return false;
}

int countOtherLines(const std::string &text)
{
    int count = 0;
    for (const LoggedLine &line : g_other) count += line.text == text;
    return count;
}

// The last SENS line a board printed ("SENS <ROLE> ..."), or "".
std::string lastSensLine(const char *role)
{
    const std::string prefix = std::string("SENS ") + role + " ";
    std::string last;
    for (const LoggedLine &line : g_other)
        if (line.text.rfind(prefix, 0) == 0) last = line.text;
    return last;
}

std::string joinedLog()
{
    std::string text;
    for (const LoggedLine &line : g_log) text += line.text + "\n";
    return text;
}

std::string joinedOther()
{
    std::string text;
    for (const LoggedLine &line : g_other) text += line.text + "\n";
    return text;
}

// The count printed by the last "CAL <role> HOME <name> count=<n>" line.
bool lastHomeCount(const char *role, const char *name, long &count)
{
    const std::string prefix = std::string("CAL ") + role + " HOME " + name + " count=";
    bool isFound = false;
    for (const LoggedLine &line : g_log)
    {
        if (line.text.rfind(prefix, 0) != 0) continue;
        count = std::strtol(line.text.c_str() + prefix.size(), nullptr, 10);
        isFound = true;
    }
    return isFound;
}

void writeUsb(const char *wire)
{
    krabby_host_serial_usb_rx_write(
        reinterpret_cast<const uint8_t *>(wire),
        static_cast<int>(std::strlen(wire)));
}

void tick(Timeline &timeline)
{
    if (g_dropLeftCommands)
    {
        // LEFT never hears the leader: its uplink receive buffer is discarded.
        SimSerialPort &leftUplink = serialPort(BoardId::kLeft, SerialPortId::kSerial1);
        while (leftUplink.rxAvailable() > 0) leftUplink.rxRead();
    }
    krabby_host_advance_millis(kTickMs);
    krabby_host_firmware_loop_once();
    timeline.advancePlant(kTickMs);
    timeline.sampleEeprom();
    drainUsb();
}

void runFor(Timeline &timeline, uint32_t ms)
{
    for (uint32_t elapsed = 0; elapsed < ms; elapsed += kTickMs) tick(timeline);
}

bool runUntil(Timeline &timeline, const char *line, uint32_t maxMs)
{
    for (uint32_t elapsed = 0; elapsed < maxMs; elapsed += kTickMs)
    {
        if (hasLine(line)) return true;
        tick(timeline);
    }
    return hasLine(line);
}

void assertNoViolations(const Timeline &timeline)
{
    TEST_ASSERT_TRUE_MESSAGE(timeline.violations.empty(), timeline.report().c_str());
}

void assertRunUntil(Timeline &timeline, const char *line, uint32_t maxMs)
{
    const bool isFound = runUntil(timeline, line, maxMs);
    const std::string message = std::string("missing \"") + line + "\"\n" + joinedLog() + timeline.report();
    TEST_ASSERT_TRUE_MESSAGE(isFound, message.c_str());
}

void assertMoves(const Timeline &timeline, const std::vector<ExpectedMove> &expected, size_t first = 0)
{
    const std::string difference = compareMoves(timeline.moves, expected, first);
    TEST_ASSERT_TRUE_MESSAGE(difference.empty(), (difference + joinedLog()).c_str());
}

// Every actuator on `boardCount` boards stopped, with no position target.
void assertAllStoppedWithoutTargets(const Timeline &timeline, int boardCount)
{
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());
    if (boardCount == 1) return;
    for (int board = 0; board < boardCount; ++board)
        for (int joint = 0; joint < kJoints; ++joint)
            TEST_ASSERT_EQUAL_INT_MESSAGE(0,
                krabby_host_three_board_actuator_has_target(kJointNames[board][joint]),
                kJointNames[board][joint]);
}

// Distinct travel per board and joint so each board's saved record is its own.
void setDistinctRodLimits(Timeline &timeline)
{
    for (int board = 0; board < kBoards; ++board)
        for (int joint = 0; joint < kJoints; ++joint)
        {
            timeline.rods[board][joint].low = 150 + 10 * board + joint;
            timeline.rods[board][joint].high = 850 - 10 * board - joint;
        }
}

void bootThreeBoards(Timeline &timeline)
{
    timeline.install();
    int roles[3] = {0, 0, 0};
    krabby_host_run_three_board_role_election(roles);
    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, roles[0]);
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, roles[1]);
    TEST_ASSERT_EQUAL_INT(ROLE_RIGHT, roles[2]);
    krabby_host_enable_three_board_firmware_loop(1);
    krabby_host_firmware_setup();
    // Boot, settle the pot filters and let the open-pot probes run.
    runFor(timeline, 3000);
    timeline.resetEepromTracking();  // role election and boot write EEPROM too
    g_log.clear();
}

void bootSingleBoard(Timeline &timeline)
{
    timeline.install();
    krabby_host_firmware_setup();
    runFor(timeline, 3000);
    timeline.resetEepromTracking();  // boot writes EEPROM too
    g_log.clear();
}

// Starts a strict run (options 0: no knee nudge) of both passes on the single
// board. Only ActuatorManager::startCalibration can: every C command run uses
// the shipped default, the knee nudge.
void startStrictRun(Timeline &timeline, int pwm)
{
    timeline.kneeNudgeOption = false;
    TEST_ASSERT_TRUE(krabby::sim::host::actuatorManager().startCalibration(
        CAL_PASS_SEEK | CAL_PASS_SWEEP, pwm, 0));
    TEST_ASSERT_EQUAL_UINT8(0, krabby::sim::host::actuatorManager().calibrationSequencer().options());
}

void setYawStops(int board, int minStop, int maxStop)
{
    krabby_host_three_board_set_actuator_stops(kJointNames[board][0], minStop, maxStop);
    krabby_host_three_board_set_actuator_stops(kJointNames[board][3], minStop + 1, maxStop - 1);
}

// The board's saved record: hall hips [0, 1069] (curl switch to extend switch,
// derived), knees [-S, 0] with S within [minStroke, maxStroke], yaws Unknown
// 0/0, kind bits for hall hips and knees.
void assertCanonicalHallRecord(int board, int minStroke, int maxStroke)
{
    const HostCalData saved = readCalData(board);
    TEST_ASSERT_EQUAL_HEX16(kSpecCalMagic, saved.magic);
    TEST_ASSERT_EQUAL_HEX16(calDataCheck(saved), saved.check);
    TEST_ASSERT_EQUAL_HEX16(kHallHipsAndKneesBits, saved.sensorBits);
    for (int joint = 0; joint < kJoints; ++joint)
    {
        const char *name = kJointNames[board][joint];
        if (calibration_timeline::isYaw(joint))
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(0, saved.minVals[joint], name);
            TEST_ASSERT_EQUAL_INT_MESSAGE(0, saved.maxVals[joint], name);
            continue;
        }
        if (calibration_timeline::isHip(joint))
        {
            TEST_ASSERT_EQUAL_INT_MESSAGE(0, saved.minVals[joint], name);
            TEST_ASSERT_EQUAL_INT_MESSAGE(kSpecHipStrokeEdges, saved.maxVals[joint], name);
            continue;
        }
        const int stroke = saved.maxVals[joint] - saved.minVals[joint];
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, saved.maxVals[joint], name);
        TEST_ASSERT_TRUE_MESSAGE(stroke >= minStroke && stroke <= maxStroke,
            (std::string(name) + " stroke " + std::to_string(stroke)).c_str());
    }
}

// The SENS code of every joint of a board ('\0': none seen), via the C API.
std::string sensorStates(int board)
{
    std::string states;
    for (int joint = 0; joint < kJoints; ++joint)
    {
        const char code = krabby_host_three_board_sensor_state(kJointNames[board][joint]);
        states.push_back(code == '\0' ? '-' : code);
    }
    return states;
}

// Operator jog of one joint: J at pwm for ms, then J 0.
void jog(Timeline &timeline, const char *name, int pwm, uint32_t ms)
{
    char wire[32];
    std::snprintf(wire, sizeof(wire), "J%s %d\n", name, pwm);
    writeUsb(wire);
    runFor(timeline, ms);
    std::snprintf(wire, sizeof(wire), "J%s 0\n", name);
    writeUsb(wire);
    runFor(timeline, 100);
}

// Edges a move covered (rod travel times the rod's edges per count).
double moveEdges(const Timeline &timeline, const Move &move)
{
    const Rod &rod = timeline.rods[move.board][move.joint];
    return std::fabs(move.endPosition - move.startPosition) * rod.edgesPerCount;
}

// The first logged line starting with `prefix`, or nullptr.
const LoggedLine *findLineStarting(const std::string &prefix)
{
    for (const LoggedLine &line : g_log)
        if (line.text.rfind(prefix, 0) == 0) return &line;
    return nullptr;
}

void runUntilLineStarting(Timeline &timeline, const std::string &prefix, uint32_t maxMs)
{
    for (uint32_t elapsed = 0; elapsed < maxMs && findLineStarting(prefix) == nullptr; elapsed += kTickMs)
        tick(timeline);
    const std::string message = "missing \"" + prefix + "...\"\n" + joinedLog() + timeline.report();
    TEST_ASSERT_TRUE_MESSAGE(findLineStarting(prefix) != nullptr, message.c_str());
}

}  // namespace

// A failed assertion longjmps out of a test, skipping ~Timeline: drop its observer first.
static void dropTimelineObserver() { krabby::sim::pinState().setWriteObserver(nullptr); }

void setUp()
{
    dropTimelineObserver();
    krabby_host_reset();
    g_log.clear();
    g_other.clear();
    g_dropLeftCommands = false;
}

void tearDown()
{
    dropTimelineObserver();
    krabby_host_reset();
}

// ---------------------------------------------------------------- three boards

static void test_robot_calibration_full_run_keeps_every_invariant_and_saves_each_board()
{
    Timeline timeline;
    setDistinctRodLimits(timeline);
    timeline.requestedPwm = 120;
    bootThreeBoards(timeline);
    for (int board = 0; board < kBoards; ++board) setYawStops(board, 120 + board, 880 - board);

    writeUsb("C 120\n");
    assertRunUntil(timeline, "CAL LEFT PASS SEEK pwm=120", 60000);
    // A second C while it runs is refused and changes nothing.
    writeUsb("C 150\n");
    assertRunUntil(timeline, "CAL ROBOT BUSY", 100);
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    // Pot hips: each leg's SWEEP keeps the earlier order (no table-safe park).
    std::vector<ExpectedMove> expected;
    for (int board = 0; board < kBoards; ++board) appendSeek(expected, board);
    for (int board = 0; board < kBoards; ++board) appendSweep(expected, board, false);
    assertMoves(timeline, expected);
    for (const Move &move : timeline.moves)
    {
        TEST_ASSERT_EQUAL_INT(120, move.peakMagnitude);
        TEST_ASSERT_NOT_EQUAL(0, move.endMs);
        TEST_ASSERT_TRUE(move.endMs - move.startMs < kSpecMoveTimeoutMs);
    }
    // Each move's END line comes after the motor was stopped and the pot settled.
    std::vector<uint32_t> endLineMs;
    for (const LoggedLine &line : g_log)
        if (line.text.find(" END ") != std::string::npos) endLineMs.push_back(line.ms);
    TEST_ASSERT_EQUAL_UINT(timeline.moves.size(), endLineMs.size());
    for (size_t i = 0; i < endLineMs.size(); ++i)
    {
        const uint32_t settled = endLineMs[i] - timeline.moves[i].endMs;
        TEST_ASSERT_TRUE_MESSAGE(settled >= kSpecSettleMs && settled <= kSpecSettleMs + 2 * kTickMs,
            std::to_string(settled).c_str());
    }
    // The whole robot is curled before the first SWEEP move (SEEK: 3 boards x 2 legs x 3 moves).
    TEST_ASSERT_EQUAL_HEX32(Timeline::allHipsAndKnees(kBoards), timeline.moves[18].curledAtStart);

    // Each board saved exactly once, right after its own SWEEP and before the next board moved.
    TEST_ASSERT_EQUAL_UINT(3, timeline.eepromChanges.size());
    for (int board = 0; board < kBoards; ++board)
    {
        TEST_ASSERT_EQUAL_INT(board, timeline.eepromChanges[board].first);
        const uint32_t savedMs = timeline.eepromChanges[board].second;
        const size_t lastMove = 18 + 12 * board + 11;
        TEST_ASSERT_TRUE(savedMs >= timeline.moves[lastMove].endMs);
        if (lastMove + 1 < timeline.moves.size())
            TEST_ASSERT_TRUE(savedMs <= timeline.moves[lastMove + 1].startMs);
    }

    // min = end reached with -PWM (retracted), max = with +PWM (extended); yaws untouched.
    // Without quadrature halls every joint is saved as a pot.
    for (int board = 0; board < kBoards; ++board)
    {
        const HostCalData saved = readCalData(board);
        TEST_ASSERT_EQUAL_HEX16(kSpecCalMagic, saved.magic);
        TEST_ASSERT_EQUAL_HEX16(kAllPotBits, saved.sensorBits);
        TEST_ASSERT_EQUAL_HEX16(calDataCheck(saved), saved.check);
        for (int joint = 0; joint < kJoints; ++joint)
        {
            const char *name = kJointNames[board][joint];
            const int minStop = krabby_host_three_board_actuator_min_stop(name);
            const int maxStop = krabby_host_three_board_actuator_max_stop(name);
            TEST_ASSERT_EQUAL_INT_MESSAGE(minStop, saved.minVals[joint], name);
            TEST_ASSERT_EQUAL_INT_MESSAGE(maxStop, saved.maxVals[joint], name);
            if (calibration_timeline::isYaw(joint))
            {
                const int offset = joint == 0 ? 0 : 1;
                TEST_ASSERT_EQUAL_INT_MESSAGE(120 + board + offset, minStop, name);
                TEST_ASSERT_EQUAL_INT_MESSAGE(880 - board - offset, maxStop, name);
                continue;
            }
            TEST_ASSERT_INT_WITHIN_MESSAGE(1, timeline.rods[board][joint].low, minStop, name);
            TEST_ASSERT_INT_WITHIN_MESSAGE(1, timeline.rods[board][joint].high, maxStop, name);
        }
    }

    static const char *const kOrder[] = {
        "CAL ROBOT START pwm=120 passes=SEEK,SWEEP",
        "CAL FRONT PASS SEEK pwm=120",
        "CAL FRONT PASS_DONE SEEK",
        "CAL LEFT PASS SEEK pwm=120",
        "CAL LEFT PASS_DONE SEEK",
        "CAL RIGHT PASS SEEK pwm=120",
        "CAL RIGHT PASS_DONE SEEK",
        "CAL FRONT PASS SWEEP pwm=120",
        "CAL FRONT PASS_DONE SWEEP saved",
        "CAL LEFT PASS SWEEP pwm=120",
        "CAL LEFT PASS_DONE SWEEP saved",
        "CAL RIGHT PASS SWEEP pwm=120",
        "CAL RIGHT PASS_DONE SWEEP saved",
        "CAL ROBOT DONE",
    };
    size_t cursor = 0;
    for (const char *line : kOrder)
    {
        while (cursor < g_log.size() && g_log[cursor].text != line) ++cursor;
        TEST_ASSERT_TRUE_MESSAGE(cursor < g_log.size(), line);
    }
    TEST_ASSERT_EQUAL_INT(1, countLines("CAL ROBOT BUSY"));
    // The filtered pot settles within a count of the stop (JOINT prints the truncated reading).
    bool isJointReported = false;
    for (const LoggedLine &line : g_log)
        isJointReported = isJointReported || line.text.rfind("CAL LEFT JOINT RLHL min=161 max=83", 0) == 0;
    TEST_ASSERT_TRUE_MESSAGE(isJointReported, joinedLog().c_str());
    TEST_ASSERT_FALSE(hasLine("CAL ROBOT SKIP LEFT no link"));
    for (const LoggedLine &line : g_log)
    {
        TEST_ASSERT_TRUE_MESSAGE(line.text.size() < 100, line.text.c_str());
        TEST_ASSERT_TRUE_MESSAGE(line.text.find(';') == std::string::npos, line.text.c_str());
    }
    // Without quadrature halls no SENS line is printed and no SENSOR line either.
    TEST_ASSERT_EQUAL_STRING("------", sensorStates(0).c_str());
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT SENSOR "));
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// A pot that reads lower when extended moves "the wrong way" for its drive:
// the pass fails at once (it used to warn and save) and its hip never moves.
static void test_sweep_only_from_mid_stroke_curls_first_and_reversed_pot_fails_the_pass()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[1][5].isReversedPot = true;  // MLKL
    bootThreeBoards(timeline);

    writeUsb("C SWEEP\n");
    assertRunUntil(timeline,
        "CAL ROBOT FAIL LEFT: MLKL pot moves the wrong way: check motor or pot end wires", 600000);
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSweep(expected, 0, false);
    appendSweepLeg(expected, 1, 0, false);  // LEFT leg A
    expected.push_back({1, 5, kSpecKneeCurlSign});  // MLKL curls, wrong way: the pass ends
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL ROBOT START pwm=128 passes=SWEEP"));
    TEST_ASSERT_FALSE(hasLine("CAL FRONT PASS SEEK pwm=128"));
    TEST_ASSERT_TRUE(hasLine("CAL LEFT FAIL MLKL pot moves the wrong way: check motor or pot end wires"));
    TEST_ASSERT_FALSE(hasLine("CAL LEFT MOVE MLHL curl"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL LEFT WARN "));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT PASS_DONE SWEEP saved"));
    TEST_ASSERT_FALSE(hasLine("CAL LEFT PASS_DONE SWEEP saved"));
    // FRONT saved its SWEEP; LEFT and RIGHT saved nothing.
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    TEST_ASSERT_EQUAL_INT(0, timeline.eepromChanges[0].first);
    TEST_ASSERT_EQUAL_INT(0, timeline.movesOnBoard(2));
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

static void test_seek_only_curls_every_leg_and_never_saves()
{
    Timeline timeline;
    timeline.requestedPwm = 110;
    bootThreeBoards(timeline);

    writeUsb("C SEEK 110\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 300000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int board = 0; board < kBoards; ++board) appendSeek(expected, board);
    assertMoves(timeline, expected);
    TEST_ASSERT_EQUAL_HEX32(Timeline::allHipsAndKnees(kBoards), timeline.curledMask());
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_FALSE(hasLine("CAL FRONT PASS SWEEP pwm=110"));
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// Starts a full calibration, waits for `waitLine` plus 300 ms (one joint mid-move),
// sends `wire` and checks that every board stopped at once and stays stopped.
static void runStopCommandCase(
    const char *waitLine,
    const char *wire,
    const char *robotLine,
    const char *boardLine,
    size_t expectedSaves)
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);
    writeUsb("C\n");
    assertRunUntil(timeline, waitLine, 600000);
    runFor(timeline, 300);
    TEST_ASSERT_EQUAL_INT(1, timeline.drivenCount());
    const size_t savesBefore = timeline.eepromChanges.size();
    TEST_ASSERT_EQUAL_UINT(expectedSaves, savesBefore);

    writeUsb(wire);
    tick(timeline);
    TEST_ASSERT_EQUAL_INT_MESSAGE(0, timeline.drivenCount(), wire);
    timeline.forbidDrives();
    runFor(timeline, 30000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasLine(robotLine), joinedLog().c_str());
    TEST_ASSERT_TRUE_MESSAGE(hasLine(boardLine), joinedLog().c_str());
    TEST_ASSERT_EQUAL_UINT(savesBefore, timeline.eepromChanges.size());
    TEST_ASSERT_FALSE(hasLine("CAL ROBOT DONE"));
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

static void test_hold_during_front_seek_stops_the_robot()
{
    runStopCommandCase("CAL FRONT MOVE FLHL curl", "H\n", "CAL ROBOT ABORTED H", "CAL FRONT ABORTED H", 0);
}

static void test_jog_during_left_seek_stops_the_robot()
{
    runStopCommandCase("CAL LEFT MOVE RLHL curl", "JRLKL 0\n", "CAL ROBOT ABORTED J", "CAL LEFT ABORTED H", 0);
}

static void test_trajectory_during_front_sweep_stops_the_robot_before_any_save()
{
    runStopCommandCase("CAL FRONT MOVE FRKL uncurl", "T\n", "CAL ROBOT ABORTED T", "CAL FRONT ABORTED T", 0);
}

static void test_batch_jog_during_right_sweep_keeps_earlier_saves_only()
{
    runStopCommandCase("CAL RIGHT MOVE MRHL uncurl", "B MRHL 0\n", "CAL ROBOT ABORTED B", "CAL RIGHT ABORTED H", 2);
}

static void test_jog_after_abort_still_moves_the_jogged_joint_only()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);
    writeUsb("C\n");
    assertRunUntil(timeline, "CAL LEFT MOVE RLHL curl", 600000);
    runFor(timeline, 300);

    // From here only the jogged joint may move (and a jog need not ramp).
    timeline.forbidDrives(1, 1);
    writeUsb("JRLHL 60\n");
    tick(timeline);
    TEST_ASSERT_EQUAL_INT(1, timeline.drivenCount());
    TEST_ASSERT_EQUAL_INT(60, timeline.drive(1, 1));
    runFor(timeline, 2000);
    TEST_ASSERT_EQUAL_INT(60, timeline.drive(1, 1));
    TEST_ASSERT_EQUAL_INT(1, timeline.drivenCount());
    TEST_ASSERT_TRUE(hasLine("CAL ROBOT ABORTED J"));
    TEST_ASSERT_TRUE(hasLine("CAL LEFT ABORTED H"));
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());

    writeUsb("H\n");
    tick(timeline);
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
}

static void test_follower_open_pot_fails_the_robot_and_saves_nothing()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[1][1].isOpenPot = true;  // RLHL
    bootThreeBoards(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT FAIL LEFT: RLHL pot not connected", 300000);
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL LEFT FAIL RLHL pot not connected"));
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    expected.push_back({1, 2, kSpecKneeCurlSign});  // RLKL curls, then RLHL refuses to start
    assertMoves(timeline, expected);
    TEST_ASSERT_EQUAL_INT(0, timeline.movesOnBoard(2));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

static void test_follower_move_timeout_fails_the_robot_and_keeps_earlier_saves()
{
    Timeline timeline;
    timeline.requestedPwm = 150;
    // RRHL has no extended end: it creeps 6 counts/s on its uncurl move.
    Rod &rrhl = timeline.rods[2][1];
    rrhl.high = 100000;
    rrhl.speedScale = 0.01;
    rrhl.position = 200;  // 50 counts from its curl end: SEEK's curl shows real travel
    bootThreeBoards(timeline);
    for (int joint = 0; joint < kJoints; ++joint)
        krabby_host_three_board_set_actuator_stops(kJointNames[2][joint], 111 + joint, 999 - joint);

    writeUsb("C 150\n");
    assertRunUntil(timeline, "CAL ROBOT FAIL RIGHT: RRHL uncurl: no end of travel in 120 s", 900000);
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL RIGHT FAIL RRHL uncurl: no end of travel in 120 s"));
    const Move &last = timeline.moves.back();
    TEST_ASSERT_EQUAL_INT(2, last.board);
    TEST_ASSERT_EQUAL_INT(1, last.joint);
    TEST_ASSERT_EQUAL_INT(-kSpecHipCurlSign, last.sign);
    TEST_ASSERT_TRUE(last.endMs - last.startMs <= kSpecMoveTimeoutMs);
    TEST_ASSERT_TRUE(last.endMs - last.startMs > kSpecMoveTimeoutMs - 1000);
    // FRONT and LEFT finished their SWEEP and keep their saves; RIGHT saved nothing.
    TEST_ASSERT_EQUAL_UINT(2, timeline.eepromChanges.size());
    TEST_ASSERT_EQUAL_INT(0, timeline.eepromChanges[0].first);
    TEST_ASSERT_EQUAL_INT(1, timeline.eepromChanges[1].first);
    for (int joint = 0; joint < kJoints; ++joint)
    {
        TEST_ASSERT_EQUAL_INT(111 + joint, krabby_host_three_board_actuator_min_stop(kJointNames[2][joint]));
        TEST_ASSERT_EQUAL_INT(999 - joint, krabby_host_three_board_actuator_max_stop(kJointNames[2][joint]));
    }
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

static void test_silent_follower_trips_the_watchdog_and_holds_everything()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);
    g_dropLeftCommands = true;
    SimSerialPort &toLeft = serialPort(BoardId::kFront, SerialPortId::kSerial1);
    SimSerialPort &toRight = serialPort(BoardId::kFront, SerialPortId::kSerial2);
    while (toLeft.txLineAvailable()) toLeft.popTxLine();
    while (toRight.txLineAvailable()) toRight.popTxLine();

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT FAIL LEFT: no CAL report for 135 s", 400000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    const LoggedLine *handedOver = findLine("CAL FRONT PASS_DONE SEEK");
    const LoggedLine *failed = findLine("CAL ROBOT FAIL LEFT: no CAL report for 135 s");
    TEST_ASSERT_NOT_NULL(handedOver);
    const uint32_t silence = failed->ms - handedOver->ms;
    TEST_ASSERT_TRUE(silence >= kSpecFollowerSilenceMs);
    TEST_ASSERT_TRUE(silence <= kSpecFollowerSilenceMs + 2 * kTickMs);

    std::vector<std::string> leftLink;
    while (toLeft.txLineAvailable()) leftLink.push_back(toLeft.popTxLine());
    std::vector<std::string> rightLink;
    while (toRight.txLineAvailable()) rightLink.push_back(toRight.popTxLine());
    TEST_ASSERT_EQUAL_UINT(3, leftLink.size());
    TEST_ASSERT_EQUAL_STRING("H", leftLink[0].c_str());
    TEST_ASSERT_EQUAL_STRING("C SEEK 128", leftLink[1].c_str());
    TEST_ASSERT_EQUAL_STRING("H", leftLink[2].c_str());
    TEST_ASSERT_EQUAL_UINT(2, rightLink.size());
    TEST_ASSERT_EQUAL_STRING("H", rightLink[1].c_str());

    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// ------------------------------------------------------- three boards, halls

// Rev 3 robot, every hip and knee a hall encoder, started mid-stroke (RIGHT's
// hips 700 edges out, above 560 with their knees anywhere: the start pose).
// Knees are loaded (they slow over their last edges and their curl end is
// deeper at more PWM). Each board runs the table-safe SWEEP (each knee parked
// at -264 before its hip sweeps to 840), homes and saves canonical ranges in
// edges: each hip [0, 1069], its curl switch to its derived extend switch
// (never sought), each knee its full stroke.
static void test_all_hall_robot_from_mid_stroke_homes_and_saves_canonical_strokes()
{
    Timeline timeline;
    for (int board = 0; board < kBoards; ++board)
    {
        timeline.quadrature[board] = true;
        for (int joint = 0; joint < kJoints; ++joint)
        {
            Rod &rod = timeline.rods[board][joint];
            rod.edgesPerCount = 1.0 + 0.5 * board;  // strokes 700, 1050, 1400 edges
            rod.kneeTaper = true;
            rod.kneeCompliance = true;
        }
    }
    timeline.setHallRods(0, 2);
    timeline.requestedPwm = 120;
    bootThreeBoards(timeline);
    // Before any calibration every joint is Unknown: no position, '?'.
    for (int board = 0; board < kBoards; ++board)
        TEST_ASSERT_EQUAL_STRING("??????", sensorStates(board).c_str());

    writeUsb("C 120\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int board = 0; board < kBoards; ++board) appendSeek(expected, board);
    for (int board = 0; board < kBoards; ++board) appendSweep(expected, board, true);
    assertMoves(timeline, expected);
    TEST_ASSERT_EQUAL_HEX32(Timeline::allHipsAndKnees(kBoards), timeline.moves[18].curledAtStart);
    // Every END line still comes after the stop and the settle.
    size_t endLines = 0;
    for (const LoggedLine &line : g_log) endLines += line.text.find(" END ") != std::string::npos;
    TEST_ASSERT_EQUAL_UINT(timeline.moves.size(), endLines);

    TEST_ASSERT_EQUAL_UINT(3, timeline.eepromChanges.size());
    for (int board = 0; board < kBoards; ++board)
    {
        const int stroke = static_cast<int>(700 * (1.0 + 0.5 * board));
        // A knee's end is up to 17 edges deeper at PWM 120.
        assertCanonicalHallRecord(board, stroke - 2, stroke + 30);
        TEST_ASSERT_EQUAL_STRING("?EE?EE", sensorStates(board).c_str());
        for (int joint = 0; joint < kJoints; ++joint)
        {
            if (calibration_timeline::isYaw(joint)) continue;
            const char *name = kJointNames[board][joint];
            const char *role = board == 0 ? "FRONT" : board == 1 ? "LEFT" : "RIGHT";
            long count = 0;
            TEST_ASSERT_TRUE_MESSAGE(lastHomeCount(role, name, count), name);
            TEST_ASSERT_TRUE_MESSAGE(
                hasLine(std::string("CAL ") + role + " SENSOR " + name + " hall"), name);
        }
    }
    // A hip saves its derived stroke and its sweep stopped at 840: these rods do
    // not coast, so it goes no further than one 5 ms tick of travel at PWM 120
    // (2.4-4.8 edges). Each knee parked inside its band first.
    TEST_ASSERT_TRUE_MESSAGE(
        hasLine("CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"),
        joinedLog().c_str());
    TEST_ASSERT_TRUE(hasLine("CAL RIGHT JOINT MRHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_TRUE_MESSAGE(findLineStarting("CAL FRONT PARK FLKL hall=-2") != nullptr, joinedLog().c_str());
    TEST_ASSERT_EQUAL_INT(6, countLinesStarting("CAL LEFT PARK ") + countLinesStarting("CAL RIGHT PARK ") +
                                 countLinesStarting("CAL FRONT PARK "));
    for (int board = 0; board < kBoards; ++board)
    {
        const double tickEdges =
            120 * calibration_timeline::kCountsPerPwmMs * kTickMs * timeline.rods[board][1].edgesPerCount;
        for (int hip : {1, 4})
        {
            const double furthest = timeline.furthestOutEdges(board, hip);
            TEST_ASSERT_TRUE_MESSAGE(furthest >= kSpecHipSweepEdges && furthest <= kSpecHipSweepEdges + tickEdges,
                std::to_string(furthest).c_str());
        }
    }
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT WARN "));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL LEFT WARN "));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL RIGHT WARN "));
    for (const LoggedLine &line : g_log)
    {
        TEST_ASSERT_TRUE_MESSAGE(line.text.find(" pot") == std::string::npos, line.text.c_str());
        TEST_ASSERT_TRUE_MESSAGE(line.text.size() < 100, line.text.c_str());
    }
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// Hall hips and knees parked fully curled after power-up, calibrated at C 150.
// With the shipped default (the knee nudge, used by every C command run) one
// run nudges each parked knee out at most 12 edges (1 s) and straight back
// before its hip is proven, nudges each parked hip out and back, curls each
// knee again (proven now, no nudge), then saves canonical strokes. Every nudge
// ramps only to PWM 128 (R4 for the hip): these stored joints need 0.75 s to
// break free (FLKL measured 0.34-0.77 s), so each nudge reaches its cap before
// it moves. Every other drive ramps to 150. No jogs are needed. The strict path is
// test_single_board_parked_curled_strict_needs_knee_jogs_then_saves.
static void test_all_hall_robot_parked_curled_nudges_every_joint_then_saves()
{
    Timeline timeline;
    for (int board = 0; board < kBoards; ++board)
    {
        timeline.quadrature[board] = true;
        for (int joint = 0; joint < kJoints; ++joint)
        {
            timeline.rods[board][joint].startAtCurlEnd = true;
            timeline.rods[board][joint].breakawayMs = 750;
        }
    }
    timeline.setHallRods(0, 2);
    timeline.requestedPwm = 150;
    TEST_ASSERT_TRUE(timeline.kneeNudgeOption);  // the C command's default
    bootThreeBoards(timeline);
    const size_t firstMove = timeline.moves.size();

    writeUsb("C 150\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int board = 0; board < kBoards; ++board)
        for (int leg = 0; leg < 2; ++leg)
        {
            appendNudged(expected, board, 3 * leg + 2);
            appendNudged(expected, board, 3 * leg + 1);
            expected.push_back({board, 3 * leg + 2, kSpecKneeCurlSign});
        }
    for (int board = 0; board < kBoards; ++board) appendSweep(expected, board, true);
    assertMoves(timeline, expected, firstMove);
    // Each SEEK nudge stays inside its bound: a hip at most 2 s and about 24
    // edges plus coast, a knee at most 1 s and 16 edges (also the invariant),
    // both at PWM 128; every other move ramps to the requested 150.
    const size_t seekMoves = kBoards * 2 * 7;  // boards x legs x (curl, nudge, curl) x 2 + knee curl
    int nudges = 0;
    for (size_t i = firstMove; i < timeline.moves.size(); ++i)
    {
        const Move &move = timeline.moves[i];
        const bool isNudge = i < firstMove + seekMoves && move.sign != calibration_timeline::curlSign(move.joint);
        TEST_ASSERT_EQUAL_INT_MESSAGE(isNudge ? kSpecNudgeMaxPwm : 150, move.peakMagnitude,
            std::to_string(i - firstMove).c_str());
        if (!isNudge) continue;
        ++nudges;
        const bool hip = calibration_timeline::isHip(move.joint);
        TEST_ASSERT_TRUE(move.endMs - move.startMs <= (hip ? kSpecHipNudgeMaxMs : kSpecKneeNudgeMaxMs));
        TEST_ASSERT_TRUE_MESSAGE(
            moveEdges(timeline, move) <= (hip ? kSpecHipNudgeEdges + 6 : kSpecKneeNudgeMaxEdges),
            std::to_string(moveEdges(timeline, move)).c_str());
    }
    TEST_ASSERT_EQUAL_INT(kBoards * 4, nudges);
    static const char *const kRoles[kBoards] = {"FRONT", "LEFT", "RIGHT"};
    for (int board = 0; board < kBoards; ++board)
        for (int joint = 0; joint < kJoints; ++joint)
        {
            if (calibration_timeline::isYaw(joint)) continue;
            const std::string line = std::string("CAL ") + kRoles[board] + " MOVE " +
                kJointNames[board][joint] + " nudge";
            TEST_ASSERT_EQUAL_INT_MESSAGE(1, countLinesStarting(line), line.c_str());
        }
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT FAIL "));
    TEST_ASSERT_EQUAL_UINT(3, timeline.eepromChanges.size());
    for (int board = 0; board < kBoards; ++board)
    {
        assertCanonicalHallRecord(board, 698, 702);
        TEST_ASSERT_EQUAL_STRING("?EE?EE", sensorStates(board).c_str());
    }
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"),
        joinedLog().c_str());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// Rev 3 boards: FRONT has hall hips and knees, LEFT is fitted with pots, RIGHT
// has a hall leg A and a pot leg B. Each joint is classified on its own; a pot
// on a hall board needs one repeat of its step. Each leg's SWEEP follows its
// hip: table-safe for a hall hip, the earlier order for a pot hip.
static void test_mixed_robot_classifies_each_joint_and_saves_its_kind()
{
    Timeline timeline;
    for (int board = 0; board < kBoards; ++board) timeline.quadrature[board] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[2][1].sensor = RodSensor::Hall;
    timeline.rods[2][1].high = timeline.rods[2][1].low + calibration_timeline::kHallHipStrokeEdges;
    timeline.rods[2][2].sensor = RodSensor::Hall;
    // Slow pots: the first, hall-windowed attempt stops well before the end.
    for (int joint : {1, 2, 4, 5}) timeline.rods[1][joint].speedScale = 0.25;
    timeline.rods[2][4].speedScale = 0.25;
    timeline.rods[2][5].speedScale = 0.25;
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 900000);
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    // LEFT: the first two steps of each leg a hall-windowed attempt, then the
    // repeat as a pot; the knee's second curl is a pot curl already.
    for (int leg = 0; leg < 2; ++leg)
    {
        for (int joint : {3 * leg + 2, 3 * leg + 1})
            for (int attempt = 0; attempt < 2; ++attempt)
                expected.push_back({1, joint, calibration_timeline::curlSign(joint)});
        expected.push_back({1, 3 * leg + 2, kSpecKneeCurlSign});
    }
    appendSeekLeg(expected, 2, 0);
    // MRKL and MRHL: a hall-windowed attempt, then the repeat as pots.
    expected.push_back({2, 5, kSpecKneeCurlSign});
    expected.push_back({2, 5, kSpecKneeCurlSign});
    expected.push_back({2, 4, kSpecHipCurlSign});
    expected.push_back({2, 4, kSpecHipCurlSign});
    expected.push_back({2, 5, kSpecKneeCurlSign});
    appendSweep(expected, 0, true);
    appendSweep(expected, 1, false);
    appendSweepLeg(expected, 2, 0, true);
    appendSweepLeg(expected, 2, 1, false);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL RIGHT SENSOR MRKL pot"));
    TEST_ASSERT_TRUE(hasLine("CAL RIGHT SENSOR MRHL pot"));
    TEST_ASSERT_TRUE(hasLine("CAL RIGHT SENSOR RRKL hall"));
    TEST_ASSERT_TRUE(hasLine("CAL LEFT SENSOR RLKL pot"));
    TEST_ASSERT_EQUAL_INT(4, countLinesStarting("CAL LEFT SENSOR "));

    TEST_ASSERT_EQUAL_UINT(3, timeline.eepromChanges.size());
    assertCanonicalHallRecord(0, 698, 702);
    const HostCalData left = readCalData(1);
    TEST_ASSERT_EQUAL_HEX16(kPotHipsAndKneesBits, left.sensorBits);
    TEST_ASSERT_INT_WITHIN(1, 150, left.minVals[1]);
    TEST_ASSERT_INT_WITHIN(1, 850, left.maxVals[2]);
    const HostCalData right = readCalData(2);
    TEST_ASSERT_EQUAL_HEX16(kSpecCalMagic, right.magic);
    TEST_ASSERT_EQUAL_HEX16(kHallLegAPotLegBBits, right.sensorBits);
    TEST_ASSERT_EQUAL_INT(0, right.minVals[1]);
    TEST_ASSERT_EQUAL_INT(kSpecHipStrokeEdges, right.maxVals[1]);  // the hip's derived stroke
    TEST_ASSERT_INT_WITHIN(2, -700, right.minVals[2]);
    TEST_ASSERT_EQUAL_INT(0, right.maxVals[2]);
    for (int joint = 4; joint <= 5; ++joint)
    {
        TEST_ASSERT_INT_WITHIN(1, 150, right.minVals[joint]);
        TEST_ASSERT_INT_WITHIN(1, 850, right.maxVals[joint]);
    }
    TEST_ASSERT_EQUAL_INT(0, right.minVals[0]);
    TEST_ASSERT_EQUAL_INT(0, right.maxVals[0]);

    TEST_ASSERT_EQUAL_STRING("?EE?EE", sensorStates(0).c_str());
    TEST_ASSERT_EQUAL_STRING("?PP?PP", sensorStates(1).c_str());
    TEST_ASSERT_EQUAL_STRING("?EE?PP", sensorStates(2).c_str());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// Review finding R1-8: a mixed firmware set. FRONT runs this firmware on a
// Rev 3 board; LEFT runs a build that prints no SENS line (an older build, or
// one without quadrature halls) and would treat hall joints as pots. A robot
// calibration is refused before anything moves, naming LEFT and telling the
// operator to flash all three boards with the same firmware. Nothing is saved.
static void test_mixed_firmware_robot_refuses_to_calibrate()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.quadrature[2] = true;
    timeline.setHallRods(0, 2);
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);
    TEST_ASSERT_FALSE(lastSensLine("RIGHT").empty());
    TEST_ASSERT_TRUE(lastSensLine("LEFT").empty());

    writeUsb("C\n");
    runFor(timeline, 10000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasLine(
        "CAL ROBOT FAIL LEFT: no SENS: flash all three boards with the same firmware, "
        "or wait a few seconds after connecting"), joinedLog().c_str());
    TEST_ASSERT_FALSE(hasLine("CAL ROBOT START pwm=128 passes=SEEK,SWEEP"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT "));
    TEST_ASSERT_TRUE(timeline.moves.empty());
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// Three Rev 3 boards running this firmware, C sent before the followers have
// printed their first SENS lines (just after connecting): refused, nothing
// moves. A few seconds later the same C runs.
static void test_calibration_right_after_connecting_waits_for_the_followers_sens_lines()
{
    Timeline timeline;
    for (int board = 0; board < kBoards; ++board) timeline.quadrature[board] = true;
    timeline.setHallRods(0, 2);
    timeline.requestedPwm = 128;
    timeline.install();
    int roles[3] = {0, 0, 0};
    krabby_host_run_three_board_role_election(roles);
    krabby_host_enable_three_board_firmware_loop(1);
    krabby_host_firmware_setup();
    timeline.resetEepromTracking();
    g_log.clear();

    writeUsb("C SEEK\n");
    tick(timeline);  // the first loop: no follower has printed SENS yet
    TEST_ASSERT_TRUE_MESSAGE(hasLine(
        "CAL ROBOT FAIL LEFT: no SENS: flash all three boards with the same firmware, "
        "or wait a few seconds after connecting"), joinedLog().c_str());
    runFor(timeline, 3000);
    TEST_ASSERT_TRUE(timeline.moves.empty());

    writeUsb("C SEEK\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 300000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int board = 0; board < kBoards; ++board) appendSeek(expected, board);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
}

// RLKL's hall A and B are swapped: its count runs against the drive. The move
// fails within the detection threshold and RLHL never moves.
static void test_reversed_hall_wiring_fails_wrong_direction_before_the_hip_moves()
{
    Timeline timeline;
    for (int board = 0; board < kBoards; ++board) timeline.quadrature[board] = true;
    timeline.setHallRods(0, 2);
    timeline.rods[1][2].hallReversed = true;
    timeline.requestedPwm = 128;
    bootThreeBoards(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline,
        "CAL ROBOT FAIL LEFT: RLKL hall counts the wrong way: check motor or hall A/B wiring", 300000);
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL LEFT FAIL RLKL hall counts the wrong way: check motor or hall A/B wiring"));
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    expected.push_back({1, 2, kSpecKneeCurlSign});
    assertMoves(timeline, expected);
    TEST_ASSERT_FALSE(hasLine("CAL LEFT MOVE RLHL curl"));
    // Stopped within the 12-edge threshold plus one tick of travel.
    TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, timeline.moves.back()) <= 12 + 4,
        std::to_string(moveEdges(timeline, timeline.moves.back())).c_str());
    TEST_ASSERT_EQUAL_INT(0, timeline.movesOnBoard(2));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, kBoards);
}

// ------------------------------------------------------------------ one board

// The accepted PWM range is 110-150 (owner decision F4, 2026-11-24: PWM 80
// stalls mid-stroke with no current signature; 40 used to be the minimum).
static void test_single_board_rejects_bad_requests_then_runs_at_minimum_pwm()
{
    Timeline timeline;
    timeline.requestedPwm = 110;
    bootSingleBoard(timeline);
    krabby_host_set_actuator_stops("FLHY", 123, 877);

    static const char *const kBadRequests[] = {
        "C 40\n", "C 80\n", "C 109\n", "C 151\n", "C fast\n", "C SEEK SWEEP\n", "C 120 x\n"};
    for (const char *wire : kBadRequests)
    {
        g_log.clear();
        writeUsb(wire);
        runFor(timeline, 200);
        TEST_ASSERT_EQUAL_UINT_MESSAGE(1, g_log.size(), wire);
        TEST_ASSERT_EQUAL_STRING_MESSAGE(
            "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150", g_log[0].text.c_str(), wire);
    }
    // C PWR_SENSE stays the power-monitor command (the host sim has no power monitors).
    g_log.clear();
    writeUsb("C PWR_SENSE ?\n");
    runFor(timeline, 3000);
    TEST_ASSERT_TRUE(g_log.empty());
    TEST_ASSERT_TRUE(timeline.moves.empty());
    TEST_ASSERT_EQUAL_INT(0, krabby_host_firmware_calibration_active());

    g_log.clear();
    writeUsb("C 110\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    appendSweep(expected, 0, false);
    assertMoves(timeline, expected);
    for (const Move &move : timeline.moves) TEST_ASSERT_EQUAL_INT(110, move.peakMagnitude);
    TEST_ASSERT_EQUAL_INT(2, countLines("CAL ROBOT SKIP LEFT no link"));
    TEST_ASSERT_EQUAL_INT(2, countLines("CAL ROBOT SKIP RIGHT no link"));
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    TEST_ASSERT_TRUE(timeline.eepromChanges[0].second >= timeline.moves.back().endMs);
    TEST_ASSERT_EQUAL_INT(123, krabby_host_actuator_min_stop("FLHY"));
    TEST_ASSERT_EQUAL_INT(877, krabby_host_actuator_max_stop("FLHY"));
    TEST_ASSERT_INT_WITHIN(1, 150, krabby_host_actuator_min_stop("FRKL"));
    TEST_ASSERT_INT_WITHIN(1, 850, krabby_host_actuator_max_stop("FRKL"));
    assertAllStoppedWithoutTargets(timeline, 1);
}

static void test_single_board_each_stop_command_aborts_and_nothing_moves_after()
{
    struct StopCase
    {
        const char *wire;
        const char *frontLine;
        const char *robotLine;
    };
    static const StopCase kCases[] = {
        {"H\n", "CAL FRONT ABORTED H", "CAL ROBOT ABORTED H"},
        {"JFLHY 0\n", "CAL FRONT ABORTED J", "CAL ROBOT ABORTED J"},
        {"B FLHY 0\n", "CAL FRONT ABORTED B", "CAL ROBOT ABORTED B"},
        {"T\n", "CAL FRONT ABORTED T", "CAL ROBOT ABORTED T"},
    };
    for (const StopCase &stop : kCases)
    {
        krabby_host_reset();
        g_log.clear();
        Timeline timeline;
        timeline.requestedPwm = 128;
        bootSingleBoard(timeline);
        writeUsb("C\n");
        assertRunUntil(timeline, "CAL FRONT MOVE FRHL curl", 600000);
        runFor(timeline, 300);
        TEST_ASSERT_EQUAL_INT_MESSAGE(1, timeline.drivenCount(), stop.wire);

        writeUsb(stop.wire);
        tick(timeline);
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, timeline.drivenCount(), stop.wire);
        timeline.forbidDrives();
        runFor(timeline, 10000);
        assertNoViolations(timeline);
        TEST_ASSERT_TRUE_MESSAGE(hasLine(stop.frontLine), stop.wire);
        TEST_ASSERT_TRUE_MESSAGE(hasLine(stop.robotLine), stop.wire);
        TEST_ASSERT_TRUE_MESSAGE(timeline.eepromChanges.empty(), stop.wire);
        assertAllStoppedWithoutTargets(timeline, 1);
    }
}

// A knee stuck at its curl end shows no motion. Without a measured limit it is
// not proven there (the old ADC-rail fallback is gone), so a strict run
// (options 0) fails it with the jog hint before its hip moves.
static void test_single_board_knee_stuck_at_its_curl_end_fails_no_motion()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[0][5].isStuck = true;  // FRKL sits at its curl end and never moves
    timeline.rods[0][5].position = timeline.rods[0][5].high;
    bootSingleBoard(timeline);
    // Stops set by hand are not a measured limit.
    krabby_host_set_actuator_stops("FRKL", 222, 850);

    startStrictRun(timeline, 128);
    assertRunUntil(timeline,
        "CAL FRONT FAIL FRKL no motion seen: jog it off its curl end, then retry", 600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeekLeg(expected, 0, 0);
    expected.push_back({0, 5, kSpecKneeCurlSign});
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT END FRKL curl pot=850 travel=0"));
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FRKL nudge"));
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FRHL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_INT(222, krabby_host_actuator_min_stop("FRKL"));
    TEST_ASSERT_EQUAL_INT(850, krabby_host_actuator_max_stop("FRKL"));
    assertAllStoppedWithoutTargets(timeline, 1);
}

// The same stuck knee on the default path (the C command): it is nudged out
// for at most 1 s, driven straight back, and, with no motion either way, the
// pass fails NoFeedback and names it before its hip moves.
static void test_single_board_knee_stuck_at_its_curl_end_is_nudged_then_fails_no_feedback()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[0][5].isStuck = true;  // FRKL sits at its curl end and never moves
    timeline.rods[0][5].position = timeline.rods[0][5].high;
    bootSingleBoard(timeline);
    krabby_host_set_actuator_stops("FRKL", 222, 850);

    writeUsb("C\n");
    assertRunUntil(timeline,
        "CAL ROBOT FAIL FRONT: FRKL no feedback: no hall edges, pot not moving (check sensor cable)",
        600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeekLeg(expected, 0, 0);
    appendNudged(expected, 0, 5);
    assertMoves(timeline, expected);
    const Move &nudge = timeline.moves[4];
    TEST_ASSERT_TRUE(nudge.endMs - nudge.startMs <= kSpecKneeNudgeMaxMs);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FRKL nudge"));
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FRHL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_INT(222, krabby_host_actuator_min_stop("FRKL"));
    TEST_ASSERT_EQUAL_INT(850, krabby_host_actuator_max_stop("FRKL"));
    assertAllStoppedWithoutTargets(timeline, 1);
}

// The same stuck knee with a measured limit from a saved record: its curls pass
// at the measured curl end without motion, and the uncurl is what fails.
static void test_single_board_joint_that_does_not_move_fails_and_saves_nothing()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[0][5].isStuck = true;  // FRKL sits at its curl end and never moves
    timeline.rods[0][5].position = timeline.rods[0][5].high;
    writePotRecord(0, 5, 222, 850);
    bootSingleBoard(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("Calibration v2 loaded"), joinedOther().c_str());

    writeUsb("C\n");
    assertRunUntil(timeline,
        "CAL ROBOT FAIL FRONT: FRKL barely moved (0 counts) at PWM 128: check pot wiring or raise speed",
        600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT END FRKL uncurl pot=850 travel=0"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_INT(222, krabby_host_actuator_min_stop("FRKL"));
    TEST_ASSERT_EQUAL_INT(850, krabby_host_actuator_max_stop("FRKL"));
    assertAllStoppedWithoutTargets(timeline, 1);
}

// Review finding: a hip that uncurls but cannot curl back (too little PWM to
// lift the leg, or held at its extend-end switch) used to count as curled after
// one still window, so its knee then swept with the leg hanging down and the
// board saved min == max. Now the pass fails there: the knee never leaves its
// curl end next to an uncurled hip (checked at every pin write) and nothing is saved.
static void test_single_board_hip_that_cannot_curl_back_fails_before_the_knee_moves()
{
    Timeline timeline;
    timeline.requestedPwm = 110;
    bootSingleBoard(timeline);
    const int minStopBefore = krabby_host_actuator_min_stop("FLHL");
    const int maxStopBefore = krabby_host_actuator_max_stop("FLHL");

    writeUsb("C 110\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLHL uncurl", 600000);
    timeline.rods[0][1].stallSign = kSpecHipCurlSign;  // from here on it cannot retract
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL did not reach its curl end", 600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT FAIL FLHL did not reach its curl end (pot=") != nullptr);
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLKL uncurl"));
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT JOINT FLHL") == nullptr);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_INT(minStopBefore, krabby_host_actuator_min_stop("FLHL"));
    TEST_ASSERT_EQUAL_INT(maxStopBefore, krabby_host_actuator_max_stop("FLHL"));
    assertAllStoppedWithoutTargets(timeline, 1);
}

// Review finding: a mid-stroke knee that does not move at all at the slowest
// speed used to count as curled after one still window, and its hip then
// moved. Now the pass fails before the hip is ever driven (strict run here, at
// the slowest accepted PWM; the default path is the hall case below).
static void test_single_board_knee_that_does_not_move_mid_stroke_fails_before_its_hip_moves()
{
    Timeline timeline;
    timeline.requestedPwm = 110;
    timeline.rods[0][2].stallSign = kSpecKneeCurlSign;  // FLKL at 500 cannot extend (curl)
    bootSingleBoard(timeline);

    startStrictRun(timeline, 110);
    assertRunUntil(timeline,
        "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry", 600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    TEST_ASSERT_EQUAL_INT(1, timeline.movesOnBoard(0));  // the knee alone, never its hip
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLHL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// The default path (the C command) with a mid-stroke hall knee that cannot
// curl: its knee nudge goes out at most 12 edges and 1 s, the curl back cannot
// move, and the pass fails and names it before the hip is ever driven.
static void test_single_board_hall_knee_that_cannot_curl_is_nudged_then_fails_before_its_hip_moves()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 110;
    timeline.rods[0][2].stallSign = kSpecKneeCurlSign;  // FLKL at 500 cannot extend (curl)
    bootSingleBoard(timeline);

    writeUsb("C 110\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLKL ", 600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendNudged(expected, 0, 2);
    assertMoves(timeline, expected);
    const Move &nudge = timeline.moves[1];
    TEST_ASSERT_TRUE(nudge.endMs - nudge.startMs <= kSpecKneeNudgeMaxMs);
    TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, nudge) <= kSpecKneeNudgeMaxEdges,
        std::to_string(moveEdges(timeline, nudge)).c_str());
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLKL nudge"));
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLHL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A pot hip mid-stroke that cannot curl shows no motion on SEEK: it is nudged
// out (stopped by the pot nudge bound), and the curl back cannot reach where
// it started, so the pass fails short before the next leg moves.
static void test_single_board_pot_hip_that_cannot_curl_is_nudged_and_fails_short()
{
    Timeline timeline;
    timeline.requestedPwm = 128;
    timeline.rods[0][1].stallSign = kSpecHipCurlSign;  // FLHL at 500 cannot retract (curl)
    bootSingleBoard(timeline);

    writeUsb("C\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL stopped ", 600000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected = {{0, 2, kSpecKneeCurlSign}};
    appendNudged(expected, 0, 1);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL nudge pot=") != nullptr);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT FAIL FLHL stopped ") != nullptr);
    const std::string failure = findLineStarting("CAL FRONT FAIL FLHL stopped ")->text;
    TEST_ASSERT_TRUE_MESSAGE(failure.find(" counts short of its curl end") != std::string::npos, failure.c_str());
    // The nudge stopped on its pot bound (30 counts of the smoothed reading,
    // which lags the rod), well inside its 2 s bound.
    const Move &nudge = timeline.moves[2];
    TEST_ASSERT_TRUE(nudge.endMs - nudge.startMs < kSpecHipNudgeMaxMs);
    TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, nudge) < 100, std::to_string(moveEdges(timeline, nudge)).c_str());
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FRKL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// ------------------------------------------------------- one board, halls

// The strict path (options 0, which no C command uses) with hall hips and
// knees parked fully curled: the first knee shows no motion and fails with the
// jog hint before its hip moves. After the operator jogs each knee off its curl
// end, a strict run nudges each parked hip, never a knee, and saves.
static void test_single_board_parked_curled_strict_needs_knee_jogs_then_saves()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    for (int joint = 0; joint < kJoints; ++joint) timeline.rods[0][joint].startAtCurlEnd = true;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    startStrictRun(timeline, 128);
    assertRunUntil(timeline,
        "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> firstRun = {{0, 2, kSpecKneeCurlSign}};
    assertMoves(timeline, firstRun);
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLKL nudge"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());

    // The operator jogs each knee a little off its curl end (it retracts to uncurl).
    timeline.setOperatorJogs(true);
    for (int joint = 2; joint < kJoints; joint += 3) jog(timeline, kJointNames[0][joint], -60, 300);
    timeline.setOperatorJogs(false);
    for (int joint = 2; joint < kJoints; joint += 3)
        TEST_ASSERT_FALSE_MESSAGE(timeline.rods[0][joint].isAtCurlEnd(joint), kJointNames[0][joint]);
    assertNoViolations(timeline);
    const size_t firstMove = timeline.moves.size();
    g_log.clear();

    startStrictRun(timeline, 128);
    assertRunUntil(timeline, "CAL FRONT PASS_DONE SWEEP saved", 600000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int leg = 0; leg < 2; ++leg)
    {
        expected.push_back({0, 3 * leg + 2, kSpecKneeCurlSign});
        appendNudged(expected, 0, 3 * leg + 1);
        expected.push_back({0, 3 * leg + 2, kSpecKneeCurlSign});
    }
    appendSweep(expected, 0, true);
    assertMoves(timeline, expected, firstMove);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLHL nudge"));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FRHL nudge"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT MOVE FLKL nudge"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT MOVE FRKL nudge"));
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    assertCanonicalHallRecord(0, 698, 702);
    assertAllStoppedWithoutTargets(timeline, 1);
}

// With CAL_OPT_KNEE_NUDGE a parked knee is nudged out at most 12 edges (1 s)
// and straight back before its hip is proven; the timeline allows exactly that
// bounded exception. The full run then saves canonical strokes. Started with
// the option passed explicitly; the C command runs are covered elsewhere.
static void test_single_board_parked_curled_with_knee_nudge_option_homes_every_joint()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    for (int joint = 0; joint < kJoints; ++joint) timeline.rods[0][joint].startAtCurlEnd = true;
    timeline.kneeNudgeOption = true;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    TEST_ASSERT_TRUE(krabby::sim::host::actuatorManager().startCalibration(
        CAL_PASS_SEEK | CAL_PASS_SWEEP, 128, CAL_OPT_KNEE_NUDGE));
    assertRunUntil(timeline, "CAL FRONT PASS_DONE SWEEP saved", 600000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int leg = 0; leg < 2; ++leg)
    {
        appendNudged(expected, 0, 3 * leg + 2);
        appendNudged(expected, 0, 3 * leg + 1);
        expected.push_back({0, 3 * leg + 2, kSpecKneeCurlSign});
    }
    appendSweep(expected, 0, true);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLKL nudge"));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FRKL nudge"));
    // Each knee nudge: at most 1 s and 16 edges (the invariant), and 12 counted
    // edges to prove motion (the rod's travel is quantised to whole edges).
    for (const size_t index : {size_t(1), size_t(8)})
    {
        const Move &nudge = timeline.moves[index];
        TEST_ASSERT_TRUE(calibration_timeline::isKnee(nudge.joint));
        TEST_ASSERT_EQUAL_INT(-kSpecKneeCurlSign, nudge.sign);
        TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, nudge) > 11, std::to_string(moveEdges(timeline, nudge)).c_str());
    }
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    assertCanonicalHallRecord(0, 698, 702);
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:E FLKL:E FRHY:? FRHL:E FRKL:E",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// The knee-nudge invariant itself: with the option off a knee leaving its curl
// end before its hip is proven is a violation; with it on, only a short,
// small excursion followed by that knee's curl is allowed.
static void test_timeline_knee_nudge_invariant_catches_unbounded_knee_moves()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.kneeNudgeOption = false;  // a strict run (options 0)
    bootSingleBoard(timeline);

    jog(timeline, "FLKL", -60, 100);
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("knee uncurled while its hip is not curled"),
        timeline.report().c_str());

    timeline.violations.clear();
    timeline.kneeNudgeOption = true;
    jog(timeline, "FLKL", -60, 1200);  // 1.2 s and about 290 edges
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("knee nudge longer than 1000 ms"), timeline.report().c_str());
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("knee nudge moved more than 16 edges"), timeline.report().c_str());
    jog(timeline, "FLKL", 60, 100);  // its curl back

    timeline.violations.clear();
    jog(timeline, "FRKL", -20, 30);  // a short, small nudge ...
    TEST_ASSERT_TRUE_MESSAGE(timeline.violations.empty(), timeline.report().c_str());
    jog(timeline, "FLKL", 20, 30);   // ... must be followed by that knee's curl
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("a knee nudge is not followed directly by its curl"),
        timeline.report().c_str());

    timeline.violations.clear();
    jog(timeline, "FRKL", -20, 30);
    jog(timeline, "FRKL", 20, 30);
    TEST_ASSERT_TRUE_MESSAGE(timeline.violations.empty(), timeline.report().c_str());
}

// A hall knee whose hall-B level flips (the pot pin jumps by about 1000 counts)
// before its first A edge: the jump is never read as pot motion. The knee is
// classified Hall from its edges and homed.
static void test_single_board_hall_b_jump_before_the_first_edge_is_not_pot_motion()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][2].position = 500.4;  // B flips 0.1 edge on, A after 0.6
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);
    TEST_ASSERT_EQUAL_INT(calibration_timeline::kHallBLow, timeline.rods[0][2].potReading());

    writeUsb("C SEEK\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLKL curl", 10000);
    const int32_t startCount = Timeline::hallCount(0, 2);
    bool isJumpSeen = false;
    for (int i = 0; i < 100 && Timeline::hallCount(0, 2) == startCount; ++i)
    {
        tick(timeline);
        if (timeline.rods[0][2].potReading() == calibration_timeline::kHallBHigh &&
            Timeline::hallCount(0, 2) == startCount)
            isJumpSeen = true;
    }
    TEST_ASSERT_TRUE(isJumpSeen);
    TEST_ASSERT_NOT_EQUAL(startCount, Timeline::hallCount(0, 2));
    assertRunUntil(timeline, "CAL ROBOT DONE", 300000);
    runFor(timeline, 1000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT SENSOR FLKL hall"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT SENSOR FLKL pot"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT FAIL "));
    long count = 0;
    TEST_ASSERT_TRUE(lastHomeCount("FRONT", "FLKL", count));
    TEST_ASSERT_EQUAL_INT(Timeline::hallCount(0, 2), count);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());  // SEEK saves nothing
    // Homed this boot but no stroke saved yet: 's'.
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:s FLKL:s FRHY:? FRHL:s FRKL:s",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// Rev 3 board fitted with pots: the first attempt of each joint is windowed on
// hall edges and sees none; at rest both snapshots show pot travel, so the
// joint becomes Pot and the step repeats with pot windows.
static void test_single_board_rev3_pot_robot_classifies_then_repeats_each_step()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    for (int joint = 0; joint < kJoints; ++joint) timeline.rods[0][joint].speedScale = 0.25;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 900000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int leg = 0; leg < 2; ++leg)
    {
        const int hip = 3 * leg + 1;
        const int knee = 3 * leg + 2;
        expected.push_back({0, knee, kSpecKneeCurlSign});
        expected.push_back({0, knee, kSpecKneeCurlSign});
        expected.push_back({0, hip, kSpecHipCurlSign});
        expected.push_back({0, hip, kSpecHipCurlSign});
        expected.push_back({0, knee, kSpecKneeCurlSign});  // a pot now: no repeat
    }
    appendSweep(expected, 0, false);
    assertMoves(timeline, expected);
    // The classified attempt prints no END: the step repeats at once.
    TEST_ASSERT_EQUAL_INT(1, countLines("CAL FRONT SENSOR FLKL pot"));
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT END FLKL curl hall="));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT SENSOR FRHL pot"));
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    const HostCalData saved = readCalData(0);
    TEST_ASSERT_EQUAL_HEX16(kSpecCalMagic, saved.magic);
    TEST_ASSERT_EQUAL_HEX16(kPotHipsAndKneesBits, saved.sensorBits);
    for (int joint = 0; joint < kJoints; ++joint)
    {
        if (calibration_timeline::isYaw(joint)) continue;
        TEST_ASSERT_INT_WITHIN(1, 150, saved.minVals[joint]);
        TEST_ASSERT_INT_WITHIN(1, 850, saved.maxVals[joint]);
    }
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:P FLKL:P FRHY:? FRHL:P FRKL:P",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A Rev 3 board whose saved record says pots, fitted with hall encoders (the
// pots swapped out): each hip and knee starts its first step as a proven Pot,
// windowed. Its first hall edges make it Hall, and that drive is stopped and
// driven again under the end detector (a hall end needs its signature; a
// window checks none). The run then goes on as an all-hall board: table-safe
// SWEEP, canonical hall record.
static void test_single_board_saved_pots_that_count_hall_edges_are_driven_again_under_the_detector()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 128;
    // A Rev 3 record (yaws Unknown) of pot hips and knees, 150-850.
    HostCalData record = {};
    for (int joint = 0; joint < kJoints; ++joint)
    {
        if (calibration_timeline::isYaw(joint)) continue;
        record.minVals[joint] = 150;
        record.maxVals[joint] = 850;
    }
    record.magic = kSpecCalMagic;
    record.sensorBits = kPotHipsAndKneesBits;
    record.check = calDataCheck(record);
    std::memcpy(EEPROM.boards[0].bytes, &record, sizeof(record));
    bootSingleBoard(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("Calibration v2 loaded"), joinedOther().c_str());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:P FLKL:P FRHY:? FRHL:P FRKL:P", lastSensLine("FRONT").c_str());

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int leg = 0; leg < 2; ++leg)
    {
        const int hip = 3 * leg + 1;
        const int knee = 3 * leg + 2;
        expected.push_back({0, knee, kSpecKneeCurlSign});  // windowed, as a pot
        expected.push_back({0, knee, kSpecKneeCurlSign});  // again, under the detector
        expected.push_back({0, hip, kSpecHipCurlSign});
        expected.push_back({0, hip, kSpecHipCurlSign});
        expected.push_back({0, knee, kSpecKneeCurlSign});
    }
    appendSweep(expected, 0, true);
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT SENSOR FLKL hall (was pot)"));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT SENSOR FRHL hall (was pot)"));
    // The windowed drive stops at the hall evidence (12 edges), with no END
    // line, far short of its curl end; the drive again reaches it and ends by
    // the detector's signature.
    TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, timeline.moves[0]) < 30.0,
        std::to_string(moveEdges(timeline, timeline.moves[0])).c_str());
    TEST_ASSERT_TRUE_MESSAGE(findLineStarting("CAL FRONT END FLKL curl hall=350 travel=338 by=stop ") != nullptr,
        joinedLog().c_str());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=-350 travel=-338 by=switch ") != nullptr);
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    assertCanonicalHallRecord(0, 698, 702);
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A hall sensor that gives no usable feedback. A hip, parked at its curl end or
// curling blind into it from mid-stroke, shows no edge: its curl press goes on
// as a curl with no motion (an unproven home), so it is nudged, stopped 1 s
// after it started with no edge counted (R4: not driven on blind to its 2 s
// cap, toward the body), driven blind back to its curl end (the return needs
// the switch-open signature), then fails NoFeedback; never a home. On a strict
// run (options 0) a knee fails NoMotion. The default-path dead knee is the
// next test.
static void test_single_board_dead_hall_sensor_fails_without_a_partner_move()
{
    struct DeadCase
    {
        int joint;
        bool deadA;
        bool deadB;
        bool unplugged;
        bool strict;
        bool parked;
        const char *failLine;
    };
    static const DeadCase kCases[] = {
        {1, true, false, false, false, true,
         "CAL FRONT FAIL FLHL no feedback: no hall edges, pot not moving (check sensor cable)"},
        {1, false, true, false, false, true,
         "CAL FRONT FAIL FLHL no feedback: no hall edges, pot not moving (check sensor cable)"},
        {1, false, false, true, false, true,
         "CAL FRONT FAIL FLHL no feedback: no hall edges, pot not moving (check sensor cable)"},
        {1, true, false, false, false, false,
         "CAL FRONT FAIL FLHL no feedback: no hall edges, pot not moving (check sensor cable)"},
        {2, true, false, false, true, false,
         "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry"},
    };
    for (const DeadCase &dead : kCases)
    {
        krabby_host_reset();
        g_log.clear();
        g_other.clear();
        Timeline timeline;
        timeline.quadrature[0] = true;
        timeline.setHallRods(0, 0);
        Rod &rod = timeline.rods[0][dead.joint];
        rod.deadA = dead.deadA;
        rod.deadB = dead.deadB;
        if (dead.unplugged) rod.sensor = RodSensor::None;
        rod.startAtCurlEnd = dead.parked;
        timeline.requestedPwm = 128;
        bootSingleBoard(timeline);

        if (dead.strict)
            startStrictRun(timeline, 128);
        else
            writeUsb("C\n");
        assertRunUntil(timeline, dead.failLine, 300000);
        timeline.forbidDrives();
        runFor(timeline, 5000);

        assertNoViolations(timeline);
        std::vector<ExpectedMove> expected;
        expected.push_back({0, 2, kSpecKneeCurlSign});
        if (dead.joint == 1) appendNudged(expected, 0, 1);
        assertMoves(timeline, expected);
        TEST_ASSERT_TRUE(findLineStarting("CAL FRONT HOME FLHL") == nullptr);
        if (dead.joint == 1)
        {
            TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][1].isAtCurlEnd(1), dead.failLine);
            // Blind: the nudge stopped at its 1 s blind bound (its drive began
            // one 20 ms ramp step after the move), and the curl back still
            // drove at least as long before the failure.
            const Move &nudge = timeline.moves[2];
            const Move &back = timeline.moves[3];
            const uint32_t nudgeMs = nudge.endMs - nudge.startMs;
            TEST_ASSERT_TRUE_MESSAGE(nudgeMs >= kSpecHipNudgeBlindMs - 40 && nudgeMs <= kSpecHipNudgeBlindMs,
                std::to_string(nudgeMs).c_str());
            TEST_ASSERT_TRUE(back.endMs - back.startMs >= nudge.endMs - nudge.startMs);
        }
        TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
        assertAllStoppedWithoutTargets(timeline, 1);
    }
}

// The default path (the C command) with hall A dead on a stored knee that
// behaves as measured on the robot: it needs 0.34-0.75 s of drive to break free
// of its curl end. No edge is counted, so its nudge runs blind to its 1 s cap
// (an earlier no-edge stop would stop every healthy stored knee before it breaks
// free); it is driven straight back and the pass fails NoFeedback before its
// hip moves. At the nudge's PWM 128 cap, breaking free late it moves about 18
// edges (at PWM 80 it was about 11, inside the healthy 16-edge bound);
// breaking free early (here parked at its PWM 80 end) it moves about 48 and
// ends about 57 edges out: the blind worst case, past the healthy 16/25-edge
// bounds but inside the blind ones (49 / 58).
static void test_single_board_dead_hall_stored_knee_nudge_runs_to_its_cap_then_fails()
{
    struct BlindCase
    {
        uint32_t breakawayMs;
        int parkPwm;
    };
    for (const BlindCase &blind : {BlindCase{750, 200}, BlindCase{340, 80}, BlindCase{340, 200}})
    {
        const std::string label = "break-free " + std::to_string(blind.breakawayMs) + " ms, park PWM " +
            std::to_string(blind.parkPwm);
        krabby_host_reset();
        g_log.clear();
        g_other.clear();
        Timeline timeline;
        timeline.quadrature[0] = true;
        timeline.setHallRods(0, 0);
        makeRobotLikeKnee(timeline.rods[0][2]);
        timeline.rods[0][2].deadA = true;
        timeline.rods[0][2].breakawayMs = blind.breakawayMs;
        timeline.rods[0][2].parkPwm = blind.parkPwm;
        timeline.requestedPwm = 128;
        bootSingleBoard(timeline);

        writeUsb("C\n");
        assertRunUntil(timeline,
            "CAL ROBOT FAIL FRONT: FLKL no feedback: no hall edges, pot not moving (check sensor cable)",
            300000);
        timeline.forbidDrives();
        runFor(timeline, 5000);

        assertNoViolations(timeline);
        std::vector<ExpectedMove> expected;
        appendNudged(expected, 0, 2);
        assertMoves(timeline, expected);
        const Move &nudge = timeline.moves[1];
        const Move &back = timeline.moves[2];
        TEST_ASSERT_TRUE_MESSAGE(nudge.endMs - nudge.startMs <= kSpecKneeNudgeMaxMs, label.c_str());
        TEST_ASSERT_TRUE_MESSAGE(nudge.endMs - nudge.startMs >= kSpecKneeNudgeMaxMs - 50,
            (label + ": " + std::to_string(nudge.endMs - nudge.startMs)).c_str());
        const double moved = moveEdges(timeline, nudge);
        const double endedOut = std::fabs(back.startPosition - nudge.startPosition);
        TEST_ASSERT_TRUE_MESSAGE(moved > kSpecKneeNudgeMaxEdges && moved <= kSpecBlindKneeNudgeMaxEdges,
            (label + ": " + std::to_string(moved)).c_str());
        TEST_ASSERT_TRUE_MESSAGE(endedOut > kSpecKneeNudgeMaxEdges + kSpecKneeSpringMaxEdges &&
                                     endedOut <= kSpecBlindKneeNudgeMaxEdges + kSpecKneeSpringMaxEdges,
            (label + ": " + std::to_string(endedOut)).c_str());
        if (blind.breakawayMs >= 700)
            TEST_ASSERT_TRUE_MESSAGE(moved <= 20.0, (label + ": " + std::to_string(moved)).c_str());
        else
            TEST_ASSERT_TRUE_MESSAGE(moved > 40.0, (label + ": " + std::to_string(moved)).c_str());
        // Out of where it rested: the drive, then the spring-out.
        TEST_ASSERT_TRUE_MESSAGE(endedOut > moved + 5, (label + ": " + std::to_string(endedOut)).c_str());
        TEST_ASSERT_TRUE(back.endMs - back.startMs >= nudge.endMs - nudge.startMs);
        TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][2].isAtCurlEnd(2), label.c_str());
        TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLHL curl"));
        TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
        assertAllStoppedWithoutTargets(timeline, 1);
    }
}

// Curl only (C SEEK) at the default PWM 128 on a single board whose stored
// knees behave as measured (FLKL, 2026-11-23) across the measured spring-out
// (5-9 edges), the break-free (0.34-0.75 s) and both parks (a PWM 200 curl, 17
// edges deeper than PWM 80 reaches, or the PWM 80 end). Every nudge stays
// inside its bounds. At PWM 80 a PWM 200 park with a spring at the low end came
// back fewer than 2 edges (CAL_KNEE_RETURN_MIN_EDGES) and failed "curled back
// only" once; at 128 the curl back reaches a curl end only about 7 edges
// shallower than the PWM 200 park, so every knee is accepted on its first
// Curl only. A knee that did fail would rest at its curl end and pass the next
// Curl only (each knee fails at most once). Never "stopped N short": that knee
// is inside the 40-edge tolerance.
static void test_single_board_robot_like_stored_knees_pass_curl_only_at_every_measured_spring()
{
    int retried = 0;
    for (int parkPwm : {200, 80})
        for (uint32_t breakawayMs : {340U, 750U})
            for (int spring = 5; spring <= 9; ++spring)
            {
                const std::string label = "park PWM " + std::to_string(parkPwm) + ", break-free " +
                    std::to_string(breakawayMs) + " ms, spring " + std::to_string(spring);
                krabby_host_reset();
                g_log.clear();
                g_other.clear();
                Timeline timeline;
                timeline.quadrature[0] = true;
                for (int leg = 0; leg < 2; ++leg)
                {
                    makeRobotLikeHip(timeline.rods[0][3 * leg + 1]);
                    Rod &knee = timeline.rods[0][3 * leg + 2];
                    makeRobotLikeKnee(knee);
                    knee.parkPwm = parkPwm;
                    knee.breakawayMs = breakawayMs;
                    knee.springOutEdges = spring;
                }
                timeline.requestedPwm = 128;
                bootSingleBoard(timeline);

                // Runs a C SEEK to its CAL ROBOT DONE or FAIL; returns the FAIL line, or "".
                const auto runCurlOnly = [&]() {
                    g_log.clear();
                    writeUsb("C SEEK\n");
                    for (int step = 0; step < 3000; ++step)
                    {
                        if (findLineStarting("CAL ROBOT DONE") != nullptr) return std::string();
                        if (const LoggedLine *fail = findLineStarting("CAL ROBOT FAIL")) return fail->text;
                        runFor(timeline, 100);
                    }
                    return std::string("no end");
                };
                std::string failed = runCurlOnly();
                std::string failedKnees;
                for (int attempt = 0; attempt < 3 && !failed.empty(); ++attempt)
                {
                    TEST_ASSERT_TRUE_MESSAGE(parkPwm == 200 && spring <= 7, (label + ": " + failed).c_str());
                    TEST_ASSERT_TRUE_MESSAGE(failed.find(" curled back only ") != std::string::npos,
                        (label + ": " + failed).c_str());
                    const std::string knee = failed.substr(std::string("CAL ROBOT FAIL FRONT: ").size(), 4);
                    TEST_ASSERT_TRUE_MESSAGE(knee == "FLKL" || knee == "FRKL", (label + ": " + failed).c_str());
                    // Each knee fails at most once: it then rests at its curl end.
                    TEST_ASSERT_TRUE_MESSAGE(failedKnees.find(knee) == std::string::npos,
                        (label + ": " + failed).c_str());
                    failedKnees += knee;
                    // That knee's hip never moved in this run.
                    TEST_ASSERT_FALSE_MESSAGE(hasLine(knee == "FLKL" ? "CAL FRONT MOVE FLHL curl"
                                                                     : "CAL FRONT MOVE FRHL curl"),
                        label.c_str());
                    ++retried;
                    runFor(timeline, 1000);
                    failed = runCurlOnly();
                }
                TEST_ASSERT_TRUE_MESSAGE(failed.empty(), (label + " retry: " + failed).c_str());
                TEST_ASSERT_TRUE_MESSAGE(findLineStarting("CAL ROBOT DONE") != nullptr, label.c_str());
                runFor(timeline, 1000);

                assertNoViolations(timeline);
                TEST_ASSERT_EQUAL_INT_MESSAGE(0, countLinesStarting("CAL FRONT FAIL "), label.c_str());
                TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
                for (int leg = 0; leg < 2; ++leg)
                {
                    TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][3 * leg + 1].isAtCurlEnd(3 * leg + 1), label.c_str());
                    TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][3 * leg + 2].isAtCurlEnd(3 * leg + 2), label.c_str());
                }
                // Every knee nudge drive stopped within its 16-edge bound.
                for (const Move &move : timeline.moves)
                {
                    if (move.joint % 3 != 2 || move.sign == kSpecKneeCurlSign) continue;
                    TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, move) <= kSpecKneeNudgeMaxEdges,
                        (label + ": " + std::to_string(moveEdges(timeline, move))).c_str());
                }
                assertAllStoppedWithoutTargets(timeline, 1);
            }
    // At PWM 128 no stored knee needs the retry (see above).
    TEST_ASSERT_EQUAL_INT(0, retried);
}

// A dead motor draws no current: on a strict run (options 0) a knee fails
// NoMotorCurrent at once; on the default path a knee, and always a hip, is
// nudged and driven back first, then fails from the nudge's current evidence.
static void test_single_board_dead_motor_fails_no_motor_current()
{
    struct DeadMotorCase
    {
        int joint;
        bool strict;
    };
    static const DeadMotorCase kCases[] = {{2, true}, {2, false}, {1, false}};
    for (const DeadMotorCase &dead : kCases)
    {
        const int joint = dead.joint;
        krabby_host_reset();
        g_log.clear();
        g_other.clear();
        Timeline timeline;
        timeline.quadrature[0] = true;
        timeline.setHallRods(0, 0);
        timeline.rods[0][joint].isDeadMotor = true;
        timeline.requestedPwm = 128;
        bootSingleBoard(timeline);

        if (dead.strict)
            startStrictRun(timeline, 128);
        else
            writeUsb("C\n");
        const std::string failLine = std::string("CAL FRONT FAIL ") + kJointNames[0][joint] +
            " no motor current at PWM 128: check motor wiring";
        assertRunUntil(timeline, failLine.c_str(), 300000);
        timeline.forbidDrives();
        runFor(timeline, 5000);

        assertNoViolations(timeline);
        std::vector<ExpectedMove> expected;
        if (joint == 1)
        {
            expected.push_back({0, 2, kSpecKneeCurlSign});
            appendNudged(expected, 0, 1);
        }
        else if (dead.strict)
        {
            expected.push_back({0, 2, kSpecKneeCurlSign});
        }
        else
        {
            appendNudged(expected, 0, 2);
        }
        assertMoves(timeline, expected);
        TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
        assertAllStoppedWithoutTargets(timeline, 1);
    }
}

// Motor leads swapped on a hall knee: its count runs against the drive.
static void test_single_board_reversed_motor_fails_wrong_direction()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][2].motorReversed = true;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline,
        "CAL FRONT FAIL FLKL hall counts the wrong way: check motor or hall A/B wiring", 300000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected = {{0, 2, kSpecKneeCurlSign}};
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE(moveEdges(timeline, timeline.moves[0]) <= 12 + 4);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A saved record (hips [0, 1069]); this time an obstacle stops FLHL's
// table-safe sweep at 420 edges, short of its 840. The detector ends the sweep
// there (a stop: speed collapsed, current up), the hip retracts to its curl
// switch with its knee still parked (nothing else moves: the knee never curls
// while the hip is out), and the pass fails SweepShort naming the shortfall.
// Nothing is saved and the hip's home is no longer trusted (kept only as the
// reference of its later curls).
static void test_single_board_obstacle_on_the_hip_sweep_retracts_then_fails_sweep_short()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][1].speedScale = 0.25;  // about 128 edges/s at PWM 128: it arms before the obstacle
    timeline.requestedPwm = 128;
    writeHallRecord(0, kSpecHipStrokeEdges, 700);
    bootSingleBoard(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("Calibration v2 loaded"), joinedOther().c_str());
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("Calibration joint FLHL extend derived (1069 edges): not switch-verified"),
        joinedOther().c_str());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:e FLKL:e FRHY:? FRHL:e FRKL:e",
        lastSensLine("FRONT").c_str());

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLHL uncurl sweep", 300000);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT PARK FLKL hall=") != nullptr);
    timeline.rods[0][1].obstacleEdges = 420.0;
    assertRunUntil(timeline,
        "CAL ROBOT FAIL FRONT: FLHL stopped 420 edges short of its 840-edge sweep: obstacle? check its home", 300000);
    timeline.forbidDrives();
    runFor(timeline, 5000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSeek(expected, 0);
    std::vector<ExpectedMove> sweep;
    appendSweepLeg(sweep, 0, 0, true);
    expected.insert(expected.end(), sweep.begin(), sweep.begin() + 6);  // up to the hip's sweep
    expected.push_back({0, 1, kSpecHipCurlSign});                       // its retract
    assertMoves(timeline, expected);
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT WARN FLHL stopped at 420 edges: retracting"), joinedLog().c_str());
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLHL curl retract"));
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL uncurl hall=420 travel=420 by=stop ") != nullptr);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=0 travel=-420 by=switch ") != nullptr);
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT WARN FLHL left out"));
    // The hip is back at its curl switch, its knee still parked; it went no
    // further than the obstacle.
    TEST_ASSERT_TRUE(timeline.rods[0][1].isAtCurlEnd(1));
    TEST_ASSERT_TRUE(timeline.isKneeInParkBand(0, 2));
    TEST_ASSERT_TRUE(timeline.furthestOutEdges(0, 1) <= 420.0 + 1e-6);
    TEST_ASSERT_EQUAL_INT(1, timeline.hipObstaclesChecked());
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:x FLKL:E FRHY:? FRHL:E FRKL:E",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A single board whose hips and knees behave as measured on the robot
// (FLKL and FLHL, 2026-11-23/24): stored curled, the knees by a PWM 200 curl; a
// knee needs 0.77 s to break free, springs out 9 edges after its nudge and
// comes back only to its calibration-PWM curl end, shallower than where it
// rested; with its knee curled a hip's tibia meets the body at 941 edges. C at
// PWM 128 calibrates it table-safe: each knee nudge ends 18-25 edges out and
// its short return is accepted; each knee parks inside its band before its hip
// sweeps 840 edges plus coast (never near the body or its extend switch), the
// hip re-homes on its curl switch, and the board saves the hips' [0, 1069] and
// the knees' strokes. Every limit-switch end shows the switch-open signature.
static void test_single_board_robot_like_actuators_calibrate_without_touching_the_body()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    for (int leg = 0; leg < 2; ++leg)
    {
        makeRobotLikeHip(timeline.rods[0][3 * leg + 1]);
        makeRobotLikeKnee(timeline.rods[0][3 * leg + 2]);
    }
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 900000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    for (int leg = 0; leg < 2; ++leg)
    {
        appendNudged(expected, 0, 3 * leg + 2);
        appendNudged(expected, 0, 3 * leg + 1);
        expected.push_back({0, 3 * leg + 2, kSpecKneeCurlSign});
    }
    appendSweep(expected, 0, true);
    assertMoves(timeline, expected);
    for (const size_t index : {size_t(1), size_t(8)})
    {
        // The knee nudge: the drive stops by its 1 s cap and 12 edges, then the
        // spring-out; the curl back that follows moves only a few edges.
        const Move &nudge = timeline.moves[index];
        const Move &back = timeline.moves[index + 1];
        TEST_ASSERT_TRUE(nudge.endMs - nudge.startMs <= kSpecKneeNudgeMaxMs);
        TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, nudge) <= kSpecKneeNudgeMaxEdges,
            std::to_string(moveEdges(timeline, nudge)).c_str());
        const double endedOut = std::fabs(back.startPosition - nudge.startPosition);
        TEST_ASSERT_TRUE_MESSAGE(endedOut >= 18.0 && endedOut <= kSpecKneeNudgeMaxEdges + kSpecKneeSpringMaxEdges,
            std::to_string(endedOut).c_str());
        TEST_ASSERT_TRUE_MESSAGE(moveEdges(timeline, back) >= 2.0 && moveEdges(timeline, back) < 12.0,
            std::to_string(moveEdges(timeline, back)).c_str());
    }
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT FAIL "));
    for (int hip : {1, 4})
    {
        // Out 840 edges plus coast (2) and one tick of travel, far from the body
        // (941) and the extend switch (1069).
        TEST_ASSERT_TRUE_MESSAGE(timeline.furthestOutEdges(0, hip) <= kSpecHipSweepEdges + 3,
            std::to_string(timeline.furthestOutEdges(0, hip)).c_str());
        TEST_ASSERT_TRUE(timeline.furthestOutEdges(0, hip) >= kSpecHipSweepEdges);
    }
    // Each knee parked inside its band ("PARK <knee> hall=<x> target=-264").
    for (const char *knee : {"FLKL", "FRKL"})
    {
        const LoggedLine *park = findLineStarting(std::string("CAL FRONT PARK ") + knee + " hall=");
        TEST_ASSERT_NOT_NULL_MESSAGE(park, joinedLog().c_str());
        const std::string prefix = std::string("CAL FRONT PARK ") + knee + " hall=";
        const long x = std::strtol(park->text.c_str() + prefix.size(), nullptr, 10);
        TEST_ASSERT_TRUE_MESSAGE(x <= -kSpecKneeParkBandMinEdges && x >= -kSpecKneeParkBandMaxEdges, park->text.c_str());
        TEST_ASSERT_TRUE_MESSAGE(park->text.find(" target=-264 (FL values)") != std::string::npos, park->text.c_str());
    }
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLKL curl park"));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT MOVE FLHL uncurl sweep"));
    TEST_ASSERT_TRUE(hasLine("CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)"));
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    // A knee's stroke: its 776 edges plus the 18 of its PWM 128 compliance.
    assertCanonicalHallRecord(0, 776, 800);
    // The end detector stopped every drive that ran into an end promptly (the
    // timeline's own checks), and the END lines name the signature: a switch
    // end prints its floor and has the switch-open signature.
    const std::string checks = std::to_string(timeline.switchEndsChecked()) + " " +
                               std::to_string(timeline.stopEndsChecked()) + " " +
                               std::to_string(timeline.pressesChecked()) + " peaks " +
                               std::to_string(timeline.switchStopPeakMs()) + " " +
                               std::to_string(timeline.mechanicalStopPeakMs()) + " " +
                               std::to_string(timeline.pressPeakMs()) + "\n" + joinedLog();
    TEST_ASSERT_TRUE_MESSAGE(timeline.switchEndsChecked() >= 4, checks.c_str());  // hip curls back, knee uncurls
    TEST_ASSERT_TRUE_MESSAGE(timeline.stopEndsChecked() >= 2, checks.c_str());  // knee curls back from the park
    TEST_ASSERT_TRUE_MESSAGE(timeline.pressesChecked() >= 8, checks.c_str());  // parked SEEK curls, SWEEP steps 0-1
    // No switch end was decided before its switch-open signature could show
    // (6 flat samples), and none later than its bound (the timeline's check).
    TEST_ASSERT_TRUE_MESSAGE(timeline.switchStopFastestMs() >= calibration_timeline::kSpecSwitchStopMinMs, checks.c_str());
    for (const char *prefix : {"CAL FRONT END FLHL curl hall=", "CAL FRONT END FLKL uncurl hall="})
    {
        bool found = false;
        for (const LoggedLine &line : g_log)
            found = found || (line.text.rfind(prefix, 0) == 0 && line.text.find(" by=switch cur=") != std::string::npos &&
                              line.text.find(" floor=") != std::string::npos);
        TEST_ASSERT_TRUE_MESSAGE(found, prefix);
    }
    // The presses at the curl switch (SWEEP step 1) end on the switch-open signature.
    TEST_ASSERT_TRUE_MESSAGE(findLineStarting("CAL FRONT END FLHL curl hall=0 travel=0 open=") != nullptr,
        joinedLog().c_str());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:E FLKL:E FRHY:? FRHL:E FRKL:E",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A hip whose limit switch never opens (bypassed: at its curl end the motor
// keeps conducting and presses, its current up instead of down to the
// switch-open level) is never accepted at its curl end: the drive presses on
// until it is stuck mid-stroke, the pass fails before the next joint moves (no
// HOME), and the timeline's own end check flags the late stop.
static void test_single_board_hip_curl_without_a_switch_drop_fails_stuck()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][1].speedScale = 0.25;  // about 128 edges/s: a running phase
    timeline.rods[0][1].switchBypassed = true;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline,
        "CAL ROBOT FAIL FRONT: FLHL stuck mid-stroke after 350 edges at PWM 128: raise PWM or check load", 300000);
    timeline.forbidDrives();
    runFor(timeline, 3000);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT FAIL FLHL stuck mid-stroke after 350 edges at PWM 128: raise PWM or check load"));
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("FLHL pwm=0: drove on past its limit switch"), timeline.report().c_str());
    TEST_ASSERT_EQUAL_UINT(1, timeline.violations.size());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=") != nullptr &&
                     findLineStarting("CAL FRONT END FLHL curl hall=")->text.find(" by=stuck ") != std::string::npos);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT HOME FLHL") == nullptr);
    TEST_ASSERT_EQUAL_INT(1, countLines("CAL FRONT MOVE FLKL curl"));  // its second curl never comes
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FRKL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// V1/R1: a hip that stalls mid-stroke for lack of force on its curl (FLHL at
// PWM 80 on 2026-11-24 held 229 edges short of its curl switch; FRHL at 128
// stalled with its current wobbling 17-31 counts, no switch drop) is never
// homed there. Here FLHL's curl back from its 840-edge sweep runs out of force
// 229 edges out: its current rises and wobbles, so the run fails "stuck
// mid-stroke" and keeps the home it proved (V1). A rerun from the stall point
// (a press with no edge: the detector's stuck goes on as a no-motion curl)
// and one whose stall reads flat at the switch-open level
// (the worst case: the detector cannot tell it from an open switch, and takes
// it as a plain stall at the switch) are caught by the kept home: both fail
// "stopped 229 edges short of its curl end". No run sets a HOME there and
// nothing is saved; the hip stays below 560 edges, so its knee may curl.
static void test_single_board_hip_stalled_mid_stroke_is_never_homed_there()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);
    Rod &hip = timeline.rods[0][1];

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLHL uncurl sweep", 300000);
    long home = 0;
    TEST_ASSERT_TRUE(lastHomeCount("FRONT", "FLHL", home));
    const int homeLines = countLinesStarting("CAL FRONT HOME FLHL ");
    hip.curlStallEdges = 229.0;  // from here on its curl runs out of force 229 edges out
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL stuck mid-stroke after ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT FAIL FLHL stuck mid-stroke after 6") != nullptr);
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(hip.edgesOut(1) - 229.0) < 1e-6, std::to_string(hip.edgesOut(1)).c_str());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=229 ") != nullptr &&
                     findLineStarting("CAL FRONT END FLHL curl hall=229 ")->text.find(" by=stuck ") != std::string::npos);
    TEST_ASSERT_EQUAL_INT(homeLines, countLinesStarting("CAL FRONT HOME FLHL "));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:x FLKL:s FRHY:? FRHL:s FRKL:s",
        lastSensLine("FRONT").c_str());

    // Rerun from the stall point: its knee curls back from its park (the hip is
    // 229 edges out, inside the any-knee zone), then the hip presses at its
    // stall, its current up and wobbling: no edge and no switch signature (the
    // detector says stuck), so it goes on as a curl with no motion, judged
    // against the kept home: "stopped 229 edges short", no HOME.
    g_log.clear();
    size_t firstMove = timeline.moves.size();
    writeUsb("C\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    std::vector<ExpectedMove> rerun = {{0, 2, kSpecKneeCurlSign}, {0, 1, kSpecHipCurlSign}};
    assertMoves(timeline, rerun, firstMove);
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT FAIL FLHL stopped 229 edges short of its curl end"), joinedLog().c_str());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=229 travel=0 by=stuck ") != nullptr);
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT HOME FLHL "));

    // The same rerun with the stall reading flat at its switch-open level (a
    // plain stall at the switch for the detector): the kept home names the
    // shortfall again instead of homing the hip there.
    hip.stallLooksOpen = true;
    g_log.clear();
    firstMove = timeline.moves.size();
    writeUsb("C\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    rerun = {{0, 2, kSpecKneeCurlSign}, {0, 1, kSpecHipCurlSign}};
    assertMoves(timeline, rerun, firstMove);
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT FAIL FLHL stopped 229 edges short of its curl end"), joinedLog().c_str());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT END FLHL curl hall=") != nullptr &&
                     findLineStarting("CAL FRONT END FLHL curl hall=")->text.find(" open=") != std::string::npos);
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT HOME FLHL "));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(hip.edgesOut(1) - 229.0) < 1e-6, std::to_string(hip.edgesOut(1)).c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A run that fails with its hip out past 560 (here FLHL's curl back from its
// sweep runs out of force 700 edges out, its knee still parked) leaves the hip
// there. The next C must not curl the knee out of its park with the hip out
// (the stand envelope): the hip is known out by its kept home, so the run
// fails at once, naming the hip and where it is, and nothing moves.
static void test_single_board_rerun_with_the_hip_left_out_moves_nothing()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);
    Rod &hip = timeline.rods[0][1];

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLHL uncurl sweep", 300000);
    hip.curlStallEdges = 700.0;  // from here on its curl runs out of force 700 edges out
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(hip.edgesOut(1) - 700.0) < 1e-6, std::to_string(hip.edgesOut(1)).c_str());
    TEST_ASSERT_TRUE_MESSAGE(findLineStarting("CAL FRONT WARN FLHL left out at 700 edges") != nullptr,
        joinedLog().c_str());

    g_log.clear();
    const size_t firstMove = timeline.moves.size();
    writeUsb("C\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    TEST_ASSERT_EQUAL_UINT(firstMove, timeline.moves.size());
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT FAIL FLHL out at 700 edges: jog it to its curl end first"),
        joinedLog().c_str());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT MOVE ") == nullptr);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A SWEEP run on its own (C SWEEP) straight after boot, FLHL resting 1000
// edges out, beyond the body contact, with FLKL 350 edges out. Nothing is known
// of the hip yet, so the knee's first curl runs: it stops on the body 83 edges
// short of its stop and is homed there. With its hip curled, the knee's second
// curl (step 2) goes 83 edges deeper and re-homes it ("WARN FLKL home moved"),
// so the park counts from the real stop: it lands inside its band and the
// sweep keeps the envelope. No envelope invariant fires.
static void test_single_board_sweep_only_with_the_hip_beyond_the_body_rehomes_the_knee_before_its_park()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    Rod &hip = timeline.rods[0][1];
    Rod &knee = timeline.rods[0][2];
    hip.position = hip.low + 1000.0 / hip.edgesPerCount;
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);
    TEST_ASSERT_TRUE(std::fabs(knee.edgesOut(2) - 350.0) < 1e-6);

    writeUsb("C SWEEP\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 600000);
    runFor(timeline, 3000);

    assertNoViolations(timeline);
    std::vector<ExpectedMove> expected;
    appendSweep(expected, 0, true);
    assertMoves(timeline, expected);
    // The first knee curl stopped on the body, 24 + 59 edges out.
    const Move &first = timeline.moves[0];
    const double firstOut = (knee.high - first.endPosition) * knee.edgesPerCount;
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(firstOut - 83.0) <= 2.0, std::to_string(firstOut).c_str());
    const LoggedLine *moved = findLineStarting("CAL FRONT WARN FLKL home moved ");
    TEST_ASSERT_NOT_NULL_MESSAGE(moved, joinedLog().c_str());
    const long movedBy = std::strtol(moved->text.c_str() + std::string("CAL FRONT WARN FLKL home moved ").size(), nullptr, 10);
    TEST_ASSERT_TRUE_MESSAGE(movedBy >= 81 && movedBy <= 85, moved->text.c_str());
    // The re-home came on the third move (step 2), before the knee's uncurl and park.
    const LoggedLine *uncurl = findLine("CAL FRONT MOVE FLKL uncurl");
    TEST_ASSERT_NOT_NULL(uncurl);
    TEST_ASSERT_TRUE(moved->ms <= uncurl->ms);
    const LoggedLine *park = findLineStarting("CAL FRONT PARK FLKL hall=");
    TEST_ASSERT_NOT_NULL_MESSAGE(park, joinedLog().c_str());
    const long x = std::strtol(park->text.c_str() + std::string("CAL FRONT PARK FLKL hall=").size(), nullptr, 10);
    TEST_ASSERT_TRUE_MESSAGE(x <= -kSpecKneeParkBandMinEdges && x >= -kSpecKneeParkBandMaxEdges, park->text.c_str());
    TEST_ASSERT_TRUE_MESSAGE(timeline.furthestOutEdges(0, 4) <= kSpecHipSweepEdges + kSpecHipSweepCoastEdges,
        std::to_string(timeline.furthestOutEdges(0, 4)).c_str());
    TEST_ASSERT_EQUAL_UINT(1, timeline.eepromChanges.size());
    assertCanonicalHallRecord(0, 698, 702);
    TEST_ASSERT_EQUAL_INT(0, countLinesStarting("CAL FRONT FAIL "));
    assertAllStoppedWithoutTargets(timeline, 1);
}

// A hip whose home a T guard cleared while it was out past 560: FLHL, homed by
// a C SEEK with a saved record, is sent by T toward 0.9 of its stroke and
// blocked 700 edges out. The T stall guard stops it and marks its home lost
// (ERR STALL, SENS 'x'), but its frame still reads 700. The next C refuses
// before its knee moves (HipOut), moves nothing, saves nothing and leaves the
// hip's state as it was; no invariant fires.
static void test_single_board_hip_left_out_by_the_t_stall_guard_refuses_the_next_run()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 128;
    writeHallRecord(0, kSpecHipStrokeEdges, 700);
    bootSingleBoard(timeline);
    writeUsb("C SEEK\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 300000);
    runFor(timeline, 500);
    assertNoViolations(timeline);
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:E FLKL:E FRHY:? FRHL:E FRKL:E", lastSensLine("FRONT").c_str());

    // T is not calibration: the position loop drives, the envelope is not its job.
    timeline.requestedPwm = 0;
    Rod &hip = timeline.rods[0][1];
    hip.obstacleEdges = 700.0;
    g_other.clear();
    writeUsb("T FLHL 0.900\n");
    runFor(timeline, 6000);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("ERR STALL FLHL"), joinedOther().c_str());
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(hip.edgesOut(1) - 700.0) < 1e-6, std::to_string(hip.edgesOut(1)).c_str());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:x FLKL:E FRHY:? FRHL:E FRKL:E", lastSensLine("FRONT").c_str());

    timeline.requestedPwm = 128;
    g_log.clear();
    timeline.resetEepromTracking();
    const size_t firstMove = timeline.moves.size();
    writeUsb("C\n");
    runUntilLineStarting(timeline, "CAL ROBOT FAIL FRONT: FLHL ", 300000);
    runFor(timeline, 2000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE_MESSAGE(hasLine("CAL FRONT FAIL FLHL out at 700 edges: jog it to its curl end first"),
        joinedLog().c_str());
    TEST_ASSERT_EQUAL_UINT(firstMove, timeline.moves.size());
    TEST_ASSERT_TRUE(findLineStarting("CAL FRONT MOVE ") == nullptr);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:x FLKL:E FRHY:? FRHL:E FRKL:E", lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// F2: a joint that starts 15-200 edges from its curl end reaches it within
// about 0.8-4 s, so the end detector arms in the end approach or not at all.
// With a compliant knee end approached over its last 30 edges (speed falling,
// current rising, as the robot's at 120-150) and hips as far out, a Curl-only
// run at PWM 128 homes every joint: none fails "stuck mid-stroke", and an end
// reached within 1.2 s stops within kSpecEarlyEndStopMaxMs of reaching it.
static void test_single_board_short_curls_into_their_ends_home_every_joint()
{
    int early = 0;
    for (double out : {15.0, 25.0, 40.0, 60.0, 80.0, 100.0, 130.0, 200.0})
    {
        const std::string label = "start " + std::to_string(static_cast<int>(out)) + " edges out";
        krabby_host_reset();
        g_log.clear();
        g_other.clear();
        Timeline timeline;
        timeline.quadrature[0] = true;
        for (int leg = 0; leg < 2; ++leg)
        {
            Rod &hip = timeline.rods[0][3 * leg + 1];
            makeRobotLikeHip(hip);
            hip.startAtCurlEnd = false;
            hip.position = hip.low + out;
            Rod &knee = timeline.rods[0][3 * leg + 2];
            makeRobotLikeKnee(knee);
            knee.startAtCurlEnd = false;
            knee.taperEdges = 30.0;
            knee.position = knee.high - out;
        }
        timeline.requestedPwm = 128;
        bootSingleBoard(timeline);

        writeUsb("C SEEK\n");
        bool done = false;
        for (int step = 0; step < 3000 && !done; ++step)
        {
            TEST_ASSERT_NULL_MESSAGE(findLineStarting("CAL ROBOT FAIL"), (label + ": " + joinedLog()).c_str());
            done = findLineStarting("CAL ROBOT DONE") != nullptr;
            runFor(timeline, 100);
        }
        TEST_ASSERT_TRUE_MESSAGE(done, label.c_str());
        runFor(timeline, 1000);
        TEST_ASSERT_TRUE_MESSAGE(timeline.violations.empty(), (label + ": " + timeline.report()).c_str());
        TEST_ASSERT_EQUAL_INT_MESSAGE(0, countLinesStarting("CAL FRONT FAIL "), label.c_str());
        for (int joint : {1, 2, 4, 5})
        {
            TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][joint].isAtCurlEnd(joint), label.c_str());
            const std::string home = std::string("CAL FRONT HOME ") + calibration_timeline::kJointNames[0][joint] +
                                     " count=";
            TEST_ASSERT_NOT_NULL_MESSAGE(findLineStarting(home.c_str()), (label + ": " + joinedLog()).c_str());
        }
        early += timeline.earlyEndsChecked();
        assertAllStoppedWithoutTargets(timeline, 1);
    }
    // The hips started 15 and 25 edges out reach their switch within 1.2 s.
    TEST_ASSERT_TRUE_MESSAGE(early >= 4, std::to_string(early).c_str());
}

// The body-contact check itself: with its knee curled, a hip driven all the way
// out presses the curled tibia into the body at 941 edges (fact 2), which is a
// violation. A full hip uncurl, as calibration did before the fixed sweep,
// would trip it.
static void test_timeline_body_contact_invariant_catches_a_full_hip_uncurl()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    makeRobotLikeHip(timeline.rods[0][1]);
    makeRobotLikeKnee(timeline.rods[0][2]);
    bootSingleBoard(timeline);
    TEST_ASSERT_TRUE(timeline.rods[0][2].isAtCurlEnd(2));
    timeline.setOperatorJogs(true);
    jog(timeline, "FLHL", 80, 25000);  // about 45 edges/s
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("hip drove its curled tibia into the body"),
        timeline.report().c_str());
    TEST_ASSERT_TRUE_MESSAGE(
        std::fabs(timeline.furthestOutEdges(0, 1) - calibration_timeline::kBodyContactEdges) <= 0.5,
        std::to_string(timeline.furthestOutEdges(0, 1)).c_str());
}

// The table-safe envelope checks themselves, on calibration-paced drives sent
// as jogs (not exempt here, unlike operator jogs between runs): a hip driven
// above 560 edges with its knee not parked, past its 840-edge sweep plus coast
// and onto its extend switch; a hip leaving its curl end with its knee not
// parked (its nudge) above PWM 128; a parked knee driven out of its band with
// its hip above 560. A hip driven above 560 with its knee parked is none of these.
static void test_timeline_envelope_invariant_catches_unsafe_hip_and_knee_moves()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][1].startAtCurlEnd = true;  // FLHL at its curl switch, FLKL mid-stroke (350 out)
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    jog(timeline, "FLHL", 128, 3000);  // about 512 edges/s: onto its extend switch
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("FLHL pwm=128: hip above 560 edges with its knee outside its park band"),
        timeline.report().c_str());
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("hip beyond its 840-edge sweep"), timeline.report().c_str());
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("hip at its extend switch on a solid stand"), timeline.report().c_str());
    TEST_ASSERT_FALSE(timeline.hasViolation("nudge PWM above 128"));
    jog(timeline, "FLHL", -128, 3000);
    TEST_ASSERT_TRUE(timeline.rods[0][1].isAtCurlEnd(1));

    timeline.violations.clear();
    timeline.requestedPwm = 150;
    jog(timeline, "FLHL", 150, 30);  // a nudge out of its curl end at the C pwm
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("nudge PWM above 128"), timeline.report().c_str());
    jog(timeline, "FLHL", -128, 500);
    timeline.requestedPwm = 128;

    // FLKL parked 264 edges out of its stop, FLHL driven to 700 edges: inside
    // the envelope; then the knee curls out of its band with the hip there.
    Rod &knee = timeline.rods[0][2];
    knee.position = knee.high - calibration_timeline::kSpecKneeParkEdges / knee.edgesPerCount;
    timeline.advancePlant(0);
    TEST_ASSERT_TRUE(timeline.isKneeInParkBand(0, 2));
    timeline.violations.clear();
    jog(timeline, "FLHL", 128, 1400);
    TEST_ASSERT_TRUE_MESSAGE(timeline.rods[0][1].edgesOut(1) > 600.0 && timeline.rods[0][1].edgesOut(1) < 840.0,
        std::to_string(timeline.rods[0][1].edgesOut(1)).c_str());
    TEST_ASSERT_FALSE_MESSAGE(timeline.hasViolation("hip above 560"), timeline.report().c_str());
    TEST_ASSERT_FALSE_MESSAGE(timeline.hasViolation("hip beyond"), timeline.report().c_str());
    jog(timeline, "FLKL", 60, 300);  // toward its stop: about 72 edges
    TEST_ASSERT_TRUE_MESSAGE(timeline.hasViolation("FLKL pwm=60: knee left its park band with its hip above 560 edges"),
        timeline.report().c_str());
}

// H while a parked hip is being nudged: everything stops at once, nothing
// moves after, nothing is saved, and homes proven before the abort are kept.
static void test_single_board_hold_during_a_hip_nudge_stops_everything()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][1].startAtCurlEnd = true;  // FLHL parked curled
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLHL nudge", 300000);
    runFor(timeline, 100);
    TEST_ASSERT_EQUAL_INT(1, timeline.drivenCount());
    writeUsb("H\n");
    tick(timeline);
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT ABORTED H"));
    TEST_ASSERT_TRUE(hasLine("CAL ROBOT ABORTED H"));
    const Move &nudge = timeline.moves.back();
    TEST_ASSERT_EQUAL_INT(1, nudge.joint);
    TEST_ASSERT_EQUAL_INT(-kSpecHipCurlSign, nudge.sign);
    TEST_ASSERT_TRUE(moveEdges(timeline, nudge) <= kSpecHipNudgeEdges);
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    // FLKL keeps the home its curl proved (no stroke saved yet: 's').
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:? FLKL:s FRHY:? FRHL:? FRKL:?",
        lastSensLine("FRONT").c_str());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// H while a parked knee is being nudged (the default path): everything stops
// at once, the knee is left within its nudge bound, nothing moves after (its
// hip is never driven) and nothing is saved.
static void test_single_board_hold_during_a_knee_nudge_stops_everything()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.rods[0][2].startAtCurlEnd = true;  // FLKL parked curled
    timeline.requestedPwm = 128;
    bootSingleBoard(timeline);

    writeUsb("C\n");
    assertRunUntil(timeline, "CAL FRONT MOVE FLKL nudge", 300000);
    runFor(timeline, 100);
    TEST_ASSERT_EQUAL_INT(1, timeline.drivenCount());
    writeUsb("H\n");
    tick(timeline);
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    timeline.forbidDrives();
    runFor(timeline, 10000);

    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(hasLine("CAL FRONT ABORTED H"));
    TEST_ASSERT_TRUE(hasLine("CAL ROBOT ABORTED H"));
    std::vector<ExpectedMove> expected = {{0, 2, kSpecKneeCurlSign}, {0, 2, -kSpecKneeCurlSign}};
    assertMoves(timeline, expected);
    const Move &nudge = timeline.moves.back();
    TEST_ASSERT_TRUE(nudge.endMs - nudge.startMs <= kSpecKneeNudgeMaxMs);
    TEST_ASSERT_TRUE(moveEdges(timeline, nudge) <= kSpecKneeNudgeMaxEdges);
    TEST_ASSERT_FALSE(hasLine("CAL FRONT MOVE FLHL curl"));
    TEST_ASSERT_TRUE(timeline.eepromChanges.empty());
    assertAllStoppedWithoutTargets(timeline, 1);
}

// With a saved stroke (a hip's [0, 1069]) but no home this boot, T is refused
// (ERR line, no motion). After a SEEK the hip is ready and T settles near its
// target, half its stroke.
static void test_single_board_trajectory_waits_for_a_home_then_settles()
{
    Timeline timeline;
    timeline.quadrature[0] = true;
    timeline.setHallRods(0, 0);
    timeline.requestedPwm = 128;
    writeHallRecord(0, kSpecHipStrokeEdges, 700);
    bootSingleBoard(timeline);

    writeUsb("T FLHL 0.500\n");
    runFor(timeline, 500);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("ERR T refused FLHL (no position)"), joinedOther().c_str());
    TEST_ASSERT_TRUE(timeline.moves.empty());
    // A second refusal inside a second is not repeated; after it, it is.
    writeUsb("T FLHL 0.500 FLHY 0.500\n");
    runFor(timeline, 100);
    TEST_ASSERT_EQUAL_INT(1, countOtherLines("ERR T refused FLHL (no position)"));
    runFor(timeline, 1000);
    writeUsb("T FLHL 0.500 FLHY 0.500\n");
    runFor(timeline, 100);
    TEST_ASSERT_TRUE_MESSAGE(hasOtherLine("ERR T refused FLHY FLHL (no position)"), joinedOther().c_str());
    TEST_ASSERT_TRUE(timeline.moves.empty());

    writeUsb("C SEEK\n");
    assertRunUntil(timeline, "CAL ROBOT DONE", 300000);
    runFor(timeline, 500);
    assertNoViolations(timeline);
    TEST_ASSERT_EQUAL_STRING("SENS FRONT FLHY:? FLHL:E FLKL:E FRHY:? FRHL:E FRKL:E",
        lastSensLine("FRONT").c_str());
    long home = 0;
    TEST_ASSERT_TRUE(lastHomeCount("FRONT", "FLHL", home));

    // T drives with the position loop, not the calibration ramp.
    timeline.requestedPwm = 0;
    const size_t movesBefore = timeline.moves.size();
    g_other.clear();
    writeUsb("T FLHL 0.500\n");
    runFor(timeline, 4000);
    assertNoViolations(timeline);
    TEST_ASSERT_TRUE(timeline.moves.size() > movesBefore);
    TEST_ASSERT_EQUAL_INT(0, timeline.drivenCount());
    const long x = -(Timeline::hallCount(0, 1) - home);  // the encoder frame
    TEST_ASSERT_INT_WITHIN_MESSAGE(kSpecEncoderSettleEdges, kSpecHipStrokeEdges / 2, x, std::to_string(x).c_str());
    TEST_ASSERT_FALSE(hasOtherLine("ERR STALL FLHL"));
    TEST_ASSERT_FALSE(hasOtherLine("ERR T refused FLHL (no position)"));
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    UNITY_BEGIN();
    RUN_TEST(test_robot_calibration_full_run_keeps_every_invariant_and_saves_each_board);
    RUN_TEST(test_sweep_only_from_mid_stroke_curls_first_and_reversed_pot_fails_the_pass);
    RUN_TEST(test_seek_only_curls_every_leg_and_never_saves);
    RUN_TEST(test_hold_during_front_seek_stops_the_robot);
    RUN_TEST(test_jog_during_left_seek_stops_the_robot);
    RUN_TEST(test_trajectory_during_front_sweep_stops_the_robot_before_any_save);
    RUN_TEST(test_batch_jog_during_right_sweep_keeps_earlier_saves_only);
    RUN_TEST(test_jog_after_abort_still_moves_the_jogged_joint_only);
    RUN_TEST(test_follower_open_pot_fails_the_robot_and_saves_nothing);
    RUN_TEST(test_follower_move_timeout_fails_the_robot_and_keeps_earlier_saves);
    RUN_TEST(test_silent_follower_trips_the_watchdog_and_holds_everything);
    RUN_TEST(test_all_hall_robot_from_mid_stroke_homes_and_saves_canonical_strokes);
    RUN_TEST(test_all_hall_robot_parked_curled_nudges_every_joint_then_saves);
    RUN_TEST(test_mixed_robot_classifies_each_joint_and_saves_its_kind);
    RUN_TEST(test_mixed_firmware_robot_refuses_to_calibrate);
    RUN_TEST(test_calibration_right_after_connecting_waits_for_the_followers_sens_lines);
    RUN_TEST(test_reversed_hall_wiring_fails_wrong_direction_before_the_hip_moves);
    RUN_TEST(test_single_board_rejects_bad_requests_then_runs_at_minimum_pwm);
    RUN_TEST(test_single_board_each_stop_command_aborts_and_nothing_moves_after);
    RUN_TEST(test_single_board_knee_stuck_at_its_curl_end_fails_no_motion);
    RUN_TEST(test_single_board_knee_stuck_at_its_curl_end_is_nudged_then_fails_no_feedback);
    RUN_TEST(test_single_board_joint_that_does_not_move_fails_and_saves_nothing);
    RUN_TEST(test_single_board_hip_that_cannot_curl_back_fails_before_the_knee_moves);
    RUN_TEST(test_single_board_knee_that_does_not_move_mid_stroke_fails_before_its_hip_moves);
    RUN_TEST(test_single_board_hall_knee_that_cannot_curl_is_nudged_then_fails_before_its_hip_moves);
    RUN_TEST(test_single_board_pot_hip_that_cannot_curl_is_nudged_and_fails_short);
    RUN_TEST(test_single_board_parked_curled_strict_needs_knee_jogs_then_saves);
    RUN_TEST(test_single_board_parked_curled_with_knee_nudge_option_homes_every_joint);
    RUN_TEST(test_timeline_knee_nudge_invariant_catches_unbounded_knee_moves);
    RUN_TEST(test_single_board_hall_b_jump_before_the_first_edge_is_not_pot_motion);
    RUN_TEST(test_single_board_rev3_pot_robot_classifies_then_repeats_each_step);
    RUN_TEST(test_single_board_saved_pots_that_count_hall_edges_are_driven_again_under_the_detector);
    RUN_TEST(test_single_board_dead_hall_sensor_fails_without_a_partner_move);
    RUN_TEST(test_single_board_dead_hall_stored_knee_nudge_runs_to_its_cap_then_fails);
    RUN_TEST(test_single_board_robot_like_stored_knees_pass_curl_only_at_every_measured_spring);
    RUN_TEST(test_single_board_dead_motor_fails_no_motor_current);
    RUN_TEST(test_single_board_reversed_motor_fails_wrong_direction);
    RUN_TEST(test_single_board_obstacle_on_the_hip_sweep_retracts_then_fails_sweep_short);
    RUN_TEST(test_single_board_robot_like_actuators_calibrate_without_touching_the_body);
    RUN_TEST(test_single_board_hip_curl_without_a_switch_drop_fails_stuck);
    RUN_TEST(test_single_board_hip_stalled_mid_stroke_is_never_homed_there);
    RUN_TEST(test_single_board_rerun_with_the_hip_left_out_moves_nothing);
    RUN_TEST(test_single_board_sweep_only_with_the_hip_beyond_the_body_rehomes_the_knee_before_its_park);
    RUN_TEST(test_single_board_hip_left_out_by_the_t_stall_guard_refuses_the_next_run);
    RUN_TEST(test_single_board_short_curls_into_their_ends_home_every_joint);
    RUN_TEST(test_timeline_body_contact_invariant_catches_a_full_hip_uncurl);
    RUN_TEST(test_timeline_envelope_invariant_catches_unsafe_hip_and_knee_moves);
    RUN_TEST(test_single_board_hold_during_a_hip_nudge_stops_everything);
    RUN_TEST(test_single_board_hold_during_a_knee_nudge_stops_everything);
    RUN_TEST(test_single_board_trajectory_waits_for_a_home_then_settles);
    return UNITY_END();
}

// Three-board leader loop: USB command relay and follower telemetry forward (arduino.ino loop()).

#include "host_battery_telemetry.hpp"
#include "host_calibration_command.hpp"
#include "host_display.hpp"
#include "host_leader_imu.hpp"
#include "host_multi_board.hpp"

#include "Arduino.h"
#include "actuator_manager.h"
#include "command.h"
#include "hall_hw.h"
#include "host_board_role.hpp"
#include "pin_state.hpp"
#include "sim_serial.hpp"
#include "src/actuator/calibration_command_dispatch.h"
#include "src/actuator/robot_calibration.h"
#include "src/controller/board_role.h"
#include "src/telemetry.h"
#include "version.h"

#include <cstring>
#include <string>

namespace krabby {
namespace sim {
namespace host {

namespace {

constexpr size_t kCmdBufSize = 18;
constexpr size_t kTelemetryLineMax = 512;

bool g_threeBoardLoop = false;
bool g_singleLeaderEighteen = false;
bool g_emitLeftFollowerTelemetry = true;
bool g_emitRightFollowerTelemetry = true;
unsigned long g_lastTelemetryMs = 0;
bool g_setupDone = false;

Command g_cmdBuf[kCmdBufSize];

char g_leftPartial[kTelemetryLineMax];
char g_rightPartial[kTelemetryLineMax];
size_t g_leftPartialPos = 0;
size_t g_rightPartialPos = 0;

LinearActuator g_flhy("FLHY", 2, 3, 22, 6, 0, 0);
LinearActuator g_flhl("FLHL", 4, 5, 24, 7, 1, 1);
LinearActuator g_flkl("FLKL", 6, 7, 26, 8, 2, 2);
LinearActuator g_frhy("FRHY", 8, 9, 23, 9, 3, 3);
LinearActuator g_frhl("FRHL", 10, 11, 25, 10, 4, 4);
LinearActuator g_frkl("FRKL", 12, 13, 27, 11, 5, 5);

LinearActuator g_rlhy("RLHY", 2, 3, 22, 6, 0, 0);
LinearActuator g_rlhl("RLHL", 4, 5, 24, 7, 1, 1);
LinearActuator g_rlkl("RLKL", 6, 7, 26, 8, 2, 2);
LinearActuator g_mlhy("MLHY", 8, 9, 23, 9, 3, 3);
LinearActuator g_mlhl("MLHL", 10, 11, 25, 10, 4, 4);
LinearActuator g_mlkl("MLKL", 12, 13, 27, 11, 5, 5);

LinearActuator g_rrhy("RRHY", 2, 3, 22, 6, 0, 0);
LinearActuator g_rrhl("RRHL", 4, 5, 24, 7, 1, 1);
LinearActuator g_rrkl("RRKL", 6, 7, 26, 8, 2, 2);
LinearActuator g_mrhy("MRHY", 8, 9, 23, 9, 3, 3);
LinearActuator g_mrhl("MRHL", 10, 11, 25, 10, 4, 4);
LinearActuator g_mrkl("MRKL", 12, 13, 27, 11, 5, 5);

LinearActuator *g_frontActs[] = {&g_flhy, &g_flhl, &g_flkl, &g_frhy, &g_frhl, &g_frkl};
LinearActuator *g_leftActs[] = {&g_rlhy, &g_rlhl, &g_rlkl, &g_mlhy, &g_mlhl, &g_mlkl};
LinearActuator *g_rightActs[] = {&g_rrhy, &g_rrhl, &g_rrkl, &g_mrhy, &g_mrhl, &g_mrkl};

ActuatorManager g_frontManager(g_frontActs, 6);
ActuatorManager g_leftManager(g_leftActs, 6);
ActuatorManager g_rightManager(g_rightActs, 6);

// Latest "SENS <ROLE> <NAME>:<c> ..." state code per joint, in g_frontActs, g_leftActs,
// g_rightActs order; '\0' until a SENS line has been seen.
constexpr int kSensJointCount = 18;
constexpr size_t kSensLineMax = 256;
char g_sensState[kSensJointCount];

int sensSlotForName(const std::string &name)
{
    for (int i = 0; i < 6; ++i)
    {
        if (String(g_frontActs[i]->getName()) == name) return i;
        if (String(g_leftActs[i]->getName()) == name) return 6 + i;
        if (String(g_rightActs[i]->getName()) == name) return 12 + i;
    }
    return -1;
}

void noteSensLine(const std::string &line)
{
    if (line.compare(0, 5, "SENS ") != 0) return;
    size_t pos = 5;
    while (pos < line.size())
    {
        size_t end = line.find(' ', pos);
        if (end == std::string::npos) end = line.size();
        const std::string token = line.substr(pos, end - pos);
        pos = end + 1;
        // NAME:c. The role token has no colon and is skipped.
        const size_t colon = token.find(':');
        if (colon == std::string::npos || colon + 2 != token.size()) continue;
        const int slot = sensSlotForName(token.substr(0, colon));
        if (slot >= 0) g_sensState[slot] = token[colon + 1];
    }
}

// A board's output port that also records the SENS lines written to it. Every
// manager prints through one of these (g_usb, g_leftUplink, g_rightUplink), so a
// SENS line from any board is seen, and one the leader forwards is seen twice.
class SensCapturingSerial final : public BoardHardwareSerial
{
public:
    using BoardHardwareSerial::BoardHardwareSerial;
    using BoardHardwareSerial::write;

    void write(uint8_t byte) override
    {
        BoardHardwareSerial::write(byte);
        if (byte == '\n')
        {
            noteSensLine(line_);
            line_.clear();
        }
        else if (byte != '\r' && line_.size() < kSensLineMax)
        {
            line_.push_back(static_cast<char>(byte));
        }
    }
    void resetLine() { line_.clear(); }

private:
    std::string line_;
};

SensCapturingSerial g_usb(BoardId::kFront, SerialPortId::kUsb);
BoardHardwareSerial g_frontLeft(BoardId::kFront, SerialPortId::kSerial1);
BoardHardwareSerial g_frontRight(BoardId::kFront, SerialPortId::kSerial2);
SensCapturingSerial g_leftUplink(BoardId::kLeft, SerialPortId::kSerial1);
SensCapturingSerial g_rightUplink(BoardId::kRight, SerialPortId::kSerial2);

RobotCalibration g_robotCalibration;
// A follower board's own robot sequence: as on the robot, it never starts there.
RobotCalibration g_followerRobotCalibration;

// The leader's robot calibration (arduino.ino SketchRobotCalibrationHooks). The
// single-leader-eighteen mode has no follower links, so only FRONT calibrates there.
class LeaderRobotCalibrationHooks final : public RobotCalibrationHooks
{
public:
    uint32_t nowMs() override { return millis(); }
    CalibrationTextSink &out() override { return usb_; }
    const char *localRoleLabel() override
    {
        return boardRoleLabel(g_frontManager.calibrationOutputRole());
    }
    CalibrationTextSink *followerLink(BoardRole role) override
    {
        if (g_singleLeaderEighteen) return nullptr;
        if (role == ROLE_LEFT) return &left_;
        if (role == ROLE_RIGHT) return &right_;
        return nullptr;
    }
    bool startLocalPass(uint8_t pass, int pwm) override
    {
        return g_frontManager.startCalibration(pass, pwm);
    }
    bool isLocalRunning() override { return g_frontManager.isCalibrating(); }
    LegCalibrationOutcome localOutcome() override { return g_frontManager.calibrationOutcome(); }
    void printLocalFailure(CalibrationTextSink &out) override
    {
        g_frontManager.printCalibrationFailure(out);
    }
    void abortLocal(char cause) override { g_frontManager.abortCalibration(cause); }
    void stopLocalMotors() override { g_frontManager.holdAll(); }

private:
    PrintCalibrationSink usb_{&g_usb};
    PrintCalibrationSink left_{&g_frontLeft};
    PrintCalibrationSink right_{&g_frontRight};
};

LeaderRobotCalibrationHooks g_robotCalibrationHooks;

// Runs one board's firmware against that board's EEPROM, and tags its pin writes
// and analog reads with the board (the boards share one pin map in this sim; see
// PinStateModel::setActiveBoard). The single-leader mode is one physical board,
// so everything runs as FRONT there.
class ScopedHostBoard
{
public:
    explicit ScopedHostBoard(BoardId board)
        : previousEeprom_(activeEepromBoard()), previousPins_(pinState().activeBoard())
    {
        const BoardId effective = g_singleLeaderEighteen ? BoardId::kFront : board;
        setActiveEepromBoard(effective);
        pinState().setActiveBoard(static_cast<int>(effective));
    }
    ~ScopedHostBoard()
    {
        setActiveEepromBoard(previousEeprom_);
        pinState().setActiveBoard(previousPins_);
    }
    ScopedHostBoard(const ScopedHostBoard &) = delete;
    ScopedHostBoard &operator=(const ScopedHostBoard &) = delete;

private:
    BoardId previousEeprom_;
    int previousPins_;
};

// This board's robot sequence: the leader's on FRONT, an idle one on a follower.
RobotCalibration &robotCalibrationFor(BoardRole role)
{
    return isCalibrationLeaderRole(role) ? g_robotCalibration : g_followerRobotCalibration;
}

// H, J, B and T end any calibration before they run (the sketch's glue).
void abortCalibrationForCommand(BoardRole role, ActuatorManager &manager, char command)
{
    abortBoardCalibrationForCommand(robotCalibrationFor(role), g_robotCalibrationHooks, manager, command);
}

// The sketch's C dispatch: the leader runs the robot, a follower its own passes.
// C PWR_SENSE is a no-op here (no power monitors).
void handleCalibrationLine(
    BoardRole role,
    BoardHardwareSerial &port,
    ActuatorManager &manager,
    const String &line)
{
    ActuatorCalibrationRequest request = {CAL_PASS_ALL, CAL_DEFAULT_PWM};
    const ActuatorCalibrationCommandKind kind = parseHostCalibrationLine(line, request);
    PrintCalibrationSink out(&port);
    dispatchActuatorCalibrationCommand(
        role, kind, request, out, robotCalibrationFor(role), g_robotCalibrationHooks, manager);
}

void forwardFullLines(
    BoardHardwareSerial *from,
    BoardHardwareSerial *to,
    char *partial,
    size_t cap,
    size_t *partialPos,
    BoardRole noteForOled = ROLE_UNKNOWN)
{
    if (from == nullptr || to == nullptr || partial == nullptr || partialPos == nullptr)
        return;

    while (from->available())
    {
        const int next = from->read();
        if (next < 0) break;
        const char c = static_cast<char>(next);
        if (c == '\n')
        {
            partial[*partialPos] = '\0';
            if (*partialPos > 0)
            {
                to->println(static_cast<const char *>(partial));
                if (noteForOled != ROLE_UNKNOWN)
                {
                    hostDisplayNoteTelemetryLine(partial, noteForOled);
                    g_robotCalibration.onFollowerLine(g_robotCalibrationHooks, noteForOled, partial);
                }
            }
            *partialPos = 0;
            continue;
        }
        if (c == '\r') continue;
        if (*partialPos < cap - 1)
            partial[(*partialPos)++] = c;
        else
        {
            while (from->available())
            {
                const int d = from->read();
                if (d == '\n' || d == '\r') break;
            }
            *partialPos = 0;
        }
    }
}

template <typename LinkT>
void relayCommandLine(LinkT &link, const char *prefix, const String &payload)
{
    if (prefix != nullptr)
    {
        link.print(prefix);
        link.print(" ");
    }
    link.println(payload);
}

ActuatorManager *managerForActuatorName(const String &name)
{
    for (LinearActuator *actuator : g_frontActs)
    {
        if (String(actuator->getName()) == name) return &g_frontManager;
    }
    for (LinearActuator *actuator : g_leftActs)
    {
        if (String(actuator->getName()) == name) return &g_leftManager;
    }
    for (LinearActuator *actuator : g_rightActs)
    {
        if (String(actuator->getName()) == name) return &g_rightManager;
    }
    return nullptr;
}

void processCommandStream(
    BoardRole role,
    BoardHardwareSerial &port,
    ActuatorManager &manager,
    BoardHardwareSerial *relayLeft,
    BoardHardwareSerial *relayRight)
{
    while (port.available())
    {
        const char cmdType = static_cast<char>(port.peek());
        if (cmdType == 'T')
        {
            port.read();
            abortCalibrationForCommand(role, manager, 'T');
            const String payload = port.readStringUntil('\n');
            const size_t cmdCount = parseCommands(payload, g_cmdBuf, kCmdBufSize);
            manager.applyCommands(g_cmdBuf, cmdCount);
            if (relayLeft != nullptr) relayCommandLine(*relayLeft, "T", payload);
            if (relayRight != nullptr) relayCommandLine(*relayRight, "T", payload);
        }
        else if (cmdType == 'B')
        {
            port.read();
            abortCalibrationForCommand(role, manager, 'B');
            const String payload = port.readStringUntil('\n');
            int i = 0;
            const int len = payload.length();
            while (true)
            {
                const String name = nextTok(payload, i, len);
                const String pwm = nextTok(payload, i, len);
                if (name.length() == 0 || pwm.length() == 0) break;
                manager.handleJog(name, pwm.toInt());
            }
            if (relayLeft != nullptr) relayCommandLine(*relayLeft, "B", payload);
            if (relayRight != nullptr) relayCommandLine(*relayRight, "B", payload);
        }
        else if (cmdType == 'J')
        {
            port.read();
            abortCalibrationForCommand(role, manager, 'J');
            while (port.available() && port.peek() == ' ')
            {
                port.read();
            }
            const String name = port.readStringUntil(' ');
            const int pwm = port.readStringUntil('\n').toInt();
            manager.handleJog(name, pwm);
            if (relayLeft != nullptr)
            {
                relayLeft->print("J");
                relayLeft->print(name);
                relayLeft->print(" ");
                relayLeft->println(pwm);
            }
            if (relayRight != nullptr)
            {
                relayRight->print("J");
                relayRight->print(name);
                relayRight->print(" ");
                relayRight->println(pwm);
            }
        }
        else if (cmdType == 'C')
        {
            port.read();
            handleCalibrationLine(role, port, manager, port.readStringUntil('\n'));
        }
        else if (cmdType == 'H')
        {
            port.read();
            abortCalibrationForCommand(role, manager, 'H');
            port.readStringUntil('\n');
            manager.holdAll();
            if (relayLeft != nullptr) relayLeft->println("H");
            if (relayRight != nullptr) relayRight->println("H");
        }
        else if (cmdType == 'V')
        {
            port.read();
            port.readStringUntil('\n');
            port.print("VER ");
            port.print(KRABBY_FW_VERSION);
            port.print("|");
            port.print("-");
            port.print("|");
            port.print("-");
            port.print(" ");
            port.print(KRABBY_FW_BRANCH);
            port.print("|");
            port.print("-");
            port.print("|");
            port.print("-");
            port.print(" ");
            port.print(KRABBY_FW_COMMIT);
            port.print("|");
            port.print("-");
            port.print("|");
            port.println("-");
        }
        else
        {
            port.readStringUntil('\n');
        }
    }
}

void processSingleLeaderUsbCommands()
{
    BoardHardwareSerial &port = g_usb;
    while (port.available())
    {
        const char cmdType = static_cast<char>(port.peek());
        if (cmdType == 'T')
        {
            port.read();
            abortCalibrationForCommand(ROLE_FRONT, g_frontManager, 'T');
            const String payload = port.readStringUntil('\n');
            const size_t cmdCount = parseCommands(payload, g_cmdBuf, kCmdBufSize);
            for (size_t i = 0; i < cmdCount; i++)
            {
                ActuatorManager *manager = managerForActuatorName(g_cmdBuf[i].name);
                if (manager != nullptr) manager->applyCommands(&g_cmdBuf[i], 1);
            }
        }
        else if (cmdType == 'B')
        {
            port.read();
            abortCalibrationForCommand(ROLE_FRONT, g_frontManager, 'B');
            const String payload = port.readStringUntil('\n');
            int i = 0;
            const int len = payload.length();
            while (true)
            {
                const String name = nextTok(payload, i, len);
                const String pwm = nextTok(payload, i, len);
                if (name.length() == 0 || pwm.length() == 0) break;
                ActuatorManager *manager = managerForActuatorName(name);
                if (manager != nullptr) manager->handleJog(name, pwm.toInt());
            }
        }
        else if (cmdType == 'J')
        {
            port.read();
            abortCalibrationForCommand(ROLE_FRONT, g_frontManager, 'J');
            while (port.available() && port.peek() == ' ')
            {
                port.read();
            }
            const String name = port.readStringUntil(' ');
            const int pwm = port.readStringUntil('\n').toInt();
            ActuatorManager *manager = managerForActuatorName(name);
            if (manager != nullptr) manager->handleJog(name, pwm);
        }
        else if (cmdType == 'C')
        {
            // One USB leader without follower links: only FRONT's joints calibrate.
            port.read();
            handleCalibrationLine(ROLE_FRONT, port, g_frontManager, port.readStringUntil('\n'));
        }
        else if (cmdType == 'H')
        {
            port.read();
            abortCalibrationForCommand(ROLE_FRONT, g_frontManager, 'H');
            port.readStringUntil('\n');
            g_frontManager.holdAll();
            g_leftManager.holdAll();
            g_rightManager.holdAll();
        }
        else if (cmdType == 'V')
        {
            port.read();
            port.readStringUntil('\n');
            port.print("VER ");
            port.print(KRABBY_FW_VERSION);
            port.print("|");
            port.print(KRABBY_FW_VERSION);
            port.print("|");
            port.print(KRABBY_FW_VERSION);
            port.print(" ");
            port.print(KRABBY_FW_BRANCH);
            port.print("|");
            port.print(KRABBY_FW_BRANCH);
            port.print("|");
            port.print(KRABBY_FW_BRANCH);
            port.print(" ");
            port.print(KRABBY_FW_COMMIT);
            port.print("|");
            port.print(KRABBY_FW_COMMIT);
            port.print("|");
            port.println(KRABBY_FW_COMMIT);
        }
        else
        {
            port.readStringUntil('\n');
        }
    }
}

void emitTelemetry(BoardRole role, ActuatorManager &manager, BoardHardwareSerial &port)
{
    port.print(boardTelemetryRoleLabel(role));
    port.print(TELEMETRY_SEGMENT_DELIMITER);
    port.print(TELEMETRY_FIELD_SEPARATOR);
    manager.printTelemetry(port);
    if (role == ROLE_FRONT)
    {
        const ImuMeasurement measurement = leaderImu().measure();
        appendImuMeasurement(port, measurement);
        if (leaderBattery().isOnline())
        {
            krabby::sim::host::appendBatteryTelemetry(port, leaderBattery().sample());
        }
    }
    port.println();
}

}  // namespace

void setThreeBoardFirmwareLoopEnabled(const bool enabled) { g_threeBoardLoop = enabled; }

bool threeBoardFirmwareLoopEnabled() { return g_threeBoardLoop; }

void setSingleLeaderEighteenJointEnabled(const bool enabled) { g_singleLeaderEighteen = enabled; }

bool singleLeaderEighteenJointEnabled() { return g_singleLeaderEighteen; }

void threeBoardFirmwareSetup()
{
    if (g_setupDone) return;
    {
        ScopedHostBoard board(BoardId::kFront);
        hallHwInit();
        g_frontManager.initAll();
        g_frontManager.setCalibrationOutput(&g_usb, ROLE_FRONT);
        g_frontManager.loadCalibration();
        leaderImu().setup();
        leaderBattery().setup();
    }
    {
        ScopedHostBoard board(BoardId::kLeft);
        hallHwInit();
        g_leftManager.initAll();
        g_leftManager.setCalibrationOutput(&g_leftUplink, ROLE_LEFT);
        g_leftManager.loadCalibration();
    }
    {
        ScopedHostBoard board(BoardId::kRight);
        hallHwInit();
        g_rightManager.initAll();
        g_rightManager.setCalibrationOutput(&g_rightUplink, ROLE_RIGHT);
        g_rightManager.loadCalibration();
    }
    g_lastTelemetryMs = 0;
    g_setupDone = true;
}

void threeBoardFirmwareLoopOnce()
{
    if (!g_setupDone) threeBoardFirmwareSetup();

    {
        ScopedHostBoard board(BoardId::kFront);
        processCommandStream(ROLE_FRONT, g_usb, g_frontManager, &g_frontLeft, &g_frontRight);
    }
    {
        ScopedHostBoard board(BoardId::kLeft);
        processCommandStream(ROLE_LEFT, g_leftUplink, g_leftManager, nullptr, nullptr);
    }
    {
        ScopedHostBoard board(BoardId::kRight);
        processCommandStream(ROLE_RIGHT, g_rightUplink, g_rightManager, nullptr, nullptr);
    }

    // The rest of the leader's work (forwarding, robot calibration) runs as FRONT.
    ScopedHostBoard leader(BoardId::kFront);
    forwardFullLines(
        &g_frontLeft,
        &g_usb,
        g_leftPartial,
        kTelemetryLineMax,
        &g_leftPartialPos,
        ROLE_LEFT);
    forwardFullLines(
        &g_frontRight,
        &g_usb,
        g_rightPartial,
        kTelemetryLineMax,
        &g_rightPartialPos,
        ROLE_RIGHT);

    g_frontManager.updateAll();
    {
        ScopedHostBoard board(BoardId::kLeft);
        g_leftManager.updateAll();
    }
    {
        ScopedHostBoard board(BoardId::kRight);
        g_rightManager.updateAll();
    }
    g_robotCalibration.update(g_robotCalibrationHooks);

    forwardFullLines(
        &g_frontLeft,
        &g_usb,
        g_leftPartial,
        kTelemetryLineMax,
        &g_leftPartialPos,
        ROLE_LEFT);
    forwardFullLines(
        &g_frontRight,
        &g_usb,
        g_rightPartial,
        kTelemetryLineMax,
        &g_rightPartialPos,
        ROLE_RIGHT);

    const unsigned long nowMs = millis();
    if (nowMs - g_lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;

    g_lastTelemetryMs = nowMs;
    if (g_emitLeftFollowerTelemetry)
    {
        ScopedHostBoard board(BoardId::kLeft);
        emitTelemetry(ROLE_LEFT, g_leftManager, g_leftUplink);
    }
    if (g_emitRightFollowerTelemetry)
    {
        ScopedHostBoard board(BoardId::kRight);
        emitTelemetry(ROLE_RIGHT, g_rightManager, g_rightUplink);
    }
    emitTelemetry(ROLE_FRONT, g_frontManager, g_usb);

    forwardFullLines(
        &g_frontLeft,
        &g_usb,
        g_leftPartial,
        kTelemetryLineMax,
        &g_leftPartialPos,
        ROLE_LEFT);
    forwardFullLines(
        &g_frontRight,
        &g_usb,
        g_rightPartial,
        kTelemetryLineMax,
        &g_rightPartialPos,
        ROLE_RIGHT);
}

void singleLeaderEighteenFirmwareLoopOnce()
{
    if (!g_setupDone) threeBoardFirmwareSetup();

    processSingleLeaderUsbCommands();

    g_frontManager.updateAll();
    g_leftManager.updateAll();
    g_rightManager.updateAll();
    g_robotCalibration.update(g_robotCalibrationHooks);

    const unsigned long nowMs = millis();
    if (nowMs - g_lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;

    g_lastTelemetryMs = nowMs;
    emitTelemetry(ROLE_FRONT, g_frontManager, g_usb);
    emitTelemetry(ROLE_LEFT, g_leftManager, g_usb);
    emitTelemetry(ROLE_RIGHT, g_rightManager, g_usb);
}

void threeBoardFirmwareResetLoopState()
{
    g_lastTelemetryMs = 0;
    g_setupDone = false;
    g_robotCalibration.reset();
    g_frontManager.resetCalibration();
    g_leftManager.resetCalibration();
    g_rightManager.resetCalibration();
    g_leftPartialPos = 0;
    g_rightPartialPos = 0;
    std::memset(g_sensState, 0, sizeof(g_sensState));
    g_usb.resetLine();
    g_leftUplink.resetLine();
    g_rightUplink.resetLine();
    g_emitLeftFollowerTelemetry = true;
    g_emitRightFollowerTelemetry = true;
}

bool threeBoardCalibrationActive()
{
    return g_robotCalibration.isActive() ||
           g_frontManager.isCalibrating() ||
           g_leftManager.isCalibrating() ||
           g_rightManager.isCalibrating();
}

void setThreeBoardFollowerTelemetryEnabled(const bool leftEnabled, const bool rightEnabled)
{
    g_emitLeftFollowerTelemetry = leftEnabled;
    g_emitRightFollowerTelemetry = rightEnabled;
}

LinearActuator *findThreeBoardActuator(const char *name)
{
    if (name == nullptr) return nullptr;
    for (LinearActuator *actuator : g_frontActs)
    {
        if (String(actuator->getName()) == name) return actuator;
    }
    for (LinearActuator *actuator : g_leftActs)
    {
        if (String(actuator->getName()) == name) return actuator;
    }
    for (LinearActuator *actuator : g_rightActs)
    {
        if (String(actuator->getName()) == name) return actuator;
    }
    return nullptr;
}

char threeBoardSensorState(const char *name)
{
    if (name == nullptr) return '\0';
    const int slot = sensSlotForName(name);
    return slot < 0 ? '\0' : g_sensState[slot];
}

int threeBoardActuatorCommandPwm(const char *name)
{
    LinearActuator *actuator = findThreeBoardActuator(name);
    if (actuator == nullptr) return 0;
    if (digitalRead(actuator->pinEn) != HIGH) return 0;
    return actuator->currentPwm;
}

}  // namespace host
}  // namespace sim
}  // namespace krabby

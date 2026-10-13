// Three-board leader loop: USB command relay and follower telemetry forward (arduino.ino loop()).

#include "host_leader_imu.hpp"
#include "host_multi_board.hpp"

#include "Arduino.h"
#include "actuator_manager.h"
#include "command.h"
#include "hall_hw.h"
#include "sim_serial.hpp"
#include "src/controller/board_role.h"
#include "src/telemetry.h"
#include "version.h"

namespace krabby {
namespace sim {
namespace host {

namespace {

constexpr size_t kCmdBufSize = 18;
constexpr size_t kTelemetryLineMax = 512;

bool g_threeBoardLoop = false;
bool g_singleLeaderEighteen = false;
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

BoardHardwareSerial g_usb(BoardId::kFront, SerialPortId::kUsb);
BoardHardwareSerial g_frontLeft(BoardId::kFront, SerialPortId::kSerial1);
BoardHardwareSerial g_frontRight(BoardId::kFront, SerialPortId::kSerial2);
BoardHardwareSerial g_leftUplink(BoardId::kLeft, SerialPortId::kSerial1);
BoardHardwareSerial g_rightUplink(BoardId::kRight, SerialPortId::kSerial2);

void forwardFullLines(
    BoardHardwareSerial *from,
    BoardHardwareSerial *to,
    char *partial,
    size_t cap,
    size_t *partialPos)
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
            if (*partialPos > 0) to->println(static_cast<const char *>(partial));
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
            const String payload = port.readStringUntil('\n');
            const size_t cmdCount = parseCommands(payload, g_cmdBuf, kCmdBufSize);
            manager.applyCommands(g_cmdBuf, cmdCount);
            if (relayLeft != nullptr) relayCommandLine(*relayLeft, "T", payload);
            if (relayRight != nullptr) relayCommandLine(*relayRight, "T", payload);
        }
        else if (cmdType == 'B')
        {
            port.read();
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
            port.readStringUntil('\n');
            manager.startAutoCalibration();
            if (relayLeft != nullptr) relayLeft->println("C");
            if (relayRight != nullptr) relayRight->println("C");
        }
        else if (cmdType == 'H')
        {
            port.read();
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
            port.read();
            port.readStringUntil('\n');
            g_frontManager.startAutoCalibration();
            g_leftManager.startAutoCalibration();
            g_rightManager.startAutoCalibration();
        }
        else if (cmdType == 'H')
        {
            port.read();
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
    hallHwInit();
    g_frontManager.initAll();
    g_leftManager.initAll();
    g_rightManager.initAll();
    g_frontManager.loadCalibration();
    g_leftManager.loadCalibration();
    g_rightManager.loadCalibration();
    leaderImu().setup();
    g_lastTelemetryMs = 0;
    g_setupDone = true;
}

void threeBoardFirmwareLoopOnce()
{
    if (!g_setupDone) threeBoardFirmwareSetup();

    processCommandStream(g_usb, g_frontManager, &g_frontLeft, &g_frontRight);
    processCommandStream(g_leftUplink, g_leftManager, nullptr, nullptr);
    processCommandStream(g_rightUplink, g_rightManager, nullptr, nullptr);

    forwardFullLines(&g_frontLeft, &g_usb, g_leftPartial, kTelemetryLineMax, &g_leftPartialPos);
    forwardFullLines(&g_frontRight, &g_usb, g_rightPartial, kTelemetryLineMax, &g_rightPartialPos);

    g_frontManager.updateAll();
    g_leftManager.updateAll();
    g_rightManager.updateAll();

    forwardFullLines(&g_frontLeft, &g_usb, g_leftPartial, kTelemetryLineMax, &g_leftPartialPos);
    forwardFullLines(&g_frontRight, &g_usb, g_rightPartial, kTelemetryLineMax, &g_rightPartialPos);

    const unsigned long nowMs = millis();
    if (nowMs - g_lastTelemetryMs < TELEMETRY_INTERVAL_MS) return;

    g_lastTelemetryMs = nowMs;
    emitTelemetry(ROLE_LEFT, g_leftManager, g_leftUplink);
    emitTelemetry(ROLE_RIGHT, g_rightManager, g_rightUplink);
    emitTelemetry(ROLE_FRONT, g_frontManager, g_usb);

    forwardFullLines(&g_frontLeft, &g_usb, g_leftPartial, kTelemetryLineMax, &g_leftPartialPos);
    forwardFullLines(&g_frontRight, &g_usb, g_rightPartial, kTelemetryLineMax, &g_rightPartialPos);
}

void singleLeaderEighteenFirmwareLoopOnce()
{
    if (!g_setupDone) threeBoardFirmwareSetup();

    processSingleLeaderUsbCommands();

    g_frontManager.updateAll();
    g_leftManager.updateAll();
    g_rightManager.updateAll();

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
    g_leftPartialPos = 0;
    g_rightPartialPos = 0;
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

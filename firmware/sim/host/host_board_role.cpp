#include "host_board_role.hpp"

#include "EEPROM.h"
#include "sim_clock.hpp"

namespace krabby {
namespace sim {

namespace {

bool g_front_imu_present[3] = {true, false, false};

class SimLinkSerial
{
public:
    explicit SimLinkSerial(SimSerialPort &port) : port_(port) {}

    int available() { return port_.rxAvailable(); }

    int read() { return port_.rxRead(); }

    String readStringUntil(char terminator) { return port_.readStringUntil(terminator); }

    void println(const char *text)
    {
        if (text == nullptr) return;
        for (const char *p = text; *p != '\0'; ++p)
        {
            port_.txWrite(static_cast<uint8_t>(*p));
        }
        port_.txWrite('\n');
    }

    void println(const String &text) { println(text.c_str()); }

private:
    SimSerialPort &port_;
};

class SimUsbPrint
{
public:
    explicit SimUsbPrint(SimSerialPort &port) : port_(port) {}

    void println(const char *text)
    {
        if (text == nullptr) return;
        for (const char *p = text; *p != '\0'; ++p)
        {
            port_.txWrite(static_cast<uint8_t>(*p));
        }
        port_.txWrite('\n');
    }

private:
    SimSerialPort &port_;
};

BoardRole statusToRole(BoardRoleElectionStatus status)
{
    switch (status)
    {
        case BoardRoleElectionStatus::FollowerLeft: return ROLE_LEFT;
        case BoardRoleElectionStatus::FollowerRight: return ROLE_RIGHT;
        case BoardRoleElectionStatus::LeaderFront: return ROLE_FRONT;
        case BoardRoleElectionStatus::UnknownFrontActuators: return ROLE_UNKNOWN;
        default: return ROLE_UNKNOWN;
    }
}

}  // namespace

void resetHostFrontImuPresence()
{
    g_front_imu_present[0] = true;
    g_front_imu_present[1] = false;
    g_front_imu_present[2] = false;
}

void setHostBoardHasFrontImu(BoardId board, bool present)
{
    g_front_imu_present[static_cast<int>(board)] = present;
}

bool hostBoardHasFrontImu(BoardId board)
{
    return g_front_imu_present[static_cast<int>(board)];
}

void setActiveEepromBoard(BoardId board) { EEPROM.setActiveBoard(board); }

BoardId activeEepromBoard() { return EEPROM.activeBoardId(); }

void hostBoardRoleElectionBegin(HostBoardRoleElectionContext &ctx)
{
    ctx.state = BoardRoleElectionState{};
    ctx.electedRole = ROLE_UNKNOWN;
    ctx.finished = false;
    setActiveEepromBoard(ctx.boardId);
}

bool hostBoardRoleElectionStep(HostBoardRoleElectionContext &ctx)
{
    if (ctx.finished) return true;

    setActiveEepromBoard(ctx.boardId);

    SimUsbPrint usb(serialPort(ctx.boardId, SerialPortId::kUsb));
    SimLinkSerial serialLeft(serialPort(ctx.boardId, SerialPortId::kSerial1));
    SimLinkSerial serialRight(serialPort(ctx.boardId, SerialPortId::kSerial2));

    const BoardRoleElectionStatus status = boardRoleElectionStepOnce(
        ctx.state,
        hostBoardHasFrontImu(ctx.boardId),
        usb,
        serialLeft,
        serialRight);

    if (status == BoardRoleElectionStatus::InProgress) return false;

    ctx.electedRole = statusToRole(status);
    ctx.finished = true;
    return true;
}

void runThreeBoardRoleElectionCooperative(BoardRole rolesOut[3], unsigned long maxSteps)
{
    wireDefaultThreeBoardSerialLinks();

    HostBoardRoleElectionContext contexts[3];
    contexts[0].boardId = BoardId::kFront;
    contexts[1].boardId = BoardId::kLeft;
    contexts[2].boardId = BoardId::kRight;

    for (HostBoardRoleElectionContext &ctx : contexts)
    {
        hostBoardRoleElectionBegin(ctx);
    }

    for (unsigned long step = 0; step < maxSteps; ++step)
    {
        bool allFinished = true;
        for (HostBoardRoleElectionContext &ctx : contexts)
        {
            if (!ctx.finished)
            {
                allFinished = false;
                hostBoardRoleElectionStep(ctx);
            }
        }
        simClock().advance(BOARD_ROLE_SYNC_INTERVAL_MS);
        if (allFinished) break;
    }

    rolesOut[0] = contexts[0].electedRole;
    rolesOut[1] = contexts[1].electedRole;
    rolesOut[2] = contexts[2].electedRole;
}

}  // namespace sim
}  // namespace krabby

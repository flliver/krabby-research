#pragma once

#include "board_role.h"

#include <Arduino.h>

#if defined(KRABBY_HOST_FIRMWARE)
#include "EEPROM.h"
#else
#include <EEPROM.h>
#endif

// Mirrors role boot in firmware/arduino/arduino.ino (determineRole, loadRole, saveRole).

constexpr int BOARD_ROLE_EEPROM_ADDR = 32;
constexpr uint8_t BOARD_ROLE_EEPROM_MAGIC = 0xAB;
constexpr unsigned long BOARD_ROLE_ELECTION_MS = 3000;
constexpr unsigned long BOARD_ROLE_SYNC_INTERVAL_MS = 10;

constexpr const char *BOARD_ROLE_SYNC_TOKEN = "SYNC";
constexpr const char *BOARD_ROLE_ASSIGN_LEFT = "ROLE:LEFT";
constexpr const char *BOARD_ROLE_ASSIGN_RIGHT = "ROLE:RIGHT";

inline BoardRole loadBoardRoleHintFromEeprom()
{
    if (EEPROM.read(BOARD_ROLE_EEPROM_ADDR) != BOARD_ROLE_EEPROM_MAGIC)
        return ROLE_UNKNOWN;
    const uint8_t roleByte = EEPROM.read(BOARD_ROLE_EEPROM_ADDR + 1);
    if (roleByte == ROLE_FRONT || roleByte == ROLE_LEFT || roleByte == ROLE_RIGHT)
        return static_cast<BoardRole>(roleByte);
    return ROLE_UNKNOWN;
}

inline void saveBoardRoleHintToEeprom(BoardRole role)
{
    EEPROM.update(BOARD_ROLE_EEPROM_ADDR, BOARD_ROLE_EEPROM_MAGIC);
    EEPROM.update(BOARD_ROLE_EEPROM_ADDR + 1, static_cast<uint8_t>(role));
}

enum class BoardRoleElectionStatus
{
    InProgress,
    FollowerLeft,
    FollowerRight,
    LeaderFront,
    UnknownFrontActuators,
};

struct BoardRoleElectionState
{
    bool started = false;
    bool hasSyncFromLeft = false;
    bool hasSyncFromRight = false;
    bool isLeftAssigned = false;
    bool isRightAssigned = false;
    unsigned long startMs = 0;
    unsigned long lastSyncMs = 0;
};

template <typename UsbPrintT>
inline void boardRoleElectionEmitBootBanner(UsbPrintT &usb, BoardRole cachedRole)
{
    usb.println("--- SYNC ---");
    switch (cachedRole)
    {
        case ROLE_FRONT: usb.println("ROLE_HINT: FRONT"); break;
        case ROLE_LEFT: usb.println("ROLE_HINT: LEFT"); break;
        case ROLE_RIGHT: usb.println("ROLE_HINT: RIGHT"); break;
        default: break;
    }
}

template <typename UsbPrintT, typename LinkSerialT>
inline BoardRoleElectionStatus boardRoleElectionStepOnce(
    BoardRoleElectionState &state,
    const bool isI2cHost,
    UsbPrintT &usb,
    LinkSerialT &serialLeft,
    LinkSerialT &serialRight)
{
    if (!state.started)
    {
        boardRoleElectionEmitBootBanner(usb, loadBoardRoleHintFromEeprom());
        state.started = true;
        state.startMs = millis();
        state.lastSyncMs = 0;
    }

    if (millis() - state.lastSyncMs >= BOARD_ROLE_SYNC_INTERVAL_MS)
    {
        state.lastSyncMs = millis();
        serialLeft.println(BOARD_ROLE_SYNC_TOKEN);
        serialRight.println(BOARD_ROLE_SYNC_TOKEN);
    }

    if (serialLeft.available())
    {
        const String line = serialLeft.readStringUntil('\n');
        if (!isI2cHost && line.indexOf(BOARD_ROLE_ASSIGN_LEFT) >= 0)
        {
            saveBoardRoleHintToEeprom(ROLE_LEFT);
            usb.println("ROLE: LEFT");
            return BoardRoleElectionStatus::FollowerLeft;
        }
        if (line.indexOf(BOARD_ROLE_SYNC_TOKEN) >= 0) state.hasSyncFromLeft = true;
    }

    if (serialRight.available())
    {
        const String line = serialRight.readStringUntil('\n');
        if (!isI2cHost && line.indexOf(BOARD_ROLE_ASSIGN_RIGHT) >= 0)
        {
            saveBoardRoleHintToEeprom(ROLE_RIGHT);
            usb.println("ROLE: RIGHT");
            return BoardRoleElectionStatus::FollowerRight;
        }
        if (line.indexOf(BOARD_ROLE_SYNC_TOKEN) >= 0) state.hasSyncFromRight = true;
    }

    if (isI2cHost || (state.hasSyncFromLeft && state.hasSyncFromRight))
    {
        if (state.hasSyncFromLeft && !state.isLeftAssigned)
        {
            serialLeft.println(BOARD_ROLE_ASSIGN_LEFT);
            state.isLeftAssigned = true;
        }
        if (state.hasSyncFromRight && !state.isRightAssigned)
        {
            serialRight.println(BOARD_ROLE_ASSIGN_RIGHT);
            state.isRightAssigned = true;
        }
    }

    const bool electionWindowClosed =
        millis() - state.startMs >= BOARD_ROLE_ELECTION_MS;
    const bool bothSyncSeen = state.hasSyncFromLeft && state.hasSyncFromRight;
    if (!electionWindowClosed && !bothSyncSeen)
        return BoardRoleElectionStatus::InProgress;

    if (isI2cHost || bothSyncSeen)
    {
        saveBoardRoleHintToEeprom(ROLE_FRONT);
        usb.println("ROLE: FRONT");
        return BoardRoleElectionStatus::LeaderFront;
    }

    usb.println("ROLE: UNKNOWN (front actuators)");
    return BoardRoleElectionStatus::UnknownFrontActuators;
}

template <typename UsbPrintT, typename LinkSerialT>
inline BoardRoleElectionStatus runBoardRoleElectionBlocking(
    const bool isI2cHost,
    UsbPrintT &usb,
    LinkSerialT &serialLeft,
    LinkSerialT &serialRight)
{
    BoardRoleElectionState state;
    while (true)
    {
        const BoardRoleElectionStatus status = boardRoleElectionStepOnce(
            state, isI2cHost, usb, serialLeft, serialRight);
        if (status != BoardRoleElectionStatus::InProgress) return status;
    }
}

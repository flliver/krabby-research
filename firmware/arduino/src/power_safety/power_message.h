#pragma once

#include <stdint.h>
#include <string.h>

#define POWER_MSG_PREFIX "PWR"
#define POWER_TYPE_POWERING_DOWN_TOKEN "POWERING_DOWN"
#define POWER_TYPE_SHUTDOWN_ACK_TOKEN "SHUTDOWN_ACK"
#define POWER_TYPE_RESUMING_TOKEN "RESUMING"
#define POWER_TYPE_EMERGENCY_SHUTDOWN_TOKEN "EMERGENCY_SHUTDOWN"
#define POWER_REASON_UNDER_VOLTAGE_SOFT_TOKEN "under_voltage_soft"
#define POWER_REASON_HARD_CUT_TOKEN "hard_cut"
#define POWER_REASON_OVER_VOLTAGE_TOKEN "over_voltage"
#define POWER_REASON_VOLTAGE_RECOVERED_TOKEN "voltage_recovered"

static constexpr uint8_t POWER_MSG_SCHEMA = 1;

enum class PowerMessageType : uint8_t
{
    PoweringDown,
    ShutdownAck,
    Resuming,
    EmergencyShutdown,
};

enum class PoweringDownReason : uint8_t
{
    UnderVoltageSoft,
};

enum class EmergencyShutdownReason : uint8_t
{
    HardCut,
    OverVoltage,
};

enum class ResumingReason : uint8_t
{
    VoltageRecovered,
};

#if !defined(ARDUINO)
constexpr const char *powerMessageTypeToken(PowerMessageType type)
{
    return type == PowerMessageType::PoweringDown
        ? POWER_TYPE_POWERING_DOWN_TOKEN
        : type == PowerMessageType::ShutdownAck
            ? POWER_TYPE_SHUTDOWN_ACK_TOKEN
            : type == PowerMessageType::Resuming
                ? POWER_TYPE_RESUMING_TOKEN
                : type == PowerMessageType::EmergencyShutdown
                    ? POWER_TYPE_EMERGENCY_SHUTDOWN_TOKEN
                    : "";
}

constexpr const char *powerReasonToken(PoweringDownReason reason)
{
    return reason == PoweringDownReason::UnderVoltageSoft
        ? POWER_REASON_UNDER_VOLTAGE_SOFT_TOKEN
        : "";
}

constexpr const char *powerReasonToken(EmergencyShutdownReason reason)
{
    return reason == EmergencyShutdownReason::HardCut
        ? POWER_REASON_HARD_CUT_TOKEN
        : reason == EmergencyShutdownReason::OverVoltage
            ? POWER_REASON_OVER_VOLTAGE_TOKEN
            : "";
}

constexpr const char *powerReasonToken(ResumingReason reason)
{
    return reason == ResumingReason::VoltageRecovered
        ? POWER_REASON_VOLTAGE_RECOVERED_TOKEN
        : "";
}
#endif

#if defined(ARDUINO)
#include <Arduino.h>

inline void printPowerMessageType(Print &out, PowerMessageType type)
{
    switch (type)
    {
        case PowerMessageType::PoweringDown:
            out.print(F(POWER_TYPE_POWERING_DOWN_TOKEN));
            break;
        case PowerMessageType::ShutdownAck:
            out.print(F(POWER_TYPE_SHUTDOWN_ACK_TOKEN));
            break;
        case PowerMessageType::Resuming:
            out.print(F(POWER_TYPE_RESUMING_TOKEN));
            break;
        case PowerMessageType::EmergencyShutdown:
            out.print(F(POWER_TYPE_EMERGENCY_SHUTDOWN_TOKEN));
            break;
    }
}

inline void printPowerMessageReason(Print &out, PoweringDownReason reason)
{
    switch (reason)
    {
        case PoweringDownReason::UnderVoltageSoft:
            out.print(F(POWER_REASON_UNDER_VOLTAGE_SOFT_TOKEN));
            break;
    }
}

inline void printPowerMessageReason(
    Print &out, EmergencyShutdownReason reason)
{
    switch (reason)
    {
        case EmergencyShutdownReason::HardCut:
            out.print(F(POWER_REASON_HARD_CUT_TOKEN));
            break;
        case EmergencyShutdownReason::OverVoltage:
            out.print(F(POWER_REASON_OVER_VOLTAGE_TOKEN));
            break;
    }
}

inline void printPowerMessageReason(Print &out, ResumingReason reason)
{
    if (reason == ResumingReason::VoltageRecovered)
        out.print(F(POWER_REASON_VOLTAGE_RECOVERED_TOKEN));
}

inline void printPowerMessageHeader(Print &out, PowerMessageType type)
{
    out.print(F(POWER_MSG_PREFIX));
    out.print(' ');
    out.print(POWER_MSG_SCHEMA);
    out.print(' ');
    printPowerMessageType(out, type);
}

inline void emitPowerMessage(Print &out, PoweringDownReason reason)
{
    printPowerMessageHeader(out, PowerMessageType::PoweringDown);
    out.print(' ');
    printPowerMessageReason(out, reason);
    out.println();
}

inline void emitPowerMessage(Print &out, EmergencyShutdownReason reason)
{
    printPowerMessageHeader(out, PowerMessageType::EmergencyShutdown);
    out.print(' ');
    printPowerMessageReason(out, reason);
    out.println();
}

inline void emitPowerMessage(Print &out, ResumingReason reason)
{
    printPowerMessageHeader(out, PowerMessageType::Resuming);
    out.print(' ');
    printPowerMessageReason(out, reason);
    out.println();
}
#endif

inline bool isExactPowerToken(const char *text, const char *expected)
{
    if (text == nullptr || expected == nullptr)
        return false;

    const size_t expectedLength = strlen(expected);
    const size_t actualLength = strlen(text);
    if (actualLength == expectedLength)
        return memcmp(text, expected, expectedLength) == 0;
    return actualLength == expectedLength + 1 &&
           text[expectedLength] == '\r' &&
           memcmp(text, expected, expectedLength) == 0;
}

inline bool isShutdownAckMessage(const char *line, const char *prefix)
{
    if (line == nullptr || prefix == nullptr)
        return false;

    const size_t prefixLength = strlen(prefix);
    if (strncmp(line, prefix, prefixLength) != 0)
        return false;

    const char *cursor = line + prefixLength;
    if (*cursor < '0' || *cursor > '9')
        return false;

    uint16_t schema = 0;
    do
    {
        schema = static_cast<uint16_t>(schema * 10 + (*cursor - '0'));
        if (schema > UINT8_MAX)
            return false;
        ++cursor;
    } while (*cursor >= '0' && *cursor <= '9');

    if (*cursor != ' ' || schema != POWER_MSG_SCHEMA)
        return false;

    return isExactPowerToken(
        cursor + 1, POWER_TYPE_SHUTDOWN_ACK_TOKEN);
}

inline bool isShutdownAckLine(const char *line)
{
    return isShutdownAckMessage(line, POWER_MSG_PREFIX " ");
}

// The command dispatcher has already consumed the leading P.
inline bool isShutdownAckPayload(const char *payload)
{
    return isShutdownAckMessage(payload, "WR ");
}

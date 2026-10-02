#pragma once

// TODO: Consolidate serial command parsing and dispatch from arduino.ino here.

#include <errno.h>
#include <math.h>
#include <stdlib.h>

static const char CALIBRATION_COMMAND_PREFIX = 'C';
static const char ACTUATOR_CALIBRATION_TARGET[] = "ACTUATOR";
static const char POWER_SENSOR_CALIBRATION_TARGET[] = "PWR_SENSE";
static const char POWER_CALIBRATION_VOLTAGE_ACTION[] = "VOLTAGE";
static const char POWER_CALIBRATION_CURRENT_ACTION[] = "CURRENT";
static const char POWER_CALIBRATION_SHOW_ACTION[] = "SHOW";
static const char POWER_CALIBRATION_HELP_ACTION[] = "?";

enum class PowerCalibrationAction {
    Invalid,
    CalibrateVoltageOffsets,
    CalibrateCurrentScale,
    ShowCalibration,
    ShowHelp,
};

struct PowerCalibrationCommand {
    PowerCalibrationAction action;
    float firstReference;
    float secondReference;
};

inline char powerCalibrationAsciiUpper(char value)
{
    return value >= 'a' && value <= 'z' ? value - ('a' - 'A') : value;
}

inline bool powerCalibrationTokenEquals(
    const char* actual,
    const char* expected)
{
    if (actual == nullptr || expected == nullptr)
        return false;

    while (*actual != '\0' && *expected != '\0')
    {
        if (powerCalibrationAsciiUpper(*actual) != *expected)
            return false;
        ++actual;
        ++expected;
    }
    return *actual == '\0' && *expected == '\0';
}

inline bool isActuatorCalibrationCommand(
    size_t tokenCount,
    const char* const* tokens)
{
    return tokenCount == 0 ||
           (tokens != nullptr &&
            tokenCount == 1 &&
            powerCalibrationTokenEquals(
                tokens[0], ACTUATOR_CALIBRATION_TARGET));
}

inline PowerCalibrationAction parsePowerCalibrationAction(
    const char* namespaceToken,
    const char* actionToken)
{
    if (!powerCalibrationTokenEquals(
            namespaceToken, POWER_SENSOR_CALIBRATION_TARGET))
        return PowerCalibrationAction::Invalid;
    if (powerCalibrationTokenEquals(
            actionToken, POWER_CALIBRATION_VOLTAGE_ACTION))
        return PowerCalibrationAction::CalibrateVoltageOffsets;
    if (powerCalibrationTokenEquals(
            actionToken, POWER_CALIBRATION_CURRENT_ACTION))
        return PowerCalibrationAction::CalibrateCurrentScale;
    if (powerCalibrationTokenEquals(
            actionToken, POWER_CALIBRATION_SHOW_ACTION))
        return PowerCalibrationAction::ShowCalibration;
    if (powerCalibrationTokenEquals(
            actionToken, POWER_CALIBRATION_HELP_ACTION))
        return PowerCalibrationAction::ShowHelp;
    return PowerCalibrationAction::Invalid;
}

inline bool parsePowerCalibrationNumber(const char* token, float& result)
{
    if (token == nullptr || *token == '\0')
        return false;

    errno = 0;
    char* end = nullptr;
    const double parsed = strtod(token, &end);
    if (end == token ||
        *end != '\0' ||
        errno == ERANGE ||
        !isfinite(parsed))
        return false;

    const float value = static_cast<float>(parsed);
    if (!isfinite(value))
        return false;

    result = value;
    return true;
}

inline bool parsePowerCalibrationCommand(
    size_t tokenCount,
    const char* const* tokens,
    PowerCalibrationCommand& result)
{
    if (tokens == nullptr || tokenCount < 2)
        return false;

    const PowerCalibrationAction action =
        parsePowerCalibrationAction(tokens[0], tokens[1]);
    PowerCalibrationCommand parsed = {
        action, 0.0f, 0.0f};

    switch (action)
    {
        case PowerCalibrationAction::CalibrateVoltageOffsets:
            if (tokenCount != 4 ||
                !parsePowerCalibrationNumber(tokens[2], parsed.firstReference) ||
                !parsePowerCalibrationNumber(tokens[3], parsed.secondReference))
                return false;
            break;
        case PowerCalibrationAction::CalibrateCurrentScale:
            if (tokenCount != 3 ||
                !parsePowerCalibrationNumber(tokens[2], parsed.firstReference))
                return false;
            break;
        case PowerCalibrationAction::ShowCalibration:
        case PowerCalibrationAction::ShowHelp:
            if (tokenCount != 2)
                return false;
            break;
        case PowerCalibrationAction::Invalid:
            return false;
    }

    result = parsed;
    return true;
}

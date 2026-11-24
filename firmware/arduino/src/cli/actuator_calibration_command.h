#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../actuator/calibration_constants.h"
#include "../actuator/calibration_text_sink.h"
#include "power_calibration_command.h"

// Actuator calibration command, after the leading 'C' has been consumed:
//   C [ACTUATOR] [SEEK|SWEEP] [pwm]
// Bare C runs SEEK then SWEEP at CAL_DEFAULT_PWM; SEEK or SWEEP selects one
// pass; pwm is a decimal integer in [CAL_PWM_MIN, CAL_PWM_MAX]. Tokens are
// case-insensitive. A command whose first token is PWR_SENSE belongs to the
// power-monitor calibration and is left to parsePowerCalibrationCommand().

static const char ACTUATOR_CALIBRATION_SEEK_TOKEN[] = "SEEK";
static const char ACTUATOR_CALIBRATION_SWEEP_TOKEN[] = "SWEEP";

enum class ActuatorCalibrationCommandKind : uint8_t
{
    PowerSensor,  // C PWR_SENSE ...: not an actuator command
    Run,          // valid; request holds the passes and PWM
    Invalid,      // malformed; nothing may move
};

struct ActuatorCalibrationRequest
{
    uint8_t passes;  // CAL_PASS_SEEK, CAL_PASS_SWEEP or CAL_PASS_ALL
    int pwm;
};

// Decimal digits only, within [CAL_PWM_MIN, CAL_PWM_MAX].
inline bool parseActuatorCalibrationPwm(const char *token, int &pwm)
{
    if (token == nullptr || *token == '\0')
        return false;

    int value = 0;
    for (; *token != '\0'; ++token)
    {
        if (*token < '0' || *token > '9' || value > CAL_PWM_MAX)
            return false;
        value = value * 10 + (*token - '0');
    }
    if (value < CAL_PWM_MIN || value > CAL_PWM_MAX)
        return false;

    pwm = value;
    return true;
}

// request is written only for Run.
inline ActuatorCalibrationCommandKind parseActuatorCalibrationCommand(
    size_t tokenCount,
    const char *const *tokens,
    ActuatorCalibrationRequest &request)
{
    if (tokenCount == 0)
    {
        request.passes = CAL_PASS_ALL;
        request.pwm = CAL_DEFAULT_PWM;
        return ActuatorCalibrationCommandKind::Run;
    }
    if (tokens == nullptr)
        return ActuatorCalibrationCommandKind::Invalid;
    if (powerCalibrationTokenEquals(tokens[0], POWER_SENSOR_CALIBRATION_TARGET))
        return ActuatorCalibrationCommandKind::PowerSensor;

    size_t index = 0;
    if (powerCalibrationTokenEquals(tokens[index], ACTUATOR_CALIBRATION_TARGET))
        ++index;

    uint8_t passes = CAL_PASS_ALL;
    if (index < tokenCount &&
        powerCalibrationTokenEquals(tokens[index], ACTUATOR_CALIBRATION_SEEK_TOKEN))
    {
        passes = CAL_PASS_SEEK;
        ++index;
    }
    else if (index < tokenCount &&
             powerCalibrationTokenEquals(tokens[index], ACTUATOR_CALIBRATION_SWEEP_TOKEN))
    {
        passes = CAL_PASS_SWEEP;
        ++index;
    }

    int pwm = CAL_DEFAULT_PWM;
    if (index < tokenCount)
    {
        if (!parseActuatorCalibrationPwm(tokens[index], pwm))
            return ActuatorCalibrationCommandKind::Invalid;
        ++index;
    }
    if (index != tokenCount)
        return ActuatorCalibrationCommandKind::Invalid;

    request.passes = passes;
    request.pwm = pwm;
    return ActuatorCalibrationCommandKind::Run;
}

// Prints "usage: C [SEEK|SWEEP] [pwm], pwm <min>-<max>" without a line end.
inline void printActuatorCalibrationUsage(CalibrationTextSink &out)
{
    out.printFlash(KRABBY_FLASH_LITERAL("usage: C [SEEK|SWEEP] [pwm], pwm "));
    out.printNumber(CAL_PWM_MIN);
    out.printChar('-');
    out.printNumber(CAL_PWM_MAX);
}

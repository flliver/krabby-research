#include <math.h>

#include "display_frame_model.h"
#include "display_constants.h"

DisplayFrame::DisplayFrame()
    : role(ROLE_UNKNOWN), controllers{}, actuators{},
      batteryLevel{0.0f, 0.0f},
      batteryDecivolts{BATTERY_DECIVOLTS_NO_SIGNAL, BATTERY_DECIVOLTS_NO_SIGNAL},
      packVoltage(-1.0f), roll(), pitch()
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        actuators[actuatorId] = ActuatorGlyph::Disconnected;
}

void setBatteryVoltages(DisplayFrame &frame, const Volts (&voltage)[2])
{
    // Coarse resting-voltage gauge, not a state-of-charge estimate.
    static constexpr float BATTERY_EMPTY_VOLTS = 12.0f;
    static constexpr float BATTERY_FULL_VOLTS = 13.4f;
    for (size_t battery = 0; battery < 2; ++battery)
    {
        const float volts = voltage[battery].value();
        const bool isValid = isfinite(volts) && volts >= 0.0f && volts <= 99.9f;
        const float level = isValid
            ? (volts - BATTERY_EMPTY_VOLTS) / (BATTERY_FULL_VOLTS - BATTERY_EMPTY_VOLTS)
            : 0.0f;
        frame.batteryLevel[battery] = level < 0.0f ? 0.0f : (level > 1.0f ? 1.0f : level);
        frame.batteryDecivolts[battery] = isValid
            ? static_cast<int16_t>(lround(volts * 10.0f))
            : BATTERY_DECIVOLTS_NO_SIGNAL;
    }
    frame.packVoltage = Volts(
        frame.batteryDecivolts[0] != BATTERY_DECIVOLTS_NO_SIGNAL &&
        frame.batteryDecivolts[1] != BATTERY_DECIVOLTS_NO_SIGNAL
            ? voltage[0].value() + voltage[1].value() : -1.0f);
}

ActuatorGlyph selectActuatorGlyph(
    const ActuatorStatus &status,
    int moveThreshold)
{
    if (status.connectionState == ActuatorConnection::Disconnected)
        return ActuatorGlyph::Disconnected;
    if (status.commandedPwm >= moveThreshold)
        return ActuatorGlyph::Extend;
    if (status.commandedPwm <= -moveThreshold)
        return ActuatorGlyph::Retract;
    return status.connectionState == ActuatorConnection::Unknown
        ? ActuatorGlyph::Unverified
        : ActuatorGlyph::Hold;
}

bool displayFramesEqual(const DisplayFrame &left, const DisplayFrame &right)
{
    if (left.role != right.role ||
        left.packVoltage.value() != right.packVoltage.value() ||
        left.roll.value() != right.roll.value() ||
        left.pitch.value() != right.pitch.value())
        return false;
    for (BoardRole role : ALL_BOARD_ROLES)
        if (left.controllers[role] != right.controllers[role])
            return false;
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        if (left.actuators[actuatorId] != right.actuators[actuatorId])
            return false;
    for (size_t battery = 0; battery < 2; ++battery)
        if (left.batteryLevel[battery] != right.batteryLevel[battery] ||
            left.batteryDecivolts[battery] != right.batteryDecivolts[battery])
            return false;
    return true;
}

DisplayFrame buildDisplayFrame(
    BoardRole role,
    const ControllerFreshnessTracker (&controllerFreshnessTrackers)[BOARD_ROLE_COUNT],
    const ActuatorStatus (&actuatorStatus)[ActuatorId::ActuatorCount],
    const ImuMeasurement &measurement,
    uint32_t nowMilliseconds,
    int moveThreshold)
{
    DisplayFrame frame;
    frame.role = role;
    for (BoardRole boardRole : ALL_BOARD_ROLES)
        frame.controllers[boardRole] =
            controllerFreshnessTrackers[boardRole].isFresh(
                nowMilliseconds,
                CONTROLLER_DISPLAY_TIMEOUT_MILLISECONDS);
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        if (frame.controllers[getBoardRoleForActuator(actuatorId)])
            frame.actuators[actuatorId] = selectActuatorGlyph(
                actuatorStatus[actuatorId], moveThreshold);

    if (measurement.didSucceed())
    {
        frame.roll = Degrees(roundHalfAwayFromZero(
            computeRollFromAcceleration(measurement).value()));
        frame.pitch = Degrees(roundHalfAwayFromZero(
            computePitchFromAcceleration(measurement).value()));
    }
    return frame;
}

bool hasDisconnectedActuator(const DisplayFrame &frame)
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        if (!frame.controllers[getBoardRoleForActuator(actuatorId)])
            continue;
        if (frame.actuators[actuatorId] == ActuatorGlyph::Disconnected)
            return true;
    }
    return false;
}

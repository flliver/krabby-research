#pragma once

#include <stddef.h>
#include <stdint.h>

#include "display_constants.h"
#include "../actuator/actuator_status.h"
#include "../controller/controller_freshness_tracker.h"
#include "../imu/imu_measurement.h"
#include "../telemetry.h"
#include "../units/angular_units.h"
#include "../units/electrical_units.h"

enum class ActuatorGlyph : uint8_t
{
    Hold,
    Extend,
    Retract,
    Disconnected,
    Unverified,
};

static constexpr int16_t BATTERY_DECIVOLTS_NO_SIGNAL = INT16_MIN;

struct DisplayFrame
{
    DisplayFrame();

    BoardRole role;
    bool controllers[BOARD_ROLE_COUNT];
    ActuatorGlyph actuators[ActuatorId::ActuatorCount];
    float batteryLevel[2];
    int16_t batteryDecivolts[2];
    Volts packVoltage;
    Degrees roll;
    Degrees pitch;
};

void setBatteryVoltages(DisplayFrame &frame, const Volts (&voltage)[2]);

DisplayFrame buildDisplayFrame(
    BoardRole role,
    const ControllerFreshnessTracker (&controllerFreshnessTrackers)[BOARD_ROLE_COUNT],
    const ActuatorStatus (&actuatorStatus)[ActuatorId::ActuatorCount],
    const ImuMeasurement &measurement,
    uint32_t nowMilliseconds,
    int moveThreshold);

ActuatorGlyph selectActuatorGlyph(
    const ActuatorStatus &status,
    int moveThreshold);
bool hasDisconnectedActuator(const DisplayFrame &frame);
bool displayFramesEqual(const DisplayFrame &left, const DisplayFrame &right);

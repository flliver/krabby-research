#pragma once

#include <stddef.h>
#include <stdint.h>

#include "actuator/actuator_status.h"
#include "controller/board_role.h"
#include "imu/imu_measurement.h"

static constexpr uint16_t TELEMETRY_INTERVAL_MS = 50;
static constexpr char TELEMETRY_SEGMENT_DELIMITER = ';';
static constexpr char TELEMETRY_FIELD_SEPARATOR = ' ';
static constexpr char IMU_TELEMETRY_TAG[] = "IMU";
static constexpr size_t CONTROLLER_ACTUATOR_COUNT = 6;
static constexpr size_t ACTUATOR_TELEMETRY_FIELD_COUNT = 9;
static constexpr size_t ACTUATOR_TELEMETRY_MAX_FIELD_COUNT = 10;
static constexpr size_t ACTUATOR_TELEMETRY_NAME_FIELD_INDEX = 0;
static constexpr size_t ACTUATOR_TELEMETRY_POSITION_FIELD_INDEX = 1;
static constexpr size_t ACTUATOR_TELEMETRY_RETRACT_PWM_FIELD_INDEX = 6;
static constexpr size_t ACTUATOR_TELEMETRY_EXTEND_PWM_FIELD_INDEX = 7;
static constexpr size_t ACTUATOR_TELEMETRY_CONNECTION_STATE_FIELD_INDEX = 9;

// TODO: Move actuator telemetry encoding into this module.

bool parseActuatorStatus(
    const char *line,
    BoardRole expectedRole,
    ActuatorStatus (&status)[CONTROLLER_ACTUATOR_COUNT]);

const char *boardTelemetryRoleLabel(BoardRole role);

template <typename Output>
void appendImuMeasurement(
    Output &output,
    const ImuMeasurement &measurement)
{
    output.print(TELEMETRY_SEGMENT_DELIMITER);
    output.print(IMU_TELEMETRY_TAG);
    output.print(TELEMETRY_FIELD_SEPARATOR);

    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        output.print(measurement.acceleration[axis].value(), 3);
        output.print(TELEMETRY_FIELD_SEPARATOR);
    }

    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        output.print(measurement.angularRate[axis].value(), 4);
        output.print(TELEMETRY_FIELD_SEPARATOR);
    }

    output.print(measurement.temperature.value(), 1);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(measurement.didSucceed() ? 1 : 0);
}

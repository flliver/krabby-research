#pragma once

#include "power_monitor/power_measurement.h"

#include <stddef.h>
#include <stdint.h>

#include "actuator/actuator_status.h"
#include "controller/board_role.h"
#include "imu/imu_measurement.h"
#include "units/electrical_units.h"

static constexpr uint16_t TELEMETRY_INTERVAL_MS = 50;

// Actuator telemetry and leader sensor reads share this cadence.
// Unsigned subtraction preserves elapsed time across millis() rollover.
inline bool isTelemetryPollDue(uint32_t now, uint32_t previousPoll)
{
    return static_cast<uint32_t>(now - previousPoll) >= TELEMETRY_INTERVAL_MS;
}
static constexpr char TELEMETRY_SEGMENT_DELIMITER = ';';
static constexpr char TELEMETRY_FIELD_SEPARATOR = ' ';
static constexpr char IMU_TELEMETRY_TAG[] = "IMU";
// Appended by the leader alongside IMU, under its own tag, so a parser that does
// not know it skips one segment rather than mis-reading the line.
static constexpr char BATT_TELEMETRY_TAG[] = "BATT";
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
    Output &out,
    const ImuMeasurement &measurement)
{
    out.print(TELEMETRY_SEGMENT_DELIMITER);
    out.print(IMU_TELEMETRY_TAG);
    out.print(TELEMETRY_FIELD_SEPARATOR);

    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        out.print(measurement.acceleration[axis].value(), 3);
        out.print(TELEMETRY_FIELD_SEPARATOR);
    }

    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        out.print(measurement.angularRate[axis].value(), 4);
        out.print(TELEMETRY_FIELD_SEPARATOR);
    }

    out.print(measurement.temperature.value(), 1);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(measurement.didSucceed() ? 1 : 0);
}

// Voltage-region codes carried in BATT telemetry. Task 3 emits NORMAL;
// power-management transitions are implemented in Task 4.
enum PackVoltageRegion : uint8_t
{
    PACK_REGION_NORMAL = 0,
    PACK_REGION_WARN = 1,
    PACK_REGION_SOFT_CUT = 2,
    PACK_REGION_HARD_CUT = 3,
    PACK_REGION_OVER_VOLT = 4
};

template <typename Output>
inline void appendBatteryTelemetry(
    Output& out,
    const PowerMonitorMeasurement& packMeasurement,
    const PowerMonitorMeasurement& midpointMeasurement,
    Volts inferredBattBVoltage,
    bool isDiverged,
    uint8_t packRegion)
{
    out.print(TELEMETRY_SEGMENT_DELIMITER);
    out.print(BATT_TELEMETRY_TAG);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packMeasurement.voltage.value(), 2);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packMeasurement.current.value(), 2);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packMeasurement.power.value(), 1);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packMeasurement.charge.value(), 1);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(midpointMeasurement.voltage.value(), 2);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(inferredBattBVoltage.value(), 2);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(isDiverged ? 1 : 0);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packRegion);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(packMeasurement.isValid ? 1 : 0);
    out.print(TELEMETRY_FIELD_SEPARATOR);
    out.print(midpointMeasurement.isValid ? 1 : 0);
}

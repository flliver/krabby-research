#pragma once

#include <math.h>
#include <stdint.h>

#include "../units/angular_units.h"
#include "../units/inertial_units.h"
#include "../units/temperature_units.h"
#include "imu_constants.h"

struct ImuMeasurement
{
    explicit ImuMeasurement(bool isValid = false)
        : acceleration{},
          angularRate{},
          temperature{},
          isValid(isValid)
    {
    }

    MetersPerSecondSquared acceleration[3];
    RadiansPerSecond angularRate[3];
    Celsius temperature;
    // Telemetry carries one validity bit.
    bool isValid;

    bool didSucceed() const { return isValid; }
};

inline ImuMeasurement transformImuMeasurementToBodyFrame(
    const ImuMeasurement &sensorMeasurement)
{
    if (!sensorMeasurement.didSucceed())
        return sensorMeasurement;

    ImuMeasurement bodyMeasurement{true};
    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        const uint8_t source = IMU_AXIS_SRC[axis];
        const float direction = static_cast<float>(IMU_AXIS_SIGN[axis]);

        bodyMeasurement.acceleration[axis] =
            sensorMeasurement.acceleration[source].scalarMultiply(direction);
        bodyMeasurement.angularRate[axis] =
            sensorMeasurement.angularRate[source].scalarMultiply(direction);
    }

    bodyMeasurement.temperature = sensorMeasurement.temperature;
    return bodyMeasurement;
}

inline Degrees computeRollFromAcceleration(const ImuMeasurement &measurement)
{
    const float ay = measurement.acceleration[1].value();
    const float az = measurement.acceleration[2].value();
    return Radians(atan2(ay, az)).toDegrees();
}

inline Degrees computePitchFromAcceleration(const ImuMeasurement &measurement)
{
    const float ax = measurement.acceleration[0].value();
    const float ay = measurement.acceleration[1].value();
    const float az = measurement.acceleration[2].value();
    return Radians(atan2(-ax, sqrt(ay * ay + az * az))).toDegrees();
}

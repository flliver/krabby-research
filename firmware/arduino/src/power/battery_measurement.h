#pragma once

#include <math.h>
#include <stdint.h>

struct BatteryMeasurement
{
    float packVolts = 0.0f;
    float packAmps = 0.0f;
    float packWatts = 0.0f;
    float packChargeCoulombs = 0.0f;
    float battAVolts = 0.0f;
    float battBVolts = 0.0f;
    uint8_t divergenceFlag = 0;
    uint8_t powerState = 0;
    bool valid = false;

    bool didSucceed() const { return valid; }
};

inline void updateBatteryDivergenceFlag(
    BatteryMeasurement &measurement,
    float thresholdVolts)
{
    if (!measurement.valid) return;
    const float divergence = fabsf(measurement.battAVolts - measurement.battBVolts);
    measurement.divergenceFlag = divergence > thresholdVolts ? 1 : 0;
}

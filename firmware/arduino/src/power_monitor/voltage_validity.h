#pragma once

#include <math.h>

#include "power_monitor_constants.h"

static constexpr float BATTERY_PACK_V_MIN = 0.0f;
static constexpr float BATTERY_PACK_V_MAX = 40.0f;
static constexpr float BATTERY_CELL_V_MIN = 0.0f;
static constexpr float BATTERY_CELL_V_MAX = 20.0f;

inline bool isBatteryPackVoltageValid(float packVoltage)
{
    return isfinite(packVoltage) &&
           packVoltage >= BATTERY_PACK_V_MIN &&
           packVoltage <= BATTERY_PACK_V_MAX;
}

inline bool isBatteryCellVoltageValid(float cellVoltage)
{
    return isfinite(cellVoltage) &&
           cellVoltage >= BATTERY_CELL_V_MIN &&
           cellVoltage <= BATTERY_CELL_V_MAX;
}

// Missing readings count as diverged; range validity is checked separately.
inline bool areBatteryVoltagesDiverged(Volts batteryA, Volts batteryB)
{
    return !isfinite(batteryA.value()) || !isfinite(batteryB.value()) ||
           abs(batteryA - batteryB) > BATTERY_DIVERGENCE_THRESHOLD;
}

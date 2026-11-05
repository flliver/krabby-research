#pragma once

#include "power_calibration.h"

struct PowerReadings
{
    PowerMonitorMeasurement pack;
    PowerMonitorMeasurement midpoint;
    Volts batteryB;
};

template <class PackProvider, class MidpointProvider>
PowerReadings readPowerMeasurements(
    PackProvider &packProvider, MidpointProvider &midpointProvider,
    const PowerCalibration &calibration)
{
    const auto pack = calibration.applyPackCalibration(packProvider.get().measure());
    const auto midpoint = calibration.applyMidpointCalibration(midpointProvider.get().measure());
    return {pack, midpoint, pack.voltage - midpoint.voltage};
}

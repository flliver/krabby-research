#pragma once

#include "src/power_monitor/ina228_adapter.h"
#include "src/power_monitor/power_calibration.h"
#include "src/telemetry.h"

namespace krabby {
namespace sim {
namespace host {

struct HostBatterySample
{
    PowerMonitorMeasurement pack;
    PowerMonitorMeasurement midpoint;
    Volts batteryB{NAN};
    bool isDiverged = true;
};

class HostLeaderBattery
{
public:
    void setup();
    // Called at the firmware telemetry cadence. The OLED uses the cached sample.
    const HostBatterySample &sample();
    const HostBatterySample &latest() const { return latest_; }

private:
    Ina228Adapter pack_{PACK_POWER_MONITOR_ADDRESS,
        PACK_SHUNT_RESISTANCE_OHMS, PACK_SHUNT_MAX_CURRENT_AMPS, true};
    Ina228Adapter midpoint_{MIDPOINT_POWER_MONITOR_ADDRESS};
    PowerCalibration calibration_;
    HostBatterySample latest_;
};

HostLeaderBattery &leaderBattery();

template <typename Output>
void appendBatteryTelemetry(Output &output, const HostBatterySample &sample)
{
    // Same task 3 encoding as hardware; this runtime does not execute task 4's FSM.
    ::appendBatteryTelemetry(output, sample.pack, sample.midpoint,
        sample.batteryB, sample.isDiverged, PACK_REGION_NORMAL);
}

}  // namespace host
}  // namespace sim
}  // namespace krabby

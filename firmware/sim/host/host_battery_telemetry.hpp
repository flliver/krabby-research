#pragma once

#include "sim_battery_model.hpp"
#include "src/telemetry.h"

namespace krabby {
namespace sim {
namespace host {

class HostLeaderBattery
{
public:
    void setup();

    SimBatteryTelemetry sample() const;

    bool isOnline() const { return online_; }

private:
    bool online_ = false;
};

HostLeaderBattery &leaderBattery();

inline BatteryMeasurement batteryMeasurementFromSim(const SimBatteryTelemetry &sample)
{
    BatteryMeasurement out{};
    if (!sample.valid) return out;
    out.packVolts = sample.packVolts;
    out.packAmps = sample.packAmps;
    out.packWatts = sample.packWatts;
    out.packChargeCoulombs = sample.packChargeCoulombs;
    out.battAVolts = sample.battAVolts;
    out.battBVolts = sample.battBVolts;
    out.divergenceFlag = sample.divergenceFlag;
    out.powerState = sample.powerState;
    out.valid = true;
    return out;
}

template <typename Output>
void appendBatteryTelemetry(Output &output, const SimBatteryTelemetry &sample)
{
    appendBatteryMeasurement(output, batteryMeasurementFromSim(sample));
}

}  // namespace host
}  // namespace sim
}  // namespace krabby

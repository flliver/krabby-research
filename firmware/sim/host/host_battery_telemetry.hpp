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

template <typename Output>
void appendBatteryTelemetry(Output &output, const SimBatteryTelemetry &sample)
{
    output.print(TELEMETRY_SEGMENT_DELIMITER);
    output.print(BATT_TELEMETRY_TAG);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.packVolts, 2);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.packAmps, 3);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.packWatts, 2);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.packChargeCoulombs, 1);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.battAVolts, 2);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(sample.battBVolts, 2);
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(static_cast<int>(sample.divergenceFlag));
    output.print(TELEMETRY_FIELD_SEPARATOR);
    output.print(static_cast<int>(sample.powerState));
}

}  // namespace host
}  // namespace sim
}  // namespace krabby

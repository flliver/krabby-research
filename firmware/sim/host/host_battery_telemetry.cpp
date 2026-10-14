#include "host_battery_telemetry.hpp"

#include "sim_i2c.hpp"

#include "Wire.h"

namespace krabby {
namespace sim {
namespace host {

namespace {

HostLeaderBattery g_leader_battery;

bool probeAddress(uint8_t address)
{
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

}  // namespace

void HostLeaderBattery::setup()
{
    online_ = false;
    Wire.begin();
    if (!probeAddress(kI2cAddrIna228Pack) || !probeAddress(kI2cAddrIna228Midpoint))
    {
        return;
    }
    online_ = true;
}

SimBatteryTelemetry HostLeaderBattery::sample() const
{
    if (!online_)
    {
        return SimBatteryTelemetry{};
    }
    return computeSimBatteryTelemetry();
}

HostLeaderBattery &leaderBattery() { return g_leader_battery; }

}  // namespace host
}  // namespace sim
}  // namespace krabby

#include "host_battery_telemetry.hpp"

#include "EEPROM.h"
#include "Wire.h"
#include "src/power_monitor/voltage_validity.h"

namespace krabby {
namespace sim {
namespace host {
namespace {
HostLeaderBattery g_leader_battery;

struct FrontPowerCalibrationStorage
{
    void load(PowerCalibrationRecord &record)
    {
        EEPROM.boards[static_cast<int>(BoardId::kFront)].get(EEPROM_POWER_CAL_ADDR, record);
    }
};
}

void HostLeaderBattery::setup()
{
    // Reconstruct adapters to clear retry state after a simulated reboot.
    pack_ = Ina228Adapter(PACK_POWER_MONITOR_ADDRESS,
        PACK_SHUNT_RESISTANCE_OHMS, PACK_SHUNT_MAX_CURRENT_AMPS, true);
    midpoint_ = Ina228Adapter(MIDPOINT_POWER_MONITOR_ADDRESS);
    latest_ = HostBatterySample{};
    Wire.begin();
    pack_.begin(&Wire);
    midpoint_.begin(&Wire);
    FrontPowerCalibrationStorage storage;
    calibration_.load(storage);
}

const HostBatterySample &HostLeaderBattery::sample()
{
    latest_.pack = calibration_.applyPackCalibration(pack_.measure());
    latest_.midpoint = calibration_.applyMidpointCalibration(midpoint_.measure());
    latest_.batteryB = latest_.pack.voltage - latest_.midpoint.voltage;
    latest_.isDiverged = areBatteryVoltagesDiverged(
        latest_.midpoint.voltage, latest_.batteryB);
    return latest_;
}

HostLeaderBattery &leaderBattery() { return g_leader_battery; }

}  // namespace host
}  // namespace sim
}  // namespace krabby

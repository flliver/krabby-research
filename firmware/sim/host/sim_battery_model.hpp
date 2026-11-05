#pragma once

#include "battery_sim_constants.hpp"

#include <cstdint>

namespace krabby {
namespace sim {

// Physical inputs to the simulated INA228s; readings are decoded by the real driver.
struct SimBatteryState
{
    float packVolts = kSimDefaultPackVolts;
    float packAmps = kSimDefaultPackAmps;
    float packChargeCoulombs = kSimDefaultPackChargeCoulombs;
    float battAVolts = kSimDefaultBattAVolts;
};

void resetSimBatteryModel();
SimBatteryState simBatteryState();
void setSimBatteryState(const SimBatteryState &state);

}  // namespace sim
}  // namespace krabby

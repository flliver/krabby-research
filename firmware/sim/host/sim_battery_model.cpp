#include "sim_battery_model.hpp"
#include "sim_i2c.hpp"

namespace krabby {
namespace sim {

void resetSimBatteryModel() { setSimBatteryState(SimBatteryState{}); }

SimBatteryState simBatteryState()
{
    SimBatteryState state;
    state.packVolts = simI2cBus().packIna228().busVolts;
    state.packAmps = simI2cBus().packIna228().amps;
    state.packChargeCoulombs = simI2cBus().packIna228().coulombs;
    state.battAVolts = simI2cBus().midpointIna228().busVolts;
    return state;
}

void setSimBatteryState(const SimBatteryState &state)
{
    auto &pack = simI2cBus().packIna228();
    pack.busVolts = state.packVolts;
    pack.amps = state.packAmps;
    pack.coulombs = state.packChargeCoulombs;
    simI2cBus().midpointIna228().busVolts = state.battAVolts;
}

}  // namespace sim
}  // namespace krabby

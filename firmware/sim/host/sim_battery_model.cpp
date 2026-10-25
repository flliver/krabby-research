#include "sim_battery_model.hpp"

#include <cmath>

namespace krabby {
namespace sim {

namespace {

SimBatteryState g_battery_state;

}  // namespace

void resetSimBatteryModel() { g_battery_state = SimBatteryState{}; }

const SimBatteryState &simBatteryState() { return g_battery_state; }

void setSimBatteryState(const SimBatteryState &state) { g_battery_state = state; }

SimBatteryTelemetry computeSimBatteryTelemetry()
{
    SimBatteryTelemetry out{};
    out.packVolts = g_battery_state.packVolts;
    out.packAmps = g_battery_state.packAmps;
    out.packWatts = g_battery_state.packVolts * g_battery_state.packAmps;
    out.packChargeCoulombs = g_battery_state.packChargeCoulombs;
    out.battAVolts = g_battery_state.battAVolts;
    out.battBVolts = g_battery_state.packVolts - g_battery_state.battAVolts;
    const float divergence = std::fabs(out.battAVolts - out.battBVolts);
    out.divergenceFlag =
        divergence > kSimBatteryDivergenceThresholdVolts ? static_cast<uint8_t>(1) : 0;
    out.powerState = g_battery_state.powerState;
    out.valid = true;
    return out;
}

}  // namespace sim
}  // namespace krabby

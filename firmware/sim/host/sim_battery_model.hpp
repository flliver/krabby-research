#pragma once

#include "battery_sim_constants.hpp"

#include <cstdint>

namespace krabby {
namespace sim {

struct SimBatteryState
{
    float packVolts = kSimDefaultPackVolts;
    float packAmps = kSimDefaultPackAmps;
    float packChargeCoulombs = kSimDefaultPackChargeCoulombs;
    float battAVolts = kSimDefaultBattAVolts;
    uint8_t powerState = kSimBatteryPowerStateNormal;
};

struct SimBatteryTelemetry
{
    float packVolts = 0.0f;
    float packAmps = 0.0f;
    float packWatts = 0.0f;
    float packChargeCoulombs = 0.0f;
    float battAVolts = 0.0f;
    float battBVolts = 0.0f;
    uint8_t divergenceFlag = 0;
    uint8_t powerState = 0;
    bool valid = false;
};

void resetSimBatteryModel();
const SimBatteryState &simBatteryState();
void setSimBatteryState(const SimBatteryState &state);

SimBatteryTelemetry computeSimBatteryTelemetry();

}  // namespace sim
}  // namespace krabby

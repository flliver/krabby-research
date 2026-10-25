#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

// Nominal 24 V LiFePO4 pack (M16 / TASK-4 bench ranges: ~25–27 V pack, ~12–14 V per cell).
static constexpr float kSimDefaultPackVolts = 26.40f;
static constexpr float kSimDefaultPackAmps = 1.250f;
static constexpr float kSimDefaultBattAVolts = 13.20f;
static constexpr float kSimDefaultPackChargeCoulombs = 100000.0f;
static constexpr float kSimBatteryDivergenceThresholdVolts = 0.5f;

static constexpr uint8_t kSimBatteryPowerStateNormal = 0;

}  // namespace sim
}  // namespace krabby

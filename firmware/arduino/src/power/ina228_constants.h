#pragma once

#include <stdint.h>

// Leader Qwiic chain: pack monitor and midpoint sense (see firmware/sim/I2C_SIM_SCOPE.md).
static constexpr uint8_t INA228_PACK_ADDRESS = 0x40;
static constexpr uint8_t INA228_MIDPOINT_ADDRESS = 0x41;

// External 200 A / 75 mV shunt on the pack monitor (ohms, max amperes).
static constexpr float INA228_PACK_SHUNT_OHMS = 0.000375f;
static constexpr float INA228_PACK_SHUNT_MAX_AMPS = 200.0f;

static constexpr float INA228_DIVERGENCE_THRESHOLD_VOLTS = 0.5f;

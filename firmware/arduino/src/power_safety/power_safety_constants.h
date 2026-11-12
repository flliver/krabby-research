#pragma once

#include <stdint.h>

#include "../../eeprom_layout.h"
#include "../units/electrical_units.h"

static constexpr Volts PACK_HARD_CUT_THRESHOLD(22.4f);
static constexpr Volts PACK_SOFT_CUT_THRESHOLD(24.0f);
static constexpr Volts PACK_WARNING_THRESHOLD(24.8f);
static constexpr Volts PACK_RECOVERY_THRESHOLD(26.4f);
static constexpr Volts PACK_OVER_VOLT_THRESHOLD(29.6f);

static constexpr uint8_t POWER_CUT_DEBOUNCE_TICKS = 4;
static constexpr uint32_t POWER_RECOVERY_POLL_INTERVAL_MS = 30000UL;
static constexpr uint32_t POWER_STATUS_LED_TOGGLE_INTERVAL_MS = 10000UL;
static constexpr uint32_t POWER_SAFETY_DISPLAY_REFRESH_INTERVAL_MS = 10000UL;
static constexpr uint32_t ORIN_FORCE_OFF_MS = 60000UL;
static constexpr uint32_t ORIN_ACKED_OFF_MS = 15000UL;

static constexpr uint8_t EEPROM_POWER_SAFETY_INVALID_MAGIC = 0x00;
static constexpr uint8_t EEPROM_POWER_SAFETY_MAGIC = 0xC9;
static constexpr uint8_t EEPROM_POWER_SAFETY_SCHEMA = 1;

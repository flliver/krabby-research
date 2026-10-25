#pragma once

#include <stdint.h>

static constexpr uint8_t LSM6DSO_CTRL1_XL_REGISTER = 0x10;
static constexpr uint8_t LSM6DSO_CTRL2_G_REGISTER = 0x11;
static constexpr uint8_t LSM6DSO_CTRL3_C_REGISTER = 0x12;
static constexpr uint8_t LSM6DSO_CTRL1_XL_CONFIG_MASK = 0xFC;
static constexpr uint8_t LSM6DSO_CTRL1_XL_EXPECTED = 0x6C;
static constexpr uint8_t LSM6DSO_CTRL2_G_CONFIG_MASK = 0xFE;
static constexpr uint8_t LSM6DSO_CTRL2_G_EXPECTED = 0x64;
static constexpr uint8_t LSM6DSO_CTRL3_C_CONFIG_MASK = 0x44;
static constexpr uint8_t LSM6DSO_CTRL3_C_EXPECTED = 0x44;

inline bool lsm6dsoConfigurationMatches(
    uint8_t accelerometerConfiguration,
    uint8_t gyroscopeConfiguration,
    uint8_t commonConfiguration)
{
    return (accelerometerConfiguration & LSM6DSO_CTRL1_XL_CONFIG_MASK) == LSM6DSO_CTRL1_XL_EXPECTED &&
           (gyroscopeConfiguration & LSM6DSO_CTRL2_G_CONFIG_MASK) == LSM6DSO_CTRL2_G_EXPECTED &&
           (commonConfiguration & LSM6DSO_CTRL3_C_CONFIG_MASK) == LSM6DSO_CTRL3_C_EXPECTED;
}

#pragma once

#include <stdint.h>

static constexpr uint16_t EEPROM_CAPACITY_BYTES = 4096;

static constexpr uint16_t EEPROM_ACTUATOR_CAL_ADDR = 0;
static constexpr uint16_t EEPROM_ACTUATOR_CAL_SIZE = 26;
static constexpr uint16_t EEPROM_ACTUATOR_CAL_NEXT_ADDR =
    EEPROM_ACTUATOR_CAL_ADDR + EEPROM_ACTUATOR_CAL_SIZE;

static constexpr uint16_t EEPROM_ROLE_ADDR = 32;
static constexpr uint16_t EEPROM_ROLE_SIZE = 2;
static constexpr uint16_t EEPROM_ROLE_NEXT_ADDR =
    EEPROM_ROLE_ADDR + EEPROM_ROLE_SIZE;

static constexpr uint16_t EEPROM_IMU_CAL_ADDR = 40;
static constexpr uint16_t EEPROM_IMU_CAL_SIZE = 26;
static constexpr uint16_t EEPROM_IMU_CAL_NEXT_ADDR =
    EEPROM_IMU_CAL_ADDR + EEPROM_IMU_CAL_SIZE;

static constexpr uint16_t EEPROM_POWER_CAL_ADDR = EEPROM_IMU_CAL_NEXT_ADDR;
static constexpr uint16_t EEPROM_POWER_CAL_SIZE = 14;
static constexpr uint16_t EEPROM_POWER_CAL_NEXT_ADDR =
    EEPROM_POWER_CAL_ADDR + EEPROM_POWER_CAL_SIZE;

static constexpr uint16_t EEPROM_NEXT_AVAILABLE_ADDR =
    EEPROM_POWER_CAL_NEXT_ADDR;

static_assert(EEPROM_ACTUATOR_CAL_NEXT_ADDR <= EEPROM_ROLE_ADDR,
              "actuator calibration overlaps role data");
static_assert(EEPROM_ROLE_NEXT_ADDR <= EEPROM_IMU_CAL_ADDR,
              "role data overlaps IMU calibration");
static_assert(EEPROM_IMU_CAL_NEXT_ADDR <= EEPROM_POWER_CAL_ADDR,
              "IMU calibration overlaps power calibration");
static_assert(EEPROM_NEXT_AVAILABLE_ADDR <= EEPROM_CAPACITY_BYTES,
              "EEPROM layout exceeds device capacity");

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "../units/angular_units.h"
#include "../units/inertial_units.h"

// body[i] = IMU_AXIS_SIGN[i] * sensor[IMU_AXIS_SRC[i]]. Identity until the
// mounting orientation is fixed; see firmware/SETUP.md.
static constexpr uint8_t IMU_AXIS_SRC[3] = {0, 1, 2};
static constexpr int8_t IMU_AXIS_SIGN[3] = {1, 1, 1};

// sqrt(200) ~= 14x noise reduction on the bias estimate. The interval is >= one
// 416 Hz period so samples are independent; 200 x 5 ms ~= 1 s capture.
static constexpr uint16_t IMU_CAL_SAMPLES = 200;
static constexpr uint16_t IMU_CAL_SAMPLE_INTERVAL_MS = 5;

static constexpr float IMU_CAL_MAX_SPREAD_DPS = 2.0f;
static constexpr float IMU_CAL_MAX_BIAS_DPS = 10.0f;

// Delay and rate-limit reconfiguration after a failed transfer.
static constexpr uint8_t IMU_BAD_TICKS_BEFORE_RECOVERY = 3;
static constexpr uint32_t IMU_RECOVERY_RETRY_INTERVAL_MS = 1000UL;

// Joint calibration occupies 0-25, role data 32-33
static constexpr uint16_t EEPROM_IMU_CAL_ADDR = 40;
static constexpr uint16_t EEPROM_IMU_CAL_SIZE = 26;
static constexpr uint16_t EEPROM_SENSOR_CAL_NEXT_ADDR = EEPROM_IMU_CAL_ADDR + EEPROM_IMU_CAL_SIZE;

// 0xC7 is distinct from the role magic (0xAB) and from erased EEPROM (0xFF).
static constexpr uint8_t EEPROM_IMU_CAL_INVALID_MAGIC = 0x00;
static constexpr uint8_t EEPROM_IMU_CAL_MAGIC = 0xC7;
static constexpr uint8_t EEPROM_IMU_CAL_SCHEMA = 1;

// 0x6A when the ADR/SA0 jumper is cut.
static constexpr uint8_t LSM6DSO_PRIMARY_ADDRESS = 0x6B;
static constexpr uint8_t LSM6DSO_ALTERNATE_ADDRESS = 0x6A;

static constexpr uint8_t LSM6DSO_OUTPUT_START_REGISTER = 0x20;
static constexpr uint8_t LSM6DSO_NUM_SAMPLE_BYTES = 14;

// Standard-mode I2C, the shared bus's normal rate.
static constexpr uint32_t I2C_DEFAULT_BUS_CLOCK_HZ = 100000UL;

// Exceeds the longest transfer (~2.25 ms at 100 kHz), stays under the 50 ms tick.
static constexpr uint32_t I2C_BUS_TIMEOUT_MICROSECONDS = 10000UL;

static constexpr uint16_t LSM6DSO_TURN_ON_TIME_MS = 5;

// 416 Hz oversamples the 20 Hz telemetry tick. Auto-increment is required for
// the burst read; block-data-update prevents a read mixing high and low bytes
// from different samples.
static constexpr bool LSM6DSO_AUTO_INCREMENT_ENABLED = true;
static constexpr bool LSM6DSO_BLOCK_DATA_UPDATE_ENABLED = true;
static constexpr uint8_t LSM6DSO_ACCELERATION_RANGE_G = 8;
static constexpr uint16_t LSM6DSO_ACCELERATION_DATA_RATE_HZ = 416;
static constexpr uint16_t LSM6DSO_ANGULAR_RATE_RANGE_DEGREES_PER_SECOND = 500;
static constexpr uint16_t LSM6DSO_ANGULAR_RATE_DATA_RATE_HZ = 416;

static constexpr float LSM6DSO_ACCELERATION_METERS_PER_SECOND_SQUARED_PER_LSB =
    0.000244f * METERS_PER_SECOND_SQUARED_PER_G;
static constexpr float LSM6DSO_ANGULAR_RATE_RADIANS_PER_SECOND_PER_LSB =
    0.0175f * RADIANS_PER_DEGREE;
static constexpr float LSM6DSO_TEMPERATURE_CELSIUS_PER_LSB = 1.0f / 256.0f;
static constexpr float LSM6DSO_TEMPERATURE_OFFSET_CELSIUS = 25.0f;

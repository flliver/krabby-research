#pragma once

#include <SparkFunLSM6DSO.h>
#include <Wire.h>
#include <stdint.h>

#include "../../byte_order.h"
#include "../i2c/arduino_i2c_bus.h"
#include "../i2c/i2c_recovery.h"
#include "imu_calibrator.h"
#include "lsm6dso_configuration.h"
#include "imu_constants.h"

enum class Lsm6dsoInitializationResult : uint8_t
{
    Ok,
    NotDetected,
    ConfigurationFailed,
};

class Lsm6dsoAdapter
{
public:
    Lsm6dsoAdapter()
        : calibrator_{},
          recoveryPolicy_{},
          stuckBusLatch_{},
          isInitialized_(false),
          address_(0)
    {
    }

    Lsm6dsoInitializationResult initialize()
    {
        calibrator_ = ImuCalibrator{};
        recoveryPolicy_ = I2cRecoveryPolicy{};
        stuckBusLatch_ = I2cStuckBusLatch{};
        return configureSensor();
    }

    template <typename Storage>
    ImuCalibrationResult calibrate(
        Storage &storage,
        void (*delayMilliseconds)(unsigned long))
    {
        return calibrator_.calibrate(
            [this]() { return readSensorMeasurement(); },
            storage,
            delayMilliseconds);
    }

    ImuMeasurement measure()
    {
        const I2cRecoveryLimits limits = {
            IMU_BAD_TICKS_BEFORE_RECOVERY,
            IMU_RECOVERY_RETRY_INTERVAL_MS,
        };

        if (!isInitialized_)
        {
            if (!recoveryPolicy_.noteFailure(millis(), limits) ||
                !recoverAndConfigure())
            {
                return ImuMeasurement{false};
            }
            recoveryPolicy_.noteSuccess();
        }

        const ImuMeasurement bodyMeasurement = readSensorMeasurement();
        if (!bodyMeasurement.didSucceed())
        {
            // A replugged sensor may ACK while its outputs remain powered down.
            isInitialized_ = false;
            recoveryPolicy_.noteFailure(millis(), limits);
            return bodyMeasurement;
        }

        recoveryPolicy_.noteSuccess();
        return calibrator_.applyImuCalibration(bodyMeasurement);
    }

private:
    static bool motionIsAllZero(const ImuMeasurement &measurement)
    {
        for (uint8_t axis = 0; axis < 3; ++axis)
            if (measurement.acceleration[axis].value() != 0.0f ||
                measurement.angularRate[axis].value() != 0.0f)
                return false;
        return true;
    }

    Lsm6dsoInitializationResult configureSensor()
    {
        isInitialized_ = false;
        address_ = 0;
        Wire.begin();
        Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
        Wire.setWireTimeout(I2C_BUS_TIMEOUT_MICROSECONDS, true);

        address_ = LSM6DSO_PRIMARY_ADDRESS;
        if (!driver_.begin(address_))
        {
            address_ = LSM6DSO_ALTERNATE_ADDRESS;
            if (!driver_.begin(address_))
            {
                address_ = 0;
                return Lsm6dsoInitializationResult::NotDetected;
            }
        }

        const bool isConfigured =
            driver_.setIncrement(LSM6DSO_AUTO_INCREMENT_ENABLED) &&
            driver_.setAccelRange(LSM6DSO_ACCELERATION_RANGE_G) &&
            driver_.setAccelDataRate(LSM6DSO_ACCELERATION_DATA_RATE_HZ) &&
            driver_.setGyroRange(LSM6DSO_ANGULAR_RATE_RANGE_DEGREES_PER_SECOND) &&
            driver_.setGyroDataRate(LSM6DSO_ANGULAR_RATE_DATA_RATE_HZ) &&
            driver_.setBlockDataUpdate(LSM6DSO_BLOCK_DATA_UPDATE_ENABLED);

        if (!isConfigured)
        {
            address_ = 0;
            return Lsm6dsoInitializationResult::ConfigurationFailed;
        }

        delay(LSM6DSO_TURN_ON_TIME_MS);

        isInitialized_ = true;
        return Lsm6dsoInitializationResult::Ok;
    }

    bool recoverAndConfigure()
    {
        // Relinquish the pins before manual bus recovery.
        Wire.end();
        ArduinoI2cBus bus(
            I2C_DEFAULT_BUS_CLOCK_HZ,
            I2C_BUS_TIMEOUT_MICROSECONDS);

        // A failed clear latches recovery until SDA is released.
        if (!stuckBusLatch_.mayAttempt(bus.isSdaHigh()))
            return false;

        const I2cBusRecovery busResult = recoverI2cBus(bus);
        stuckBusLatch_.noteResult(busResult);
        if (busResult == I2cBusRecovery::Stuck)
            return false;

        return configureSensor() == Lsm6dsoInitializationResult::Ok;
    }

    ImuMeasurement readSensorMeasurement()
    {
        if (address_ == 0)
            return ImuMeasurement{false};

        Wire.beginTransmission(address_);
        if (Wire.write(LSM6DSO_OUTPUT_START_REGISTER) != 1)
            return ImuMeasurement{false};

        if (Wire.endTransmission(false) != 0)
            return ImuMeasurement{false};

        if (Wire.requestFrom(
                address_,
                LSM6DSO_NUM_SAMPLE_BYTES,
                static_cast<uint8_t>(true)) != LSM6DSO_NUM_SAMPLE_BYTES)
        {
            return ImuMeasurement{false};
        }

        uint8_t bytes[LSM6DSO_NUM_SAMPLE_BYTES];
        for (uint8_t index = 0; index < LSM6DSO_NUM_SAMPLE_BYTES; ++index)
        {
            if (Wire.available() <= 0)
                return ImuMeasurement{false};
            bytes[index] = static_cast<uint8_t>(Wire.read());
        }

        ImuMeasurement measurement{true};
        const int16_t rawTemperature = decodeInt16LittleEndian(&bytes[0]);
        measurement.temperature = Celsius(
            rawTemperature * LSM6DSO_TEMPERATURE_CELSIUS_PER_LSB +
            LSM6DSO_TEMPERATURE_OFFSET_CELSIUS);

        for (uint8_t axis = 0; axis < 3; ++axis)
        {
            const int16_t rawAngularRate =
                decodeInt16LittleEndian(&bytes[2 + axis * 2]);
            const int16_t rawAcceleration =
                decodeInt16LittleEndian(&bytes[8 + axis * 2]);

            measurement.angularRate[axis] = RadiansPerSecond(
                rawAngularRate * LSM6DSO_ANGULAR_RATE_RADIANS_PER_SECOND_PER_LSB);
            measurement.acceleration[axis] = MetersPerSecondSquared(
                rawAcceleration *
                LSM6DSO_ACCELERATION_METERS_PER_SECOND_SQUARED_PER_LSB);
        }

        // A reset sensor may still acknowledge I2C while its powered-down outputs return zero.
        if (motionIsAllZero(measurement) && !sensorConfigurationMatches())
            return ImuMeasurement{false};

        return transformImuMeasurementToBodyFrame(measurement);
    }

    bool sensorConfigurationMatches()
    {
        uint8_t accelerometerConfiguration = 0;
        uint8_t gyroscopeConfiguration = 0;
        uint8_t commonConfiguration = 0;
        return driver_.readRegister(&accelerometerConfiguration, LSM6DSO_CTRL1_XL_REGISTER) == IMU_SUCCESS &&
               driver_.readRegister(&gyroscopeConfiguration, LSM6DSO_CTRL2_G_REGISTER) == IMU_SUCCESS &&
               driver_.readRegister(&commonConfiguration, LSM6DSO_CTRL3_C_REGISTER) == IMU_SUCCESS &&
               lsm6dsoConfigurationMatches(
                   accelerometerConfiguration,
                   gyroscopeConfiguration,
                   commonConfiguration);
    }

    LSM6DSO driver_;
    ImuCalibrator calibrator_;
    I2cRecoveryPolicy recoveryPolicy_;
    I2cStuckBusLatch stuckBusLatch_;
    bool isInitialized_;
    uint8_t address_;
};

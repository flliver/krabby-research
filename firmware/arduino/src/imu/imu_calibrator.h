#pragma once

#include <math.h>
#include <stdint.h>

#include "imu_constants.h"
#include "imu_measurement.h"

struct ImuCalibrationRecord
{
    uint8_t magic;
    uint8_t schema;
    float gyroBiasDegreesPerSecond[3];
    float accelBiasG[3];
};

enum class ImuCalibrationResult : uint8_t
{
    Loaded,
    Captured,
    ReadFailed,
    MotionDetected,
    VerificationFailed,
};

struct ImuCalibrationAccumulator
{
    ImuCalibrationAccumulator()
        : sum{},
          minimum{},
          maximum{},
          sampleCount(0)
    {
    }

    DegreesPerSecond sum[3];
    DegreesPerSecond minimum[3];
    DegreesPerSecond maximum[3];
    uint16_t sampleCount;
};

class ImuCalibrator
{
public:
    ImuCalibrator()
        : calibration_{}
    {
    }

    // Source and storage are template parameters so this class names neither
    // Wire nor EEPROM and compiles on the host.
    template <typename MeasurementSource, typename Storage>
    ImuCalibrationResult calibrate(
        MeasurementSource readBodyMeasurement,
        Storage &storage,
        void (*delayMilliseconds)(unsigned long))
    {
        storage.load(calibration_);
        if (isValidRecord(calibration_))
            return ImuCalibrationResult::Loaded;

        calibration_ = ImuCalibrationRecord{};
        ImuCalibrationAccumulator accumulator;
        for (uint16_t sampleIndex = 0; sampleIndex < IMU_CAL_SAMPLES; ++sampleIndex)
        {
            const ImuMeasurement bodyMeasurement = readBodyMeasurement();
            if (!addCalibrationSample(accumulator, bodyMeasurement))
                return ImuCalibrationResult::ReadFailed;
            delayMilliseconds(IMU_CAL_SAMPLE_INTERVAL_MS);
        }

        const ImuCalibrationResult captureResult =
            calculateCalibration(accumulator, calibration_);
        if (captureResult != ImuCalibrationResult::Captured)
            return captureResult;

        return persistCalibration(storage, calibration_);
    }

    ImuMeasurement applyImuCalibration(const ImuMeasurement &measurement) const
    {
        if (!measurement.didSucceed())
            return measurement;

        ImuMeasurement calibratedMeasurement = measurement;
        for (uint8_t axis = 0; axis < 3; ++axis)
        {
            calibratedMeasurement.acceleration[axis] =
                measurement.acceleration[axis] -
                MetersPerSecondSquared(
                    calibration_.accelBiasG[axis] *
                    METERS_PER_SECOND_SQUARED_PER_G);
            const RadiansPerSecond gyroBias =
                DegreesPerSecond(calibration_.gyroBiasDegreesPerSecond[axis])
                    .toRadiansPerSecond();
            calibratedMeasurement.angularRate[axis] =
                measurement.angularRate[axis] - gyroBias;
        }

        return calibratedMeasurement;
    }

private:
    static bool isValidRecord(const ImuCalibrationRecord &calibration_)
    {
        if (calibration_.magic != EEPROM_IMU_CAL_MAGIC ||
            calibration_.schema != EEPROM_IMU_CAL_SCHEMA)
        {
            return false;
        }

        for (uint8_t axis = 0; axis < 3; ++axis)
        {
            if (!isfinite(calibration_.gyroBiasDegreesPerSecond[axis]) ||
                fabs(calibration_.gyroBiasDegreesPerSecond[axis]) > IMU_CAL_MAX_BIAS_DPS)
            {
                return false;
            }

            if (!isfinite(calibration_.accelBiasG[axis]))
                return false;
        }

        return true;
    }

    static bool addCalibrationSample(
        ImuCalibrationAccumulator &accumulator,
        const ImuMeasurement &measurement)
    {
        if (!measurement.didSucceed())
            return false;

        for (uint8_t axis = 0; axis < 3; ++axis)
        {
            const DegreesPerSecond rate =
                measurement.angularRate[axis].toDegreesPerSecond();
            accumulator.sum[axis] = accumulator.sum[axis] + rate;

            if (accumulator.sampleCount == 0 ||
                rate < accumulator.minimum[axis])
            {
                accumulator.minimum[axis] = rate;
            }

            if (accumulator.sampleCount == 0 ||
                rate > accumulator.maximum[axis])
            {
                accumulator.maximum[axis] = rate;
            }
        }

        ++accumulator.sampleCount;
        return true;
    }

    static ImuCalibrationResult calculateCalibration(
        const ImuCalibrationAccumulator &accumulator,
        ImuCalibrationRecord &record)
    {
        for (uint8_t axis = 0; axis < 3; ++axis)
        {
            if (accumulator.maximum[axis] - accumulator.minimum[axis] >
                DegreesPerSecond(IMU_CAL_MAX_SPREAD_DPS))
            {
                return ImuCalibrationResult::MotionDetected;
            }

            record.gyroBiasDegreesPerSecond[axis] =
                accumulator.sum[axis]
                    .scalarDivide(IMU_CAL_SAMPLES)
                    .value();
        }

        record.schema = EEPROM_IMU_CAL_SCHEMA;
        return ImuCalibrationResult::Captured;
    }

    template <typename Storage>
    static ImuCalibrationResult persistCalibration(
        Storage &storage,
        ImuCalibrationRecord &record)
    {
        record.magic = EEPROM_IMU_CAL_INVALID_MAGIC;
        storage.writeRecord(record);

        storage.updateMagic(EEPROM_IMU_CAL_MAGIC);
        storage.load(record);

        if (!isValidRecord(record))
        {
            record = ImuCalibrationRecord{};
            return ImuCalibrationResult::VerificationFailed;
        }

        return ImuCalibrationResult::Captured;
    }

    ImuCalibrationRecord calibration_;
};

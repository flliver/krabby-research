#include <limits>
#include <vector>

#include "src/imu/imu_constants.h"
#include "src/imu/imu_calibrator.h"
#include "unity.h"

struct RecordingStorage
{
    enum class Operation : uint8_t { Load, WriteRecord, UpdateMagic };

    ImuCalibrationRecord record{};
    std::vector<Operation> operations;
    std::vector<uint8_t> writtenMagic;
    bool shouldCorruptVerificationRead = false;

    void load(ImuCalibrationRecord &out)
    {
        operations.push_back(Operation::Load);
        out = record;
        if (shouldCorruptVerificationRead) out.schema = 0;
    }

    void writeRecord(const ImuCalibrationRecord &written)
    {
        operations.push_back(Operation::WriteRecord);
        writtenMagic.push_back(written.magic);
        record = written;
    }

    void updateMagic(uint8_t magic)
    {
        operations.push_back(Operation::UpdateMagic);
        writtenMagic.push_back(magic);
        record.magic = magic;
    }
};

static void noDelay(unsigned long) {}

static ImuMeasurement gyroMeasurement(float x, float y, float z)
{
    ImuMeasurement result{true};
    result.angularRate[0] = DegreesPerSecond(x).toRadiansPerSecond();
    result.angularRate[1] = DegreesPerSecond(y).toRadiansPerSecond();
    result.angularRate[2] = DegreesPerSecond(z).toRadiansPerSecond();
    result.acceleration[2] =
        MetersPerSecondSquared(METERS_PER_SECOND_SQUARED_PER_G);
    return result;
}

static ImuCalibrationRecord validStoredRecord()
{
    ImuCalibrationRecord record = {
        EEPROM_IMU_CAL_MAGIC, EEPROM_IMU_CAL_SCHEMA, {0, 0, 0}, {0, 0, 0}};
    return record;
}

// Captures from a constant gyro reading unless storage already holds a record
// calibrate() accepts, in which case it reloads.
static ImuCalibrationResult calibrateConstant(
    RecordingStorage &storage,
    ImuCalibrator &calibrator,
    float x,
    float y,
    float z)
{
    return calibrator.calibrate(
        [&]() { return gyroMeasurement(x, y, z); }, storage, noDelay);
}

void setUp() {}
void tearDown() {}

static void test_stored_bias_within_bounds_is_reused_not_recaptured()
{
    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        for (int sign = -1; sign <= 1; sign += 2)
        {
            for (int overBoundary = 0; overBoundary <= 1; ++overBoundary)
            {
                RecordingStorage storage;
                storage.record = validStoredRecord();
                storage.record.gyroBiasDegreesPerSecond[axis] =
                    sign * (IMU_CAL_MAX_BIAS_DPS + (overBoundary ? 0.01f : 0.0f));

                ImuCalibrator calibrator;
                TEST_ASSERT_EQUAL(
                    static_cast<int>(overBoundary ? ImuCalibrationResult::Captured
                                                  : ImuCalibrationResult::Loaded),
                    static_cast<int>(
                        calibrateConstant(storage, calibrator, 0, 0, 0)));
            }
        }
    }
}

static void test_each_invalid_stored_record_forces_a_recapture()
{
    for (int failure = 0; failure < 6; ++failure)
    {
        RecordingStorage storage;
        storage.record = validStoredRecord();
        if (failure == 0) storage.record.magic = EEPROM_IMU_CAL_INVALID_MAGIC;
        if (failure == 1) storage.record.schema = 0;
        if (failure == 2) storage.record.gyroBiasDegreesPerSecond[0] =
            std::numeric_limits<float>::quiet_NaN();
        if (failure == 3) storage.record.gyroBiasDegreesPerSecond[1] =
            std::numeric_limits<float>::infinity();
        if (failure == 4) storage.record.accelBiasG[1] =
            std::numeric_limits<float>::quiet_NaN();
        if (failure == 5) storage.record.accelBiasG[2] =
            -std::numeric_limits<float>::infinity();

        ImuCalibrator calibrator;
        TEST_ASSERT_EQUAL(
            static_cast<int>(ImuCalibrationResult::Captured),
            static_cast<int>(calibrateConstant(storage, calibrator, 0, 0, 0)));
    }
}

static void test_read_failure_at_every_capture_position_aborts_unsaved()
{
    for (uint16_t failAt = 0; failAt < IMU_CAL_SAMPLES; ++failAt)
    {
        RecordingStorage storage;
        ImuCalibrator calibrator;
        uint16_t index = 0;

        const ImuCalibrationResult result = calibrator.calibrate(
            [&]() {
                return index++ == failAt ? ImuMeasurement{false}
                                         : gyroMeasurement(0, 0, 0);
            },
            storage,
            noDelay);

        TEST_ASSERT_EQUAL(static_cast<int>(ImuCalibrationResult::ReadFailed),
                          static_cast<int>(result));
        // Only the startup load ran; nothing was written.
        TEST_ASSERT_EQUAL_UINT(1, storage.operations.size());
    }
}

static void test_capture_averages_all_axes_and_the_bias_is_applied()
{
    RecordingStorage storage;
    ImuCalibrator calibrator;
    TEST_ASSERT_EQUAL(
        static_cast<int>(ImuCalibrationResult::Captured),
        static_cast<int>(
            calibrateConstant(storage, calibrator, 1.0f, -2.0f, 4.0f)));

    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 1.0f,
                             storage.record.gyroBiasDegreesPerSecond[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, -2.0f,
                             storage.record.gyroBiasDegreesPerSecond[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 4.0f,
                             storage.record.gyroBiasDegreesPerSecond[2]);
    TEST_ASSERT_EQUAL_HEX8(EEPROM_IMU_CAL_SCHEMA, storage.record.schema);

    const ImuMeasurement corrected =
        calibrator.applyImuCalibration(gyroMeasurement(1.0f, -2.0f, 4.0f));
    for (uint8_t axis = 0; axis < 3; ++axis)
        TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f,
                                 corrected.angularRate[axis].value());
}

static void test_an_invalid_measurement_passes_through_uncalibrated()
{
    ImuCalibrator calibrator;
    const ImuMeasurement measurement{false};
    const ImuMeasurement result = calibrator.applyImuCalibration(measurement);
    TEST_ASSERT_FALSE(result.didSucceed());
}

static void test_motion_spread_boundary_is_inclusive()
{
    for (int overBoundary = 0; overBoundary <= 1; ++overBoundary)
    {
        RecordingStorage storage;
        ImuCalibrator calibrator;
        uint16_t index = 0;

        const ImuCalibrationResult result = calibrator.calibrate(
            [&]() {
                const bool isLast = ++index == IMU_CAL_SAMPLES;
                return gyroMeasurement(
                    isLast ? IMU_CAL_MAX_SPREAD_DPS +
                                 (overBoundary ? 0.01f : 0.0f)
                           : 0.0f,
                    0,
                    0);
            },
            storage,
            noDelay);

        TEST_ASSERT_EQUAL(
            static_cast<int>(overBoundary ? ImuCalibrationResult::MotionDetected
                                          : ImuCalibrationResult::Captured),
            static_cast<int>(result));
    }
}

static void test_persistence_is_invalid_first_magic_last_then_verified()
{
    RecordingStorage storage;
    ImuCalibrator calibrator;
    TEST_ASSERT_EQUAL(
        static_cast<int>(ImuCalibrationResult::Captured),
        static_cast<int>(calibrateConstant(storage, calibrator, 0, 0, 0)));

    // Startup load, then write, then magic, then the verification read.
    TEST_ASSERT_EQUAL_UINT(4, storage.operations.size());
    TEST_ASSERT_EQUAL(static_cast<int>(RecordingStorage::Operation::WriteRecord),
                      static_cast<int>(storage.operations[1]));
    TEST_ASSERT_EQUAL(static_cast<int>(RecordingStorage::Operation::UpdateMagic),
                      static_cast<int>(storage.operations[2]));
    TEST_ASSERT_EQUAL(static_cast<int>(RecordingStorage::Operation::Load),
                      static_cast<int>(storage.operations[3]));
    TEST_ASSERT_EQUAL_HEX8(EEPROM_IMU_CAL_INVALID_MAGIC, storage.writtenMagic[0]);
    TEST_ASSERT_EQUAL_HEX8(EEPROM_IMU_CAL_MAGIC, storage.writtenMagic[1]);
}

static void test_persistence_verification_failure_is_reported()
{
    RecordingStorage storage;
    storage.shouldCorruptVerificationRead = true;
    ImuCalibrator calibrator;
    TEST_ASSERT_EQUAL(
        static_cast<int>(ImuCalibrationResult::VerificationFailed),
        static_cast<int>(calibrateConstant(storage, calibrator, 0, 0, 0)));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_stored_bias_within_bounds_is_reused_not_recaptured);
    RUN_TEST(test_each_invalid_stored_record_forces_a_recapture);
    RUN_TEST(test_read_failure_at_every_capture_position_aborts_unsaved);
    RUN_TEST(test_capture_averages_all_axes_and_the_bias_is_applied);
    RUN_TEST(test_an_invalid_measurement_passes_through_uncalibrated);
    RUN_TEST(test_motion_spread_boundary_is_inclusive);
    RUN_TEST(test_persistence_is_invalid_first_magic_last_then_verified);
    RUN_TEST(test_persistence_verification_failure_is_reported);
    return UNITY_END();
}

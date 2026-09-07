#include "unity.h"

#include "src/imu/imu_measurement.h"
#include "src/imu/lsm6dso_configuration.h"

void setUp() {}
void tearDown() {}

static ImuMeasurement accelerationSample(float ax, float ay, float az)
{
    ImuMeasurement measurement{true};
    measurement.acceleration[0] = MetersPerSecondSquared(ax);
    measurement.acceleration[1] = MetersPerSecondSquared(ay);
    measurement.acceleration[2] = MetersPerSecondSquared(az);
    return measurement;
}

static void recognizes_expected_sensor_configuration()
{
    TEST_ASSERT_TRUE(lsm6dsoConfigurationMatches(0x6C, 0x64, 0x44));
    TEST_ASSERT_TRUE(lsm6dsoConfigurationMatches(0x6D, 0x65, 0x45));
}

static void rejects_reset_or_incomplete_sensor_configuration()
{
    TEST_ASSERT_FALSE(lsm6dsoConfigurationMatches(0x00, 0x00, 0x00));
    TEST_ASSERT_FALSE(lsm6dsoConfigurationMatches(0x0C, 0x64, 0x44));
    TEST_ASSERT_FALSE(lsm6dsoConfigurationMatches(0x6C, 0x04, 0x44));
    TEST_ASSERT_FALSE(lsm6dsoConfigurationMatches(0x6C, 0x64, 0x40));
}

static void calculates_accelerometer_tilt_for_known_orientations()
{
    const float g = METERS_PER_SECOND_SQUARED_PER_G;

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f,
        computeRollFromAcceleration(accelerationSample(0, 0, g)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f,
        computePitchFromAcceleration(accelerationSample(0, 0, g)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 90.0f,
        computeRollFromAcceleration(accelerationSample(0, g, 0)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -90.0f,
        computeRollFromAcceleration(accelerationSample(0, -g, 0)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 180.0f,
        computeRollFromAcceleration(accelerationSample(0, 0, -g)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 90.0f,
        computePitchFromAcceleration(accelerationSample(-g, 0, 0)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -90.0f,
        computePitchFromAcceleration(accelerationSample(g, 0, 0)).value());

    // Pitch uses the magnitude of the y/z pair, so rolling does not tilt it.
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f,
        computePitchFromAcceleration(accelerationSample(0, g, 0)).value());

    const float diagonal = g * 0.70710678f;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 45.0f,
        computeRollFromAcceleration(accelerationSample(0, diagonal, diagonal)).value());
}

static void preserves_fractional_tilt_for_consumers()
{
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.710593f,
        computeRollFromAcceleration(accelerationSample(0, 0.1f, 1.0f)).value());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -5.710593f,
        computeRollFromAcceleration(accelerationSample(0, -0.1f, 1.0f)).value());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(recognizes_expected_sensor_configuration);
    RUN_TEST(rejects_reset_or_incomplete_sensor_configuration);
    RUN_TEST(calculates_accelerometer_tilt_for_known_orientations);
    RUN_TEST(preserves_fractional_tilt_for_consumers);
    return UNITY_END();
}

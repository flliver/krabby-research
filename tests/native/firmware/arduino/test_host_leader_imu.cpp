#include "unity.h"

#include "host_leader_imu.hpp"
#include "src/imu/imu_measurement.h"
#include "sim_clock.hpp"
#include "sim_imu_model.hpp"
#include "sim_i2c.hpp"

using krabby::sim::SimImuBodyState;
using krabby::sim::host::leaderImu;
using krabby::sim::setSimImuBodyState;
using krabby::sim::simI2cBus;

void setUp() { krabby::sim::resetHostSim(); }

void tearDown() {}

static void test_measure_stationary_gravity_on_z()
{
    leaderImu().setup();
    TEST_ASSERT_TRUE(leaderImu().isOnline());

    const ImuMeasurement measurement = leaderImu().measure();
    TEST_ASSERT_TRUE(measurement.didSucceed());
    TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.0f, measurement.acceleration[0].value());
    TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.0f, measurement.acceleration[1].value());
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 9.80665f, measurement.acceleration[2].value());
    TEST_ASSERT_FLOAT_WITHIN(0.2f, 25.0f, measurement.temperature.value());
}

static void test_measure_follows_sim_body_tilt()
{
    SimImuBodyState tilted{};
    tilted.accelerationMps2[0] = 9.80665f;
    tilted.accelerationMps2[2] = 0.0f;
    setSimImuBodyState(tilted);

    leaderImu().setup();
    const ImuMeasurement measurement = leaderImu().measure();
    TEST_ASSERT_TRUE(measurement.didSucceed());
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 9.80665f, measurement.acceleration[0].value());
    TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.0f, measurement.acceleration[2].value());
}

static void test_measure_invalid_when_imu_absent()
{
    simI2cBus().setDevicePresent(krabby::sim::kI2cAddrLsm6dsoPrimary, false);
    simI2cBus().setDevicePresent(krabby::sim::kI2cAddrLsm6dsoAlternate, false);
    leaderImu().setup();
    TEST_ASSERT_FALSE(leaderImu().isOnline());
    TEST_ASSERT_FALSE(leaderImu().measure().didSucceed());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_measure_stationary_gravity_on_z);
    RUN_TEST(test_measure_follows_sim_body_tilt);
    RUN_TEST(test_measure_invalid_when_imu_absent);
    return UNITY_END();
}

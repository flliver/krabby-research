#include "sim_imu_model.hpp"

#include "src/imu/imu_constants.h"
#include "src/imu/lsm6dso_configuration.h"

#include <cmath>
#include <cstring>

namespace krabby {
namespace sim {

namespace {

SimImuBodyState g_body_state;

int16_t encodeTemperatureRaw(float celsius)
{
    const float raw =
        (celsius - LSM6DSO_TEMPERATURE_OFFSET_CELSIUS) / LSM6DSO_TEMPERATURE_CELSIUS_PER_LSB;
    return static_cast<int16_t>(std::lround(raw));
}

int16_t encodeAngularRateRaw(float radPerSecond)
{
    const float raw = radPerSecond / LSM6DSO_ANGULAR_RATE_RADIANS_PER_SECOND_PER_LSB;
    return static_cast<int16_t>(std::lround(raw));
}

int16_t encodeAccelerationRaw(float metersPerSecondSquared)
{
    const float raw =
        metersPerSecondSquared / LSM6DSO_ACCELERATION_METERS_PER_SECOND_SQUARED_PER_LSB;
    return static_cast<int16_t>(std::lround(raw));
}

void writeInt16Le(int16_t value, uint8_t *out)
{
    out[0] = static_cast<uint8_t>(value & 0xFF);
    out[1] = static_cast<uint8_t>((value >> 8) & 0xFF);
}

}  // namespace

void resetSimImuModel() { g_body_state = SimImuBodyState{}; }

const SimImuBodyState &simImuBodyState() { return g_body_state; }

void setSimImuBodyState(const SimImuBodyState &state) { g_body_state = state; }

void encodeLsm6dsoOutputBurst(const SimImuBodyState &state, uint8_t outBytes[14])
{
    std::memset(outBytes, 0, 14);
    writeInt16Le(encodeTemperatureRaw(state.temperatureCelsius), &outBytes[0]);
    for (uint8_t axis = 0; axis < 3; ++axis)
    {
        writeInt16Le(
            encodeAngularRateRaw(state.angularRateRadS[axis]),
            &outBytes[2 + axis * 2]);
        writeInt16Le(
            encodeAccelerationRaw(state.accelerationMps2[axis]),
            &outBytes[8 + axis * 2]);
    }
}

uint8_t lsm6dsoRegisterByte(uint8_t reg)
{
    if (reg == 0x0Fu)
    {
        return 0x6Cu;
    }
    if (reg == LSM6DSO_CTRL1_XL_REGISTER)
    {
        return LSM6DSO_CTRL1_XL_EXPECTED;
    }
    if (reg == LSM6DSO_CTRL2_G_REGISTER)
    {
        return LSM6DSO_CTRL2_G_EXPECTED;
    }
    if (reg == LSM6DSO_CTRL3_C_REGISTER)
    {
        return LSM6DSO_CTRL3_C_EXPECTED;
    }
    if (reg >= LSM6DSO_OUTPUT_START_REGISTER &&
        reg < LSM6DSO_OUTPUT_START_REGISTER + LSM6DSO_NUM_SAMPLE_BYTES)
    {
        uint8_t burst[LSM6DSO_NUM_SAMPLE_BYTES];
        encodeLsm6dsoOutputBurst(g_body_state, burst);
        return burst[reg - LSM6DSO_OUTPUT_START_REGISTER];
    }
    return 0;
}

}  // namespace sim
}  // namespace krabby

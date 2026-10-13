#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

/** Body-frame inertial state fed into the LSM6DSO output register burst (sensor frame = body with identity mount). */
struct SimImuBodyState
{
    float accelerationMps2[3] = {0.0f, 0.0f, 9.80665f};
    float angularRateRadS[3] = {0.0f, 0.0f, 0.0f};
    float temperatureCelsius = 25.0f;
};

void resetSimImuModel();
const SimImuBodyState &simImuBodyState();
void setSimImuBodyState(const SimImuBodyState &state);

/** Fill 14 bytes (temp, gyro xyz, accel xyz) little-endian for register window starting at 0x20. */
void encodeLsm6dsoOutputBurst(const SimImuBodyState &state, uint8_t outBytes[14]);

uint8_t lsm6dsoRegisterByte(uint8_t reg);

}  // namespace sim
}  // namespace krabby

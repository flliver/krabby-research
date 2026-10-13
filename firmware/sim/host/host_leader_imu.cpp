#include "host_leader_imu.hpp"

#include "sim_i2c.hpp"

#include "Wire.h"
#include "byte_order.h"
#include "src/imu/imu_constants.h"
#include "src/units/temperature_units.h"

namespace krabby {
namespace sim {
namespace host {

namespace {

HostLeaderImu g_leader_imu;

bool probeAddress(uint8_t address)
{
    Wire.beginTransmission(address);
    return Wire.endTransmission() == 0;
}

}  // namespace

void HostLeaderImu::setup()
{
    address_ = 0;
    Wire.begin();
    Wire.setClock(I2C_DEFAULT_BUS_CLOCK_HZ);
    Wire.setWireTimeout(I2C_BUS_TIMEOUT_MICROSECONDS, true);

    if (probeAddress(kI2cAddrLsm6dsoPrimary))
    {
        address_ = kI2cAddrLsm6dsoPrimary;
        return;
    }
    if (probeAddress(kI2cAddrLsm6dsoAlternate))
    {
        address_ = kI2cAddrLsm6dsoAlternate;
    }
}

ImuMeasurement HostLeaderImu::measure()
{
    if (address_ == 0)
    {
        return ImuMeasurement{false};
    }

    Wire.beginTransmission(address_);
    if (Wire.write(LSM6DSO_OUTPUT_START_REGISTER) != 1)
    {
        return ImuMeasurement{false};
    }
    if (Wire.endTransmission(false) != 0)
    {
        return ImuMeasurement{false};
    }

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
        {
            return ImuMeasurement{false};
        }
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

    return transformImuMeasurementToBodyFrame(measurement);
}

HostLeaderImu &leaderImu() { return g_leader_imu; }

}  // namespace host
}  // namespace sim
}  // namespace krabby

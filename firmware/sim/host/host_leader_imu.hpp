#pragma once

#include "src/imu/imu_measurement.h"

namespace krabby {
namespace sim {
namespace host {

class HostLeaderImu
{
public:
    void setup();

    ImuMeasurement measure();

    bool isOnline() const { return address_ != 0; }

private:
    uint8_t address_ = 0;
};

HostLeaderImu &leaderImu();

}  // namespace host
}  // namespace sim
}  // namespace krabby

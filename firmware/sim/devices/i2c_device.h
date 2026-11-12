#pragma once

#include <cstdint>
#include <vector>

namespace krabby {
namespace sim {

// One register-level peripheral, shared by scripted test buses and runtime Wire.
class I2cDevice
{
public:
    virtual ~I2cDevice() {}
    virtual uint8_t transmit(const std::vector<uint8_t> &bytes) = 0;
    virtual std::vector<uint8_t> receive(uint8_t count) = 0;
};

}  // namespace sim
}  // namespace krabby

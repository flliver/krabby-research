#pragma once

#include <stdint.h>
#include <string>
#include <stdexcept>
#include <vector>

#include "ina228.h"

namespace powerPollFake
{
// Register-level INA228 fakes answering the real driver: [0] pack, [1] midpoint.
extern ina228_native::Device devices[2];
// Driver calls observed per monitor, counted from register traffic.
struct Counters
{
    unsigned beginCount = 0;
    unsigned shuntCount = 0;
    unsigned resetCount = 0;
};
extern Counters counters[2];
// Milliseconds a monitor's bus-voltage read takes, charged to the fake clock.
extern uint32_t voltageReadDuration[2];
extern uint32_t now;
extern std::vector<std::string> events;
inline size_t index(uint8_t address)
{
    if (address == 0x40) return 0;
    if (address == 0x41) return 1;
    throw std::runtime_error("unexpected INA228 address: " + std::to_string(address));
}
inline void event(uint8_t address, const std::string &operation)
{
    events.push_back(std::string(index(address) == 0 ? "pack." : "mid.") + operation);
}
}

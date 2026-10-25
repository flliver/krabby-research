#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

class SimClock
{
public:
    void reset();
    void set(unsigned long ms);
    void advance(unsigned long deltaMs);
    unsigned long nowMs() const;

private:
    unsigned long now_ms_ = 0;
};

SimClock& simClock();

void resetHostSim();

}  // namespace sim
}  // namespace krabby

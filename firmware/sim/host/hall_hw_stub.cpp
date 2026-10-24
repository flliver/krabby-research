#include "hall_hw.h"

#include "hall_sim.hpp"

void hallHwInit() { krabby::sim::resetHallSim(); }

int32_t hallHwGetEdgeCount(uint8_t hallSlot)
{
    if (hallSlot >= 6) return 0;
    return static_cast<int32_t>(krabby::sim::hallSimEdgeCount(hallSlot));
}

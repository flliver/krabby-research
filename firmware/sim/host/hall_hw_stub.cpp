#include "hall_hw.h"

#include "hall_sim.hpp"
#include "pin_state.hpp"

// The counts belong to the board whose firmware is running (pinState().activeBoard()).
void hallHwInit() { krabby::sim::hallSimResetActiveBoard(); }

int32_t hallHwGetEdgeCount(uint8_t hallSlot) { return krabby::sim::hallSimEdgeCount(hallSlot); }

// Per-board flag, false unless a test turns it on (Rev 3 reports true on hardware).
bool hallHwHasQuadrature()
{
    return krabby::sim::hallSimQuadrature(krabby::sim::pinState().activeBoard());
}

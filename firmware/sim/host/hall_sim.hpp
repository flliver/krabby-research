#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

void resetHallSim();

// Accumulate hall edges from absolute joint-angle delta scaled by kHallEdgesPerFullTravel.
void hallSimSyncJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad);

uint32_t hallSimEdgeCount(uint8_t slot);

}  // namespace sim
}  // namespace krabby

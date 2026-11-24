#pragma once

#include <cstdint>

namespace krabby {
namespace sim {

// Hall counts are signed and kept per board (FRONT, LEFT, RIGHT = BoardId order).
// The slot functions without a board act on pinState().activeBoard(), which is
// the board whose firmware is running (see ScopedHostBoard).
constexpr int kHallSimBoardCount = 3;
constexpr int kHallSimSlotCount = 6;

// Clears every board's counts, baselines and quadrature flag, and resets the scale.
void resetHallSim();

// Zeroes the active board's counts and baselines (what hallHwInit() does on a
// board reset). Leaves the scale and the quadrature flag alone.
void hallSimResetActiveBoard();

void setHallEdgesPerFullTravel(uint32_t edges);

// Accumulate signed hall edges from the joint-angle delta scaled by the edges per
// full travel. A rising angle is extension and counts down by
// ACTUATOR_HALL_EXTEND_COUNT_SIGN, as +PWM does on the robot (the pot convention).
void hallSimSyncJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad);

// Set baseline angle without incrementing (firmware commanded stop; no shaft motion on hardware).
void hallSimRetargetJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad);

// Count of the active board.
int32_t hallSimEdgeCount(uint8_t slot);

// Per-board access. An out-of-range board or slot reads 0 and writes nothing.
int32_t hallSimEdgeCountBoard(int board, uint8_t slot);
void hallSimSetEdgeCount(int board, uint8_t slot, int32_t value);
// Adds d, saturating at the int32 range.
void hallSimAddEdges(int board, uint8_t slot, int32_t d);

// Quadrature (signed) counting for a board: what hallHwHasQuadrature() reports
// there. Defaults to false on every board.
void hallSimSetQuadrature(int board, bool on);
bool hallSimQuadrature(int board);

}  // namespace sim
}  // namespace krabby

#include "hall_sim.hpp"

#include "hall_encoder_scale.hpp"
#include "pin_state.hpp"
#include "src/actuator/actuator_constants.h"

#include <cmath>
#include <cstring>

namespace krabby {
namespace sim {

namespace {

float g_lastAngleRad[kHallSimBoardCount][kHallSimSlotCount];
float g_lowerRad[kHallSimBoardCount][kHallSimSlotCount];
float g_upperRad[kHallSimBoardCount][kHallSimSlotCount];
int32_t g_edgeCount[kHallSimBoardCount][kHallSimSlotCount];
bool g_hasLastAngle[kHallSimBoardCount][kHallSimSlotCount];
bool g_quadrature[kHallSimBoardCount];
uint32_t g_edgesPerFullTravel = kHallEdgesPerFullTravel;

bool validBoard(int board) { return board >= 0 && board < kHallSimBoardCount; }

bool validSlot(int board, uint8_t slot) { return validBoard(board) && slot < kHallSimSlotCount; }

int activeBoard() { return pinState().activeBoard(); }

int edgesForAbsDelta(float absDeltaRad, float spanRad)
{
    if (spanRad <= 0.0f || absDeltaRad <= 0.0f) return 0;
    const double scaled =
        static_cast<double>(absDeltaRad) / static_cast<double>(spanRad) *
        static_cast<double>(g_edgesPerFullTravel);
    // Floor so sub-detent physics jitter does not round up to a spurious edge.
    return static_cast<int>(std::floor(scaled));
}

void clearBoard(int board)
{
    std::memset(g_lastAngleRad[board], 0, sizeof(g_lastAngleRad[board]));
    std::memset(g_lowerRad[board], 0, sizeof(g_lowerRad[board]));
    std::memset(g_upperRad[board], 0, sizeof(g_upperRad[board]));
    std::memset(g_edgeCount[board], 0, sizeof(g_edgeCount[board]));
    std::memset(g_hasLastAngle[board], 0, sizeof(g_hasLastAngle[board]));
}

}  // namespace

void setHallEdgesPerFullTravel(uint32_t edges) { g_edgesPerFullTravel = edges; }

void resetHallSim()
{
    g_edgesPerFullTravel = kHallEdgesPerFullTravel;
    for (int board = 0; board < kHallSimBoardCount; ++board)
    {
        clearBoard(board);
        g_quadrature[board] = false;
    }
}

void hallSimResetActiveBoard()
{
    const int board = activeBoard();
    if (validBoard(board)) clearBoard(board);
}

void hallSimRetargetJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad)
{
    const int board = activeBoard();
    if (!validSlot(board, slot)) return;
    g_lowerRad[board][slot] = lowerRad;
    g_upperRad[board][slot] = upperRad;
    g_lastAngleRad[board][slot] = angleRad;
    g_hasLastAngle[board][slot] = true;
}

void hallSimSyncJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad)
{
    const int board = activeBoard();
    if (!validSlot(board, slot)) return;

    g_lowerRad[board][slot] = lowerRad;
    g_upperRad[board][slot] = upperRad;
    const float span = upperRad - lowerRad;
    if (!g_hasLastAngle[board][slot])
    {
        g_lastAngleRad[board][slot] = angleRad;
        g_hasLastAngle[board][slot] = true;
        return;
    }

    const float delta = angleRad - g_lastAngleRad[board][slot];
    g_lastAngleRad[board][slot] = angleRad;
    const int increment = edgesForAbsDelta(std::fabs(delta), span);
    if (increment > 0)
    {
        const int direction = delta > 0.0f ? ACTUATOR_HALL_EXTEND_COUNT_SIGN
                                           : -ACTUATOR_HALL_EXTEND_COUNT_SIGN;
        hallSimAddEdges(board, slot, direction * increment);
    }
}

int32_t hallSimEdgeCount(uint8_t slot) { return hallSimEdgeCountBoard(activeBoard(), slot); }

int32_t hallSimEdgeCountBoard(int board, uint8_t slot)
{
    if (!validSlot(board, slot)) return 0;
    return g_edgeCount[board][slot];
}

void hallSimSetEdgeCount(int board, uint8_t slot, int32_t value)
{
    if (!validSlot(board, slot)) return;
    g_edgeCount[board][slot] = value;
}

void hallSimAddEdges(int board, uint8_t slot, int32_t d)
{
    if (!validSlot(board, slot)) return;
    const int64_t sum = static_cast<int64_t>(g_edgeCount[board][slot]) + d;
    if (sum > INT32_MAX)
        g_edgeCount[board][slot] = INT32_MAX;
    else if (sum < INT32_MIN)
        g_edgeCount[board][slot] = INT32_MIN;
    else
        g_edgeCount[board][slot] = static_cast<int32_t>(sum);
}

void hallSimSetQuadrature(int board, bool on)
{
    if (validBoard(board)) g_quadrature[board] = on;
}

bool hallSimQuadrature(int board) { return validBoard(board) && g_quadrature[board]; }

}  // namespace sim
}  // namespace krabby

#include "hall_sim.hpp"

#include "hall_encoder_scale.hpp"

#include <cmath>
#include <cstring>

namespace krabby {
namespace sim {

namespace {

float g_lastAngleRad[6];
float g_lowerRad[6];
float g_upperRad[6];
uint32_t g_edgeCount[6];
bool g_hasLastAngle[6];
uint32_t g_edgesPerFullTravel = kHallEdgesPerFullTravel;

int edgesForAbsDelta(float absDeltaRad, float spanRad)
{
    if (spanRad <= 0.0f || absDeltaRad <= 0.0f) return 0;
    const double scaled =
        static_cast<double>(absDeltaRad) / static_cast<double>(spanRad) *
        static_cast<double>(g_edgesPerFullTravel);
    return static_cast<int>(lround(scaled));
}

}  // namespace

void setHallEdgesPerFullTravel(uint32_t edges) { g_edgesPerFullTravel = edges; }

void resetHallSim()
{
    g_edgesPerFullTravel = kHallEdgesPerFullTravel;
    std::memset(g_lastAngleRad, 0, sizeof(g_lastAngleRad));
    std::memset(g_lowerRad, 0, sizeof(g_lowerRad));
    std::memset(g_upperRad, 0, sizeof(g_upperRad));
    std::memset(g_edgeCount, 0, sizeof(g_edgeCount));
    std::memset(g_hasLastAngle, 0, sizeof(g_hasLastAngle));
}

void hallSimSyncJointAngle(uint8_t slot, float angleRad, float lowerRad, float upperRad)
{
    if (slot >= 6) return;

    g_lowerRad[slot] = lowerRad;
    g_upperRad[slot] = upperRad;
    const float span = upperRad - lowerRad;
    if (!g_hasLastAngle[slot])
    {
        g_lastAngleRad[slot] = angleRad;
        g_hasLastAngle[slot] = true;
        return;
    }

    const float delta = angleRad - g_lastAngleRad[slot];
    g_lastAngleRad[slot] = angleRad;
    const int increment = edgesForAbsDelta(std::fabs(delta), span);
    if (increment > 0)
    {
        g_edgeCount[slot] += static_cast<uint32_t>(increment);
    }
}

uint32_t hallSimEdgeCount(uint8_t slot)
{
    if (slot >= 6) return 0;
    return g_edgeCount[slot];
}

}  // namespace sim
}  // namespace krabby

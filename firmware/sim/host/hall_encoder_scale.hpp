#pragma once

#include <cstdint>

// Keep in sync with firmware/sim/hall_encoder.py DEFAULT_HALL_EDGES_PER_FULL_TRAVEL.
// Placeholder until stop-to-stop ``saf`` is measured on a rev-3 robot (fleet telemetry
// or mechanical sign-off).
static const uint32_t kHallEdgesPerFullTravel = 3200u;

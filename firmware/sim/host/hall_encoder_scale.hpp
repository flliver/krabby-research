#pragma once

#include <cstdint>

// Keep in sync with firmware/sim/hall_encoder.py DEFAULT_HALL_EDGES_PER_FULL_TRAVEL.
// Placeholder until stop-to-stop ``saf`` is measured on a rev-3 robot (fleet telemetry
// or mechanical sign-off). TASK-2 §3 / M21 OVERVIEW 2d.
static const uint32_t kHallEdgesPerFullTravel = 3200u;

#pragma once

#include <stdint.h>

#include "board.h"

/**
 * Periodic TX (monotonic now_ms).
 *
 * Joint snapshot: one LegTelemetry_<this leg> per tick (0x0C000100 + leg_index), packing
 * all three motors in that 64-byte frame — not three separate motor telemetries.
 * NodeStatus (0x0C000200 + leg_index) is a separate, lower-rate message (CAN_PROTOCOL.md).
 */
void telemetry_handler_tick(leg_t *leg, uint32_t now_ms);

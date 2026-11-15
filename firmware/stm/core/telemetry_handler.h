#pragma once

#include <stdint.h>

#include "board.h"

/**
 * Periodic TX (monotonic now_ms).
 *
 * TELEMETRY_<leg_id> (0x200 + leg_id) packs three joints; BOARD_INFO at 1 Hz.
 */
void telemetry_handler_tick(leg_t *leg, uint32_t now_ms);

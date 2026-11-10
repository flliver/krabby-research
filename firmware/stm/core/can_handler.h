#pragma once

#include <stdint.h>

#include "board.h"

/**
 * Decode one RX frame for this leg (main loop / sim step).
 *
 * Accept at most one LegCommand_<this leg> per call (0x0C000000 + leg_index, CAN-FD).
 * That single frame carries all three joints (hip_yaw, hip_pitch, knee); apply every
 * field in one pass — do not expect separate per-motor command IDs on this node.
 */
void can_handler_on_frame(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len);

#pragma once

#include <stdint.h>

#include "board.h"

/**
 * Decode one RX frame for this leg (main loop / sim step).
 *
 * SET_DUTY (0x100) carries all legs; apply this node's three joints from its slice.
 * CALIBRATE (0x180 + leg_id) starts on-board calibration for one joint.
 */
void can_handler_on_frame(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len);

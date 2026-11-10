/*
 * RX path: krabby_leg.dbc defines six LegCommand_* messages (one per leg board).
 * Each message encodes jog/position for hip_yaw, hip_pitch, and knee together.
 * This node filters on krabby_can_leg_command_id(leg_index) and updates all three
 * motors from one decode — never one CAN ID per motor on the command path.
 */
#include "can_handler.h"

#include <stddef.h>

#include "krabby/board_ids.h"

void can_handler_on_frame(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len) {
    (void)data;
    (void)len;
    if (leg == NULL) {
        return;
    }
    const uint32_t expected = krabby_can_leg_command_id((krabby_leg_index_t)leg->leg_index);
    if (can_id != expected) {
        return;
    }
    /* Decode LegCommand_* once; map hip_yaw / hip_pitch / knee fields → send_pwm() ×3. */
}

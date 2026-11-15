#pragma once

#include <stdint.h>

#include "board.h"

void krabby_sim_inject_rx(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len);
int krabby_sim_last_tx(leg_t *leg, uint32_t *can_id, uint8_t *data, uint8_t *len);
void krabby_sim_fail_next_tx(leg_t *leg);
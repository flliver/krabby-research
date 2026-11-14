#pragma once

#include "board.h"

void krabby_sim_set_pot(leg_t *leg, int motor, uint16_t raw);
void krabby_sim_set_hall(leg_t *leg, int motor, int32_t count);
float krabby_sim_get_duty(leg_t *leg, int motor);
int krabby_sim_last_tx(leg_t *leg, uint32_t *can_id, uint8_t *data, uint8_t *len);

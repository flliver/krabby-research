#pragma once

#include <stdint.h>

#include "board.h"
#include "board_pins.h"

void krabby_sim_set_tick_ms(int ms);
void krabby_sim_set_pot(leg_t *leg, int motor, uint16_t raw);
void krabby_sim_set_hall(leg_t *leg, int motor, int32_t count);
float krabby_sim_get_duty(leg_t *leg, int motor);

int board_pins_isaac_slot(const board_pins_t *pins);
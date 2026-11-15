#pragma once

#include "board.h"
#include "board_pins.h"

void app_init(leg_t *leg, board_pins_t *pins);
void app_tick(leg_t *leg, uint32_t now_ms);
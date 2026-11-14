#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board.h"

void can_init(leg_t *leg);
bool can_send(leg_t *leg, uint32_t id, const uint8_t *data, uint8_t len);
bool can_receive(leg_t *leg, uint32_t *id, uint8_t *data, uint8_t *len);
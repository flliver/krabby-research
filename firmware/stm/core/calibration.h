#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board.h"

void calibration_init(leg_t *leg);
void calibration_on_command(leg_t *leg, uint8_t joint_id);
void calibration_tick(leg_t *leg, uint32_t now_ms);

bool calibration_joint_calibrating(leg_t *leg, uint8_t joint_id);
bool calibration_joint_calibrated(leg_t *leg, uint8_t joint_id);

bool calibration_blocks_duty(leg_t *leg, int motor);
void calibration_abort(leg_t *leg);
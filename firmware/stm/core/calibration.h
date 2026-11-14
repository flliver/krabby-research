#pragma once

#include <stdint.h>

#include "board.h"

/** Begin on-board stall calibration for ``joint_id`` (2 or 3). Joint 1 is ignored. */
void calibration_on_command(leg_t *leg, uint8_t joint_id);

bool calibration_joint_calibrating(leg_t *leg, uint8_t joint_id);
bool calibration_joint_calibrated(leg_t *leg, uint8_t joint_id);
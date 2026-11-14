#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board_pins.h"

/**
 * Per-leg runtime state. Firmware uses one instance; the host sim creates six.
 * leg_index 0–5 maps to leg_id 1–6 (node switch order).
 */
typedef struct leg {
    uint8_t leg_index;
    uint8_t joint_count;
    board_pins_t *pins;
} leg_t;

/* motor / joint_index 0 = hip_yaw, 1 = hip_pitch, 2 = knee (wire joint_id = index + 1). */
void send_pwm(leg_t *leg, int motor, bool direction, int power_0_to_1000);
float read_applied_duty(leg_t *leg, int motor);
uint16_t read_pot(leg_t *leg, int motor);
int32_t read_hall(leg_t *leg, int motor);
bool read_fault(leg_t *leg, int motor);
void clear_fault(leg_t *leg, int motor);
bool save_joint_calibration(leg_t *leg, int motor, const joint_cal_t *cal);
bool load_joint_calibration(leg_t *leg, int motor, joint_cal_t *cal);
uint32_t leg_now_ms(leg_t *leg);
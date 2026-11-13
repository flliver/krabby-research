#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * Per-leg runtime state. Firmware uses one instance; the host sim creates six.
 * Leg index 0–5 maps to leg_id 1–6 (FL … RR on DBC suffixes).
 */
typedef struct leg {
    uint8_t leg_index;
    uint8_t joint_count; /* always 3: hip_yaw, hip_pitch, knee */
} leg_t;

/* Motor index 0 = hip_yaw, 1 = hip_pitch, 2 = knee. */
void send_pwm(leg_t *leg, int motor, bool direction, int power_0_to_1000);
uint16_t read_pot(leg_t *leg, int motor);
int32_t read_hall(leg_t *leg, int motor);
bool read_fault(leg_t *leg, int motor);
void clear_fault(leg_t *leg, int motor);

/* CAN-FD TX hook; returns false if the controller TX queue is full. */
bool can_send(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len);

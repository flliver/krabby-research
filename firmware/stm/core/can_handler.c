/*
 * RX path: SET_DUTY (0x100) and per-leg CALIBRATE (see krabby.dbc).
 * SET_DUTY carries all legs; this node reads its leg_id slice. CALIBRATE is per leg.
 */
#include "can_handler.h"

#include <stddef.h>
#include <stdint.h>

#include "calibration.h"
#include "krabby.h"
#include "krabby/board_ids.h"

static int16_t raw_duty_at(const struct krabby_set_duty_t *msg, uint8_t leg_index, uint8_t joint_index) {
    const int16_t *fields = &msg->l1_j1;
    return fields[(size_t)leg_index * 3u + (size_t)joint_index];
}

static void apply_duty(leg_t *leg, int motor, int16_t raw) {
    const double duty = (double)raw * 0.001;
    if (duty == 0.0 && read_fault(leg, motor)) {
        clear_fault(leg, motor);
    }
    const bool extend = duty >= 0.0;
    double mag = duty;
    if (mag < 0.0) {
        mag = -mag;
    }
    int power = (int)(mag * 1000.0 + 0.5);
    if (power > 1000) {
        power = 1000;
    }
    send_pwm(leg, motor, extend, power);
}

static void handle_set_duty(leg_t *leg, const uint8_t *data, uint8_t len) {
    struct krabby_set_duty_t msg;
    if (len < KRABBY_SET_DUTY_LENGTH) {
        return;
    }
    if (krabby_set_duty_unpack(&msg, data, (size_t)len) < 0) {
        return;
    }
    if (leg->leg_index >= KRABBY_LEG_COUNT) {
        return;
    }
    for (uint8_t joint_index = 0; joint_index < KRABBY_JOINTS_PER_LEG; joint_index++) {
        const int16_t raw = raw_duty_at(&msg, leg->leg_index, joint_index);
        apply_duty(leg, (int)joint_index, raw);
    }
}

static void handle_calibrate(leg_t *leg, const uint8_t *data, uint8_t len) {
    if (len < 1u) {
        return;
    }
    const uint8_t joint_id = data[0];
    calibration_on_command(leg, joint_id);
}

void can_handler_on_frame(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len) {
    if (leg == NULL || data == NULL) {
        return;
    }
    if (can_id == krabby_can_set_duty_id()) {
        handle_set_duty(leg, data, len);
        return;
    }
    const krabby_leg_index_t leg_index = (krabby_leg_index_t)leg->leg_index;
    if (can_id == krabby_can_calibrate_id_for_index(leg_index)) {
        handle_calibrate(leg, data, len);
    }
}
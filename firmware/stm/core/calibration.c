#include "calibration.h"

#include <stddef.h>
#include <string.h>

#include "board_pins.h"

#define KRABBY_CAL_STALL_TIMEOUT_MS 10000u

typedef enum {
    KRABBY_CAL_IDLE = 0,
    KRABBY_CAL_EXTEND,
    KRABBY_CAL_RETRACT,
    KRABBY_CAL_SAVE,
} cal_phase_t;

typedef struct {
    cal_phase_t phase;
    uint8_t joint_id;
    uint32_t phase_start_ms;
    joint_cal_t capture;
    bool calibrated[3];
} cal_state_t;

static cal_state_t s_cal[6];

static cal_state_t *state_for(leg_t *leg) {
    if (leg == NULL || leg->leg_index >= 6u) {
        return NULL;
    }
    return &s_cal[leg->leg_index];
}

void calibration_init(leg_t *leg) {
    cal_state_t *st = state_for(leg);
    if (st == NULL) {
        return;
    }
    memset(st, 0, sizeof(*st));
}

void calibration_on_command(leg_t *leg, uint8_t joint_id) {
    cal_state_t *st = state_for(leg);
    if (st == NULL || joint_id < 2u || joint_id > 3u) {
        return;
    }
    if (st->phase != KRABBY_CAL_IDLE) {
        return;
    }
    st->joint_id = joint_id;
    st->phase = KRABBY_CAL_EXTEND;
    st->phase_start_ms = leg_now_ms(leg);
    memset(&st->capture, 0, sizeof(st->capture));
}

static int motor_index(uint8_t joint_id) {
    return (int)joint_id - 1;
}

void calibration_tick(leg_t *leg, uint32_t now_ms) {
    cal_state_t *st = state_for(leg);
    if (st == NULL || st->phase == KRABBY_CAL_IDLE) {
        return;
    }
    const int motor = motor_index(st->joint_id);
    if (motor < 0) {
        st->phase = KRABBY_CAL_IDLE;
        return;
    }
    if ((now_ms - st->phase_start_ms) >= KRABBY_CAL_STALL_TIMEOUT_MS) {
        st->phase = KRABBY_CAL_IDLE;
        return;
    }
    if (read_fault(leg, motor)) {
        st->phase = KRABBY_CAL_IDLE;
        return;
    }
    switch (st->phase) {
        case KRABBY_CAL_EXTEND:
            send_pwm(leg, motor, true, 500);
            st->phase = KRABBY_CAL_RETRACT;
            break;
        case KRABBY_CAL_RETRACT:
            send_pwm(leg, motor, false, 500);
            st->phase = KRABBY_CAL_SAVE;
            break;
        case KRABBY_CAL_SAVE:
            send_pwm(leg, motor, true, 0);
            st->capture.pot_extended = read_pot(leg, motor);
            st->capture.hall_extended = read_hall(leg, motor);
            st->capture.pot_retracted = st->capture.pot_extended;
            st->capture.hall_retracted = st->capture.hall_extended;
            (void)save_joint_calibration(leg, motor, &st->capture);
            st->calibrated[motor] = true;
            st->phase = KRABBY_CAL_IDLE;
            break;
        default:
            st->phase = KRABBY_CAL_IDLE;
            break;
    }
}

bool calibration_joint_calibrating(leg_t *leg, uint8_t joint_id) {
    const cal_state_t *st = state_for(leg);
    if (st == NULL) {
        return false;
    }
    return st->phase != KRABBY_CAL_IDLE && st->joint_id == joint_id;
}

bool calibration_joint_calibrated(leg_t *leg, uint8_t joint_id) {
    const cal_state_t *st = state_for(leg);
    if (st == NULL || joint_id < 1u || joint_id > 3u) {
        return false;
    }
    return st->calibrated[joint_id - 1u];
}

bool calibration_blocks_duty(leg_t *leg, int motor) {
    const cal_state_t *st = state_for(leg);
    if (st == NULL || st->phase == KRABBY_CAL_IDLE) {
        return false;
    }
    return motor_index(st->joint_id) == motor;
}

void calibration_abort(leg_t *leg) {
    cal_state_t *st = state_for(leg);
    if (st != NULL) {
        st->phase = KRABBY_CAL_IDLE;
    }
}
#include "board.h"

#include <stddef.h>

static int wire_joint(int motor) {
    return motor + 1;
}

void send_pwm(leg_t *leg, int motor, bool direction, int power_0_to_1000) {
    if (leg == NULL || leg->pins == NULL || leg->pins->send_pwm == NULL) {
        return;
    }
    leg->pins->send_pwm(leg->pins, wire_joint(motor), direction, power_0_to_1000);
}

uint16_t read_pot(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || leg->pins->read_pot == NULL) {
        return 0;
    }
    return leg->pins->read_pot(leg->pins, wire_joint(motor));
}

int32_t read_hall(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || leg->pins->read_hall == NULL) {
        return 0;
    }
    return leg->pins->read_hall(leg->pins, wire_joint(motor));
}

bool read_fault(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || leg->pins->read_fault == NULL) {
        return false;
    }
    return leg->pins->read_fault(leg->pins, wire_joint(motor));
}

void clear_fault(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || leg->pins->clear_fault == NULL) {
        return;
    }
    leg->pins->clear_fault(leg->pins, wire_joint(motor));
}

bool save_joint_calibration(leg_t *leg, int motor, const joint_cal_t *cal) {
    if (leg == NULL || leg->pins == NULL || leg->pins->save_calibration == NULL || cal == NULL) {
        return false;
    }
    return leg->pins->save_calibration(leg->pins, wire_joint(motor), cal);
}

bool load_joint_calibration(leg_t *leg, int motor, joint_cal_t *cal) {
    if (leg == NULL || leg->pins == NULL || leg->pins->load_calibration == NULL || cal == NULL) {
        return false;
    }
    return leg->pins->load_calibration(leg->pins, wire_joint(motor), cal);
}

uint32_t leg_now_ms(leg_t *leg) {
    if (leg == NULL || leg->pins == NULL || leg->pins->now_ms == NULL) {
        return 0;
    }
    return leg->pins->now_ms(leg->pins);
}
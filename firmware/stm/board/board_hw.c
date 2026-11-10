#include "board.h"

void send_pwm(leg_t *leg, int motor, bool direction, int power_0_to_1000) {
    (void)leg;
    (void)motor;
    (void)direction;
    (void)power_0_to_1000;
}

uint16_t read_pot(leg_t *leg, int motor) {
    (void)leg;
    (void)motor;
    return 0;
}

int32_t read_hall(leg_t *leg, int motor) {
    (void)leg;
    (void)motor;
    return 0;
}

bool read_fault(leg_t *leg, int motor) {
    (void)leg;
    (void)motor;
    return false;
}

void clear_fault(leg_t *leg, int motor) {
    (void)leg;
    (void)motor;
}

bool can_send(leg_t *leg, uint32_t can_id, const uint8_t *data, uint8_t len) {
    (void)leg;
    (void)can_id;
    (void)data;
    (void)len;
    return false;
}

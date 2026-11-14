#include "board.h"

#include <string.h>

#define KRABBY_SIM_LEGS 6
#define KRABBY_SIM_MOTORS 3

static float s_duty[KRABBY_SIM_LEGS][KRABBY_SIM_MOTORS];
static uint16_t s_pot[KRABBY_SIM_LEGS][KRABBY_SIM_MOTORS];
static int32_t s_hall[KRABBY_SIM_LEGS][KRABBY_SIM_MOTORS];

static uint32_t s_last_tx_id[KRABBY_SIM_LEGS];
static uint8_t s_last_tx_data[KRABBY_SIM_LEGS][64];
static uint8_t s_last_tx_len[KRABBY_SIM_LEGS];

static int leg_slot(const leg_t *leg) {
    if (leg == NULL || leg->leg_index >= KRABBY_SIM_LEGS) {
        return -1;
    }
    return (int)leg->leg_index;
}

void send_pwm(leg_t *leg, int motor, bool direction, int power_0_to_1000) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return;
    }
    float signed_duty = (float)power_0_to_1000 / 1000.0f;
    if (!direction) {
        signed_duty = -signed_duty;
    }
    s_duty[li][motor] = signed_duty;
}

uint16_t read_pot(leg_t *leg, int motor) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return 0;
    }
    return s_pot[li][motor];
}

int32_t read_hall(leg_t *leg, int motor) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return 0;
    }
    return s_hall[li][motor];
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
    const int li = leg_slot(leg);
    if (li < 0 || data == NULL || len > sizeof(s_last_tx_data[0])) {
        return false;
    }
    s_last_tx_id[li] = can_id;
    s_last_tx_len[li] = len;
    memcpy(s_last_tx_data[li], data, len);
    return true;
}

int krabby_sim_last_tx(leg_t *leg, uint32_t *can_id, uint8_t *data, uint8_t *len) {
    const int li = leg_slot(leg);
    if (li < 0 || can_id == NULL || data == NULL || len == NULL) {
        return -1;
    }
    *can_id = s_last_tx_id[li];
    *len = s_last_tx_len[li];
    memcpy(data, s_last_tx_data[li], *len);
    return 0;
}

/* Test / sim bridge hooks for Isaac and unit tests. */
void krabby_sim_set_pot(leg_t *leg, int motor, uint16_t raw) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return;
    }
    s_pot[li][motor] = raw;
}

void krabby_sim_set_hall(leg_t *leg, int motor, int32_t count) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return;
    }
    s_hall[li][motor] = count;
}

float read_applied_duty(leg_t *leg, int motor) {
    const int li = leg_slot(leg);
    if (li < 0 || motor < 0 || motor >= KRABBY_SIM_MOTORS) {
        return 0.0f;
    }
    return s_duty[li][motor];
}

float krabby_sim_get_duty(leg_t *leg, int motor) {
    return read_applied_duty(leg, motor);
}

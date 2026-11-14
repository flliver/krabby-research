#include "board_pins.h"

#include "board.h"

#include <stddef.h>

typedef struct {
    board_pins_t base;
    float applied_duty[3];
} stm32_pins_t;

static stm32_pins_t s_stm32_pins;

static void stm32_init(board_pins_t *self) {
    (void)self;
    /* TIM1 PWM, DAC7574, encoders — wire to HAL from hardware/PCB-v0.3/Firmware/. */
}

static uint8_t stm32_node_id(board_pins_t *self) {
    (void)self;
    /* Board_GetNodeId() from reference firmware. */
    return 1u;
}

static uint32_t stm32_now_ms(board_pins_t *self) {
    (void)self;
    /* HAL_GetTick(). */
    return 0u;
}

static void stm32_send_pwm(board_pins_t *self, int joint, bool direction, int power) {
    stm32_pins_t *pins = (stm32_pins_t *)self;
    if (joint < 1 || joint > 3) {
        return;
    }
    float signed_duty = (float)power / 1000.0f;
    if (!direction) {
        signed_duty = -signed_duty;
    }
    pins->applied_duty[joint - 1] = signed_duty;
    (void)joint;
    (void)direction;
    (void)power;
}

static uint16_t stm32_read_pot(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return 0;
}

static int32_t stm32_read_hall(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return 0;
}

static bool stm32_read_fault(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
    return false;
}

static void stm32_clear_fault(board_pins_t *self, int joint) {
    (void)self;
    (void)joint;
}

static bool stm32_save_cal(board_pins_t *self, int joint, const joint_cal_t *cal) {
    (void)self;
    (void)joint;
    (void)cal;
    return false;
}

static bool stm32_load_cal(board_pins_t *self, int joint, joint_cal_t *cal) {
    (void)self;
    (void)joint;
    (void)cal;
    return false;
}

board_pins_t *board_pins_stm32(void) {
    stm32_pins_t *pins = &s_stm32_pins;
    pins->base.init = stm32_init;
    pins->base.node_id = stm32_node_id;
    pins->base.now_ms = stm32_now_ms;
    pins->base.send_pwm = stm32_send_pwm;
    pins->base.read_pot = stm32_read_pot;
    pins->base.read_hall = stm32_read_hall;
    pins->base.read_fault = stm32_read_fault;
    pins->base.clear_fault = stm32_clear_fault;
    pins->base.save_calibration = stm32_save_cal;
    pins->base.load_calibration = stm32_load_cal;
    return &pins->base;
}

float read_applied_duty(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor > 2) {
        return 0.0f;
    }
    const stm32_pins_t *pins = (const stm32_pins_t *)leg->pins;
    return pins->applied_duty[motor];
}
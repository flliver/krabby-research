#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint16_t pot_retracted;
    uint16_t pot_extended;
    int32_t hall_retracted;
    int32_t hall_extended;
} joint_cal_t;

typedef struct board_pins board_pins_t;

struct board_pins {
    void (*init)(board_pins_t *self);
    uint8_t (*node_id)(board_pins_t *self);
    uint32_t (*now_ms)(board_pins_t *self);
    void (*send_pwm)(board_pins_t *self, int joint, bool direction, int power);
    uint16_t (*read_pot)(board_pins_t *self, int joint);
    int32_t (*read_hall)(board_pins_t *self, int joint);
    bool (*read_fault)(board_pins_t *self, int joint);
    void (*clear_fault)(board_pins_t *self, int joint);
    bool (*save_calibration)(board_pins_t *self, int joint, const joint_cal_t *cal);
    bool (*load_calibration)(board_pins_t *self, int joint, joint_cal_t *cal);
};

/* board/board_pins_stm32.c — MCU build */
board_pins_t *board_pins_stm32(void);

/* sim/board_pins_isaac.c — Isaac sim (host-stm) */
board_pins_t *board_pins_isaac_create(uint8_t node_id);
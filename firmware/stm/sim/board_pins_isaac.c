#include "board_pins_isaac.h"

#include <stddef.h>
#include <string.h>

#define KRABBY_ISAACS_LEGS 6
#define KRABBY_ISAACS_JOINTS 3

typedef struct {
    board_pins_t base;
    uint8_t node_id;
    float duty[KRABBY_ISAACS_JOINTS];
    uint16_t pot[KRABBY_ISAACS_JOINTS];
    int32_t hall[KRABBY_ISAACS_JOINTS];
    bool fault[KRABBY_ISAACS_JOINTS];
    joint_cal_t cal[KRABBY_ISAACS_JOINTS];
    bool cal_valid[KRABBY_ISAACS_JOINTS];
} isaac_pins_t;

static isaac_pins_t s_pins[KRABBY_ISAACS_LEGS];
static int s_sim_tick_ms;

int board_pins_isaac_slot(const board_pins_t *pins) {
    if (pins == NULL) {
        return -1;
    }
    const isaac_pins_t *isaac = (const isaac_pins_t *)pins;
    if (isaac->node_id < 1u || isaac->node_id > 6u) {
        return -1;
    }
    return (int)isaac->node_id - 1;
}

static void isaac_init(board_pins_t *self) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    memset(isaac->duty, 0, sizeof(isaac->duty));
    memset(isaac->fault, 0, sizeof(isaac->fault));
}

static uint8_t isaac_node_id(board_pins_t *self) {
    return ((isaac_pins_t *)self)->node_id;
}

static uint32_t isaac_now_ms(board_pins_t *self) {
    (void)self;
    return (uint32_t)s_sim_tick_ms;
}

static void isaac_send_pwm(board_pins_t *self, int joint, bool direction, int power) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS) {
        return;
    }
    const int mi = joint - 1;
    float signed_duty = (float)power / 1000.0f;
    if (!direction) {
        signed_duty = -signed_duty;
    }
    isaac->duty[mi] = signed_duty;
}

static uint16_t isaac_read_pot(board_pins_t *self, int joint) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS) {
        return 0;
    }
    return isaac->pot[joint - 1];
}

static int32_t isaac_read_hall(board_pins_t *self, int joint) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS) {
        return 0;
    }
    return isaac->hall[joint - 1];
}

static bool isaac_read_fault(board_pins_t *self, int joint) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS) {
        return false;
    }
    return isaac->fault[joint - 1];
}

static void isaac_clear_fault(board_pins_t *self, int joint) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS) {
        return;
    }
    isaac->fault[joint - 1] = false;
}

static bool isaac_save_cal(board_pins_t *self, int joint, const joint_cal_t *cal) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS || cal == NULL) {
        return false;
    }
    isaac->cal[joint - 1] = *cal;
    isaac->cal_valid[joint - 1] = true;
    return true;
}

static bool isaac_load_cal(board_pins_t *self, int joint, joint_cal_t *cal) {
    isaac_pins_t *isaac = (isaac_pins_t *)self;
    if (joint < 1 || joint > KRABBY_ISAACS_JOINTS || cal == NULL) {
        return false;
    }
    if (!isaac->cal_valid[joint - 1]) {
        return false;
    }
    *cal = isaac->cal[joint - 1];
    return true;
}

static void bind_isaac_vtable(isaac_pins_t *isaac) {
    isaac->base.init = isaac_init;
    isaac->base.node_id = isaac_node_id;
    isaac->base.now_ms = isaac_now_ms;
    isaac->base.send_pwm = isaac_send_pwm;
    isaac->base.read_pot = isaac_read_pot;
    isaac->base.read_hall = isaac_read_hall;
    isaac->base.read_fault = isaac_read_fault;
    isaac->base.clear_fault = isaac_clear_fault;
    isaac->base.save_calibration = isaac_save_cal;
    isaac->base.load_calibration = isaac_load_cal;
}

board_pins_t *board_pins_isaac_create(uint8_t node_id) {
    if (node_id < 1u || node_id > 6u) {
        return NULL;
    }
    isaac_pins_t *isaac = &s_pins[node_id - 1u];
    memset(isaac, 0, sizeof(*isaac));
    isaac->node_id = node_id;
    bind_isaac_vtable(isaac);
    return &isaac->base;
}

void krabby_sim_set_tick_ms(int ms) {
    s_sim_tick_ms = ms;
}

float read_applied_duty(leg_t *leg, int motor) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor >= KRABBY_ISAACS_JOINTS) {
        return 0.0f;
    }
    const int li = board_pins_isaac_slot(leg->pins);
    if (li < 0) {
        return 0.0f;
    }
    return s_pins[li].duty[motor];
}

void krabby_sim_set_pot(leg_t *leg, int motor, uint16_t raw) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor >= KRABBY_ISAACS_JOINTS) {
        return;
    }
    const int li = board_pins_isaac_slot(leg->pins);
    if (li < 0) {
        return;
    }
    s_pins[li].pot[motor] = raw;
}

void krabby_sim_set_hall(leg_t *leg, int motor, int32_t count) {
    if (leg == NULL || leg->pins == NULL || motor < 0 || motor >= KRABBY_ISAACS_JOINTS) {
        return;
    }
    const int li = board_pins_isaac_slot(leg->pins);
    if (li < 0) {
        return;
    }
    s_pins[li].hall[motor] = count;
}

float krabby_sim_get_duty(leg_t *leg, int motor) {
    return read_applied_duty(leg, motor);
}
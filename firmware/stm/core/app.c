#include "app.h"

#include <stddef.h>

#include "board_can.h"
#include "calibration.h"
#include "can_handler.h"
#include "telemetry_handler.h"

void app_init(leg_t *leg, board_pins_t *pins) {
    if (leg == NULL) {
        return;
    }
    leg->pins = pins;
    leg->joint_count = 3;
    if (pins != NULL) {
        if (pins->init != NULL) {
            pins->init(pins);
        }
        if (pins->node_id != NULL) {
            const uint8_t node = pins->node_id(pins);
            if (node >= 1u && node <= 6u) {
                leg->leg_index = (uint8_t)(node - 1u);
            }
        }
    }
    can_init(leg);
    calibration_init(leg);
    for (int motor = 0; motor < 3; motor++) {
        joint_cal_t cal;
        if (load_joint_calibration(leg, motor, &cal)) {
            (void)cal;
        }
    }
}

void app_tick(leg_t *leg, uint32_t now_ms) {
    if (leg == NULL) {
        return;
    }
    uint32_t can_id = 0;
    uint8_t data[64];
    uint8_t len = 0;
    while (can_receive(leg, &can_id, data, &len)) {
        can_handler_on_frame(leg, can_id, data, len);
    }
    calibration_tick(leg, now_ms);
    telemetry_handler_tick(leg, now_ms);
}
#include "calibration.h"

#include <stddef.h>

void calibration_on_command(leg_t *leg, uint8_t joint_id) {
    if (leg == NULL) {
        return;
    }
    if (joint_id < 2u || joint_id > 3u) {
        return;
    }
    (void)joint_id;
    /* Stall capture and NV save land here. */
}

bool calibration_joint_calibrating(leg_t *leg, uint8_t joint_id) {
    (void)leg;
    (void)joint_id;
    return false;
}

bool calibration_joint_calibrated(leg_t *leg, uint8_t joint_id) {
    (void)leg;
    (void)joint_id;
    return false;
}
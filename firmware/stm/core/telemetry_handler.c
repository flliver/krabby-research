/*
 * TX: TELEMETRY_<leg_id> (0x200 + leg_id) at 50 ms; BOARD_INFO_<leg_id> at 1 s.
 */
#include "telemetry_handler.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "calibration.h"
#include "krabby.h"
#include "krabby/board_ids.h"
#include "version.h"

#define KRABBY_TELEMETRY_PERIOD_MS 50u
#define KRABBY_BOARD_INFO_PERIOD_MS 1000u

typedef struct {
    uint32_t last_telemetry_ms;
    uint32_t last_board_info_ms;
} telemetry_schedule_t;

static telemetry_schedule_t s_schedule[KRABBY_LEG_COUNT];

static bool telemetry_due(uint32_t now_ms, uint32_t *last_ms) {
    if (*last_ms == 0u) {
        *last_ms = now_ms;
        return true;
    }
    if ((now_ms - *last_ms) >= KRABBY_TELEMETRY_PERIOD_MS) {
        *last_ms = now_ms;
        return true;
    }
    return false;
}

static bool board_info_due(uint32_t now_ms, uint32_t *last_ms) {
    if (*last_ms == 0u) {
        if (now_ms < KRABBY_BOARD_INFO_PERIOD_MS) {
            return false;
        }
        *last_ms = now_ms;
        return true;
    }
    if ((now_ms - *last_ms) >= KRABBY_BOARD_INFO_PERIOD_MS) {
        *last_ms = now_ms;
        return true;
    }
    return false;
}

static double joint_position(leg_t *leg, int motor) {
    if (motor == 0) {
        return (double)NAN;
    }
    (void)leg;
    return 0.0;
}

static void fill_joint(
    leg_t *leg,
    int motor,
    uint8_t joint_id,
    int32_t *hall,
    int16_t *duty,
    uint16_t *pot,
    uint8_t *fault,
    uint8_t *calibrating,
    uint8_t *calibrated,
    uint64_t *position
) {
    const double applied = (double)read_applied_duty(leg, motor);
    *hall = read_hall(leg, motor);
    *duty = krabby_telemetry_1_j1_duty_encode(applied);
    *pot = read_pot(leg, motor);
    *fault = read_fault(leg, motor) ? 1u : 0u;
    *calibrating = calibration_joint_calibrating(leg, (uint8_t)joint_id) ? 1u : 0u;
    *calibrated = calibration_joint_calibrated(leg, (uint8_t)joint_id) ? 1u : 0u;
    {
        const double pos = joint_position(leg, motor);
        uint64_t raw = 0u;
        (void)memcpy(&raw, &pos, sizeof(raw));
        *position = raw;
    }
}

static int pack_telemetry(uint8_t leg_index, const struct krabby_telemetry_1_t *msg, uint8_t *dst, size_t size) {
    switch (leg_index) {
        case 0:
            return krabby_telemetry_1_pack(dst, msg, size);
        case 1:
            return krabby_telemetry_2_pack(dst, (const struct krabby_telemetry_2_t *)msg, size);
        case 2:
            return krabby_telemetry_3_pack(dst, (const struct krabby_telemetry_3_t *)msg, size);
        case 3:
            return krabby_telemetry_4_pack(dst, (const struct krabby_telemetry_4_t *)msg, size);
        case 4:
            return krabby_telemetry_5_pack(dst, (const struct krabby_telemetry_5_t *)msg, size);
        case 5:
            return krabby_telemetry_6_pack(dst, (const struct krabby_telemetry_6_t *)msg, size);
        default:
            return -1;
    }
}

static int pack_board_info(uint8_t leg_index, const struct krabby_board_info_1_t *msg, uint8_t *dst, size_t size) {
    switch (leg_index) {
        case 0:
            return krabby_board_info_1_pack(dst, msg, size);
        case 1:
            return krabby_board_info_2_pack(dst, (const struct krabby_board_info_2_t *)msg, size);
        case 2:
            return krabby_board_info_3_pack(dst, (const struct krabby_board_info_3_t *)msg, size);
        case 3:
            return krabby_board_info_4_pack(dst, (const struct krabby_board_info_4_t *)msg, size);
        case 4:
            return krabby_board_info_5_pack(dst, (const struct krabby_board_info_5_t *)msg, size);
        case 5:
            return krabby_board_info_6_pack(dst, (const struct krabby_board_info_6_t *)msg, size);
        default:
            return -1;
    }
}

static void send_telemetry(leg_t *leg) {
    struct krabby_telemetry_1_t msg;
    memset(&msg, 0, sizeof(msg));
    fill_joint(leg, 0, 1, &msg.j1_hall, &msg.j1_duty, &msg.j1_pot, &msg.j1_fault, &msg.j1_calibrating, &msg.j1_calibrated,
               &msg.j1_position);
    fill_joint(leg, 1, 2, &msg.j2_hall, &msg.j2_duty, &msg.j2_pot, &msg.j2_fault, &msg.j2_calibrating, &msg.j2_calibrated,
               &msg.j2_position);
    fill_joint(leg, 2, 3, &msg.j3_hall, &msg.j3_duty, &msg.j3_pot, &msg.j3_fault, &msg.j3_calibrating, &msg.j3_calibrated,
               &msg.j3_position);

    uint8_t buf[KRABBY_TELEMETRY_1_LENGTH];
    const int packed = pack_telemetry(leg->leg_index, &msg, buf, sizeof(buf));
    if (packed < 0) {
        return;
    }
    const krabby_leg_index_t leg_index = (krabby_leg_index_t)leg->leg_index;
    if (!can_send(leg, krabby_can_telemetry_id_for_index(leg_index), buf, (uint8_t)packed)) {
        return;
    }
}

static void send_board_info(leg_t *leg) {
    struct krabby_board_info_1_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.major = KRABBY_LEG_FW_VERSION_MAJOR;
    msg.minor = KRABBY_LEG_FW_VERSION_MINOR;
    msg.patch = KRABBY_LEG_FW_VERSION_PATCH;
    msg.commit = KRABBY_LEG_FW_GIT_SHA_UINT32;

    uint8_t buf[KRABBY_BOARD_INFO_1_LENGTH];
    const int packed = pack_board_info(leg->leg_index, &msg, buf, sizeof(buf));
    if (packed < 0) {
        return;
    }
    const krabby_leg_index_t leg_index = (krabby_leg_index_t)leg->leg_index;
    if (!can_send(leg, krabby_can_board_info_id_for_index(leg_index), buf, (uint8_t)packed)) {
        return;
    }
}

void telemetry_handler_tick(leg_t *leg, uint32_t now_ms) {
    if (leg == NULL || leg->leg_index >= KRABBY_LEG_COUNT) {
        return;
    }
    telemetry_schedule_t *sched = &s_schedule[leg->leg_index];
    if (telemetry_due(now_ms, &sched->last_telemetry_ms)) {
        send_telemetry(leg);
    }
    if (board_info_due(now_ms, &sched->last_board_info_ms)) {
        send_board_info(leg);
    }
}
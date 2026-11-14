#pragma once

#include <stdint.h>

/** Six legs on the robot (leg_index 0–5, node_id / leg_id 1–6). */
#define KRABBY_LEG_COUNT 6u
#define KRABBY_JOINTS_PER_LEG 3u
#define KRABBY_MCU_JOINT_COUNT (KRABBY_LEG_COUNT * KRABBY_JOINTS_PER_LEG)

/** NV leg byte when unassigned (factory default). */
#define KRABBY_LEG_CODE_UNASSIGNED 255u

/** Host discovery index for unassigned nodes (not on the leg CAN bus). */
#define KRABBY_LEG_INDEX_UNASSIGNED 6u

/** leg_index 0–5; node switch leg_id = index + 1. */
typedef enum {
    KRABBY_LEG_FL = 0,
    KRABBY_LEG_FR = 1,
    KRABBY_LEG_ML = 2,
    KRABBY_LEG_MR = 3,
    KRABBY_LEG_RL = 4,
    KRABBY_LEG_RR = 5,
} krabby_leg_index_t;

/** joint_id 1=Y, 2=H, 3=K → array index 0–2. */
typedef enum {
    KRABBY_JOINT_HIP_YAW = 0,
    KRABBY_JOINT_HIP_PITCH = 1,
    KRABBY_JOINT_KNEE = 2,
} krabby_joint_index_t;

/** krabby.dbc (11-bit standard IDs). */
#define KRABBY_CAN_ID_SET_DUTY 0x100u
#define KRABBY_CAN_ID_CALIBRATE_BASE 0x180u
#define KRABBY_CAN_ID_TELEMETRY_BASE 0x200u
#define KRABBY_CAN_ID_BOARD_INFO_BASE 0x300u

static inline uint8_t krabby_leg_id_from_index(krabby_leg_index_t leg) {
    return (uint8_t)((unsigned)leg + 1u);
}

static inline uint32_t krabby_can_set_duty_id(void) {
    return KRABBY_CAN_ID_SET_DUTY;
}

/** @param leg_id host/board identity 1–6 */
static inline uint32_t krabby_can_calibrate_id(uint8_t leg_id) {
    return KRABBY_CAN_ID_CALIBRATE_BASE + (uint32_t)leg_id;
}

static inline uint32_t krabby_can_calibrate_id_for_index(krabby_leg_index_t leg) {
    return krabby_can_calibrate_id(krabby_leg_id_from_index(leg));
}

static inline uint32_t krabby_can_telemetry_id(uint8_t leg_id) {
    return KRABBY_CAN_ID_TELEMETRY_BASE + (uint32_t)leg_id;
}

static inline uint32_t krabby_can_telemetry_id_for_index(krabby_leg_index_t leg) {
    return krabby_can_telemetry_id(krabby_leg_id_from_index(leg));
}

static inline uint32_t krabby_can_board_info_id(uint8_t leg_id) {
    return KRABBY_CAN_ID_BOARD_INFO_BASE + (uint32_t)leg_id;
}

static inline uint32_t krabby_can_board_info_id_for_index(krabby_leg_index_t leg) {
    return krabby_can_board_info_id(krabby_leg_id_from_index(leg));
}

/** can0 = FL, ML, RL; can1 = FR, MR, RR (by leg_index). */
static inline int krabby_leg_on_can0(krabby_leg_index_t leg) {
    return leg == KRABBY_LEG_FL || leg == KRABBY_LEG_ML || leg == KRABBY_LEG_RL;
}

static inline int krabby_leg_code_valid(uint8_t leg_code) {
    return leg_code > 0u && leg_code <= KRABBY_LEG_COUNT;
}

static inline int krabby_leg_index_valid(uint8_t leg_index) {
    return leg_index < KRABBY_LEG_COUNT;
}
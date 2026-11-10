#pragma once

#include <stdint.h>

/** Six provisioned legs on the robot (PROVISIONING.md, CAN_PROTOCOL.md). */
#define KRABBY_LEG_COUNT 6u
#define KRABBY_JOINTS_PER_LEG 3u
#define KRABBY_MCU_JOINT_COUNT (KRABBY_LEG_COUNT * KRABBY_JOINTS_PER_LEG)

/** NV leg byte when no leg has been assigned (factory / full chip erase). */
#define KRABBY_LEG_CODE_UNASSIGNED 255u

/** Host discovery index for unassigned nodes (no LegCommand / LegTelemetry band). */
#define KRABBY_LEG_INDEX_UNASSIGNED 6u

/** Leg index 0–5 and matching NV leg_code byte (identical numbering). */
typedef enum {
    KRABBY_LEG_FL = 0,
    KRABBY_LEG_FR = 1,
    KRABBY_LEG_ML = 2,
    KRABBY_LEG_MR = 3,
    KRABBY_LEG_RL = 4,
    KRABBY_LEG_RR = 5,
} krabby_leg_index_t;

/** Per-board joint order (LEG_BOARD_CONTROL.md). */
typedef enum {
    KRABBY_JOINT_HIP_YAW = 0,
    KRABBY_JOINT_HIP_PITCH = 1,
    KRABBY_JOINT_KNEE = 2,
} krabby_joint_index_t;

/** 29-bit extended CAN identifiers (arbitration field, before SocketCAN EFF flag). */
#define KRABBY_CAN_ID_LEG_COMMAND_BASE 0x0C000000u
#define KRABBY_CAN_ID_LEG_TELEMETRY_BASE 0x0C000100u
#define KRABBY_CAN_ID_NODE_STATUS_BASE 0x0C000200u
#define KRABBY_CAN_ID_DISCOVERY_REQUEST 0x0C0007F0u
#define KRABBY_CAN_ID_PROVISION_SET 0x0C0007E0u

static inline uint32_t krabby_can_leg_command_id(krabby_leg_index_t leg) {
    return KRABBY_CAN_ID_LEG_COMMAND_BASE + (uint32_t)leg;
}

static inline uint32_t krabby_can_leg_telemetry_id(krabby_leg_index_t leg) {
    return KRABBY_CAN_ID_LEG_TELEMETRY_BASE + (uint32_t)leg;
}

static inline uint32_t krabby_can_node_status_id(krabby_leg_index_t leg) {
    return KRABBY_CAN_ID_NODE_STATUS_BASE + (uint32_t)leg;
}

/** Robot SocketCAN interface for a provisioned leg index (can0 = FL/ML/RL, can1 = FR/MR/RR). */
static inline int krabby_leg_on_can0(krabby_leg_index_t leg) {
    return leg == KRABBY_LEG_FL || leg == KRABBY_LEG_ML || leg == KRABBY_LEG_RL;
}

static inline int krabby_leg_code_valid(uint8_t leg_code) {
    return leg_code < KRABBY_LEG_COUNT;
}

static inline int krabby_leg_index_valid(uint8_t leg_index) {
    return leg_index < KRABBY_LEG_COUNT;
}

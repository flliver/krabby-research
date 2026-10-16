#pragma once

#include <stddef.h>

#include "src/controller/board_role.h"

// Joint names per board role. Physical pins are shared; only names change after election.
static constexpr size_t ACTUATOR_ROLE_NAME_COUNT = 6;

static const char *const ACTUATOR_NAMES_FRONT[ACTUATOR_ROLE_NAME_COUNT] = {
    "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"};
static const char *const ACTUATOR_NAMES_LEFT[ACTUATOR_ROLE_NAME_COUNT] = {
    "RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL"};
static const char *const ACTUATOR_NAMES_RIGHT[ACTUATOR_ROLE_NAME_COUNT] = {
    "RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL"};

inline const char *const *actuatorNamesForRole(BoardRole role)
{
    if (role == ROLE_LEFT)
        return ACTUATOR_NAMES_LEFT;
    if (role == ROLE_RIGHT)
        return ACTUATOR_NAMES_RIGHT;
    return ACTUATOR_NAMES_FRONT; // FRONT and UNKNOWN
}

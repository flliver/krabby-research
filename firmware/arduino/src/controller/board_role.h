#pragma once

#include <string.h>

#include "../actuator/actuator_identity.h"

enum BoardRole
{
    // These values are persisted in EEPROM; keep their numeric encoding stable.
    ROLE_UNKNOWN = 0,
    ROLE_FRONT = 1,
    ROLE_LEFT = 2,
    ROLE_RIGHT = 3,
    BOARD_ROLE_COUNT = 4,
};

static_assert(ROLE_UNKNOWN == 0, "Unknown EEPROM role changed");
static_assert(ROLE_FRONT == 1, "Front EEPROM role changed");
static_assert(ROLE_LEFT == 2, "Left EEPROM role changed");
static_assert(ROLE_RIGHT == 3, "Right EEPROM role changed");

static constexpr BoardRole ALL_BOARD_ROLES[] = {
    ROLE_FRONT,
    ROLE_LEFT,
    ROLE_RIGHT,
};

inline const char *boardRoleLabel(BoardRole role)
{
    switch (role)
    {
        case ROLE_FRONT: return "FRONT";
        case ROLE_LEFT: return "LEFT";
        case ROLE_RIGHT: return "RIGHT";
        default: return "UNKWN";
    }
}

inline BoardRole getBoardRoleForActuator(ActuatorId actuatorId)
{
    switch (actuatorId)
    {
        case ActuatorId::FLHY:
        case ActuatorId::FLHL:
        case ActuatorId::FLKL:
        case ActuatorId::FRHY:
        case ActuatorId::FRHL:
        case ActuatorId::FRKL:
            return ROLE_FRONT;
        case ActuatorId::MLHY:
        case ActuatorId::MLHL:
        case ActuatorId::MLKL:
        case ActuatorId::RLHY:
        case ActuatorId::RLHL:
        case ActuatorId::RLKL:
            return ROLE_LEFT;
        case ActuatorId::MRHY:
        case ActuatorId::MRHL:
        case ActuatorId::MRKL:
        case ActuatorId::RRHY:
        case ActuatorId::RRHL:
        case ActuatorId::RRKL:
            return ROLE_RIGHT;
        default:
            return ROLE_UNKNOWN;
    }
}

// Role names for the SET/GET role config path. Distinct from boardRoleLabel(),
// whose fixed-width "UNKWN" is the telemetry wire label.
inline const char *roleConfigName(BoardRole role)
{
    switch (role)
    {
        case ROLE_FRONT: return "FRONT";
        case ROLE_LEFT: return "LEFT";
        case ROLE_RIGHT: return "RIGHT";
        default: return "UNKNOWN";
    }
}

inline bool parseRole(const char *name, BoardRole &role)
{
    static constexpr BoardRole CONFIG_ROLES[] = {ROLE_UNKNOWN, ROLE_FRONT, ROLE_LEFT, ROLE_RIGHT};
    for (BoardRole candidate : CONFIG_ROLES)
    {
        if (strcmp(name, roleConfigName(candidate)) == 0)
        {
            role = candidate;
            return true;
        }
    }
    return false;
}

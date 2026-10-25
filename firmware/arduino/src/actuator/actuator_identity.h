#pragma once

#include <stdint.h>

enum ActuatorId : uint8_t
{
    FLHY,
    FLHL,
    FLKL,
    FRHY,
    FRHL,
    FRKL,
    MLHY,
    MLHL,
    MLKL,
    MRHY,
    MRHL,
    MRKL,
    RLHY,
    RLHL,
    RLKL,
    RRHY,
    RRHL,
    RRKL,
    ActuatorCount,
};

inline ActuatorId &operator++(ActuatorId &actuatorId)
{
    actuatorId = static_cast<ActuatorId>(actuatorId + 1);
    return actuatorId;
}

inline ActuatorId parseActuatorId(const char *name)
{
    if (!name ||
        name[0] == '\0' || name[1] == '\0' ||
        name[2] == '\0' || name[3] == '\0')
    {
        return ActuatorId::ActuatorCount;
    }

    ActuatorId firstLegActuator;
    if (name[0] == 'F' && name[1] == 'L')
        firstLegActuator = ActuatorId::FLHY;
    else if (name[0] == 'F' && name[1] == 'R')
        firstLegActuator = ActuatorId::FRHY;
    else if (name[0] == 'M' && name[1] == 'L')
        firstLegActuator = ActuatorId::MLHY;
    else if (name[0] == 'M' && name[1] == 'R')
        firstLegActuator = ActuatorId::MRHY;
    else if (name[0] == 'R' && name[1] == 'L')
        firstLegActuator = ActuatorId::RLHY;
    else if (name[0] == 'R' && name[1] == 'R')
        firstLegActuator = ActuatorId::RRHY;
    else
        return ActuatorId::ActuatorCount;

    uint8_t jointOffset;
    if (name[2] == 'H' && name[3] == 'Y')
        jointOffset = 0;
    else if (name[2] == 'H' && name[3] == 'L')
        jointOffset = 1;
    else if (name[2] == 'K' && name[3] == 'L')
        jointOffset = 2;
    else
        return ActuatorId::ActuatorCount;

    return static_cast<ActuatorId>(firstLegActuator + jointOffset);
}

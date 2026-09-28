#pragma once

#include <stdint.h>

#include "actuator_current_tracker.h"
#include "actuator_identity.h"

enum class ActuatorConnection : uint8_t
{
    Unknown = 0,
    Connected = 1,
    Disconnected = 2,
};

inline ActuatorConnection determineActuatorConnectionState(
    bool isPositionValid,
    ActuatorCurrentEvidence currentEvidence)
{
    if (!isPositionValid ||
        currentEvidence == ActuatorCurrentEvidence::CurrentAbsent)
    {
        return ActuatorConnection::Disconnected;
    }
    if (currentEvidence == ActuatorCurrentEvidence::Unknown)
        return ActuatorConnection::Unknown;
    return ActuatorConnection::Connected;
}

struct ActuatorStatus
{
    ActuatorId actuatorId = ActuatorId::ActuatorCount;
    ActuatorConnection connectionState = ActuatorConnection::Unknown;
    int16_t commandedPwm = 0;
};

#include "telemetry.h"

#include "actuator/actuator_constants.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool actuatorMatchesExpectedRole(
    BoardRole expectedRole,
    ActuatorId actuatorId)
{
    if (expectedRole == ROLE_UNKNOWN)
        expectedRole = ROLE_FRONT;
    return getBoardRoleForActuator(actuatorId) == expectedRole;
}

const char *boardTelemetryRoleLabel(BoardRole role)
{
    return role == ROLE_LEFT ? "LEFT " : boardRoleLabel(role);
}

static bool parseActuatorPwmField(
    const char *begin,
    const char *end,
    uint8_t &value)
{
    if (!begin || begin == end)
        return false;

    char *parsedEnd = nullptr;
    const long parsed = strtol(begin, &parsedEnd, 10);
    if (parsedEnd != end ||
        parsed < 0 ||
        parsed > ACTUATOR_PWM_MAXIMUM_MAGNITUDE)
    {
        return false;
    }

    value = static_cast<uint8_t>(parsed);
    return true;
}

static bool parseActuatorPositionValidity(
    const char *begin,
    const char *end,
    bool &isPositionValid)
{
    if (!begin || begin == end)
        return false;

    char *parsedEnd = nullptr;
    const double position = strtod(begin, &parsedEnd);
    if (parsedEnd != end)
        return false;

    isPositionValid = isfinite(position);
    return true;
}

static bool parseActuatorConnectionState(
    const char *begin,
    const char *end,
    ActuatorConnection &state)
{
    if (!begin || begin + 1 != end || *begin < '0' || *begin > '2')
        return false;
    state = static_cast<ActuatorConnection>(*begin - '0');
    return true;
}

static bool parseActuatorStatusFields(
    const char *begin,
    const char *end,
    ActuatorStatus &status)
{
    const char *fieldBegin[ACTUATOR_TELEMETRY_MAX_FIELD_COUNT];
    const char *fieldEnd[ACTUATOR_TELEMETRY_MAX_FIELD_COUNT];
    size_t fieldCount = 0;
    const char *cursor = begin;
    while (cursor < end)
    {
        while (cursor < end && (*cursor == ' ' || *cursor == '\t'))
            ++cursor;
        if (cursor == end)
            break;
        if (fieldCount == ACTUATOR_TELEMETRY_MAX_FIELD_COUNT)
            return false;
        fieldBegin[fieldCount] = cursor;
        while (cursor < end && *cursor != ' ' && *cursor != '\t')
            ++cursor;
        fieldEnd[fieldCount++] = cursor;
    }

    if (fieldCount != ACTUATOR_TELEMETRY_FIELD_COUNT &&
        fieldCount != ACTUATOR_TELEMETRY_MAX_FIELD_COUNT)
        return false;

    const char *actuatorName =
        fieldBegin[ACTUATOR_TELEMETRY_NAME_FIELD_INDEX];
    if (fieldEnd[ACTUATOR_TELEMETRY_NAME_FIELD_INDEX] - actuatorName != 4)
        return false;
    status.actuatorId = parseActuatorId(actuatorName);
    if (status.actuatorId == ActuatorId::ActuatorCount)
        return false;

    bool isPositionValid = false;
    if (!parseActuatorPositionValidity(
            fieldBegin[ACTUATOR_TELEMETRY_POSITION_FIELD_INDEX],
            fieldEnd[ACTUATOR_TELEMETRY_POSITION_FIELD_INDEX],
            isPositionValid))
    {
        return false;
    }

    // Nine-field senders encode disconnection only through position.
    status.connectionState = isPositionValid
        ? ActuatorConnection::Unknown
        : ActuatorConnection::Disconnected;
    if (fieldCount == ACTUATOR_TELEMETRY_MAX_FIELD_COUNT)
    {
        if (!parseActuatorConnectionState(
                fieldBegin[ACTUATOR_TELEMETRY_CONNECTION_STATE_FIELD_INDEX],
                fieldEnd[ACTUATOR_TELEMETRY_CONNECTION_STATE_FIELD_INDEX],
                status.connectionState))
        {
            return false;
        }

        // Preserve the legacy non-finite disconnection encoding.
        if (!isPositionValid)
            status.connectionState = ActuatorConnection::Disconnected;
    }

    uint8_t retractPwm = 0;
    uint8_t extendPwm = 0;
    if (!parseActuatorPwmField(
            fieldBegin[ACTUATOR_TELEMETRY_RETRACT_PWM_FIELD_INDEX],
            fieldEnd[ACTUATOR_TELEMETRY_RETRACT_PWM_FIELD_INDEX],
            retractPwm) ||
        !parseActuatorPwmField(
            fieldBegin[ACTUATOR_TELEMETRY_EXTEND_PWM_FIELD_INDEX],
            fieldEnd[ACTUATOR_TELEMETRY_EXTEND_PWM_FIELD_INDEX],
            extendPwm))
    {
        return false;
    }

    if (retractPwm != 0 && extendPwm != 0)
        return false;
    status.commandedPwm =
        static_cast<int16_t>(extendPwm) - static_cast<int16_t>(retractPwm);
    return true;
}

bool parseActuatorStatus(
    const char *line,
    BoardRole expectedRole,
    ActuatorStatus (&status)[CONTROLLER_ACTUATOR_COUNT])
{
    if (!line)
        return false;

    const char *expectedRoleLabel = boardTelemetryRoleLabel(expectedRole);
    const size_t roleLength = strlen(expectedRoleLabel);
    if (strncmp(line, expectedRoleLabel, roleLength) != 0 ||
        line[roleLength] != ';')
        return false;

    ActuatorStatus parsedStatus[CONTROLLER_ACTUATOR_COUNT] = {};
    bool hasActuator[ActuatorId::ActuatorCount] = {};
    const char *cursor = line + roleLength;
    for (size_t actuator = 0;
         actuator < CONTROLLER_ACTUATOR_COUNT;
         ++actuator)
    {
        if (*cursor != ';')
            return false;
        const char *begin = cursor + 1;
        const char *end = strchr(begin, ';');
        if (!end)
            end = begin + strlen(begin);
        if (!parseActuatorStatusFields(
                begin, end, parsedStatus[actuator]))
        {
            return false;
        }
        const ActuatorId actuatorId = parsedStatus[actuator].actuatorId;
        if (!actuatorMatchesExpectedRole(expectedRole, actuatorId) ||
            hasActuator[actuatorId])
        {
            return false;
        }
        hasActuator[actuatorId] = true;
        cursor = end;
    }

    if (*cursor != '\0')
        return false;

    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        if (actuatorMatchesExpectedRole(expectedRole, actuatorId) &&
            !hasActuator[actuatorId])
            return false;

    for (size_t actuator = 0;
         actuator < CONTROLLER_ACTUATOR_COUNT;
         ++actuator)
        status[actuator] = parsedStatus[actuator];
    return true;
}

// Turns simulated hardware state into OLED draw calls using the production
// state-to-model path and renderer.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/display/display_frame_model.h"
#include "src/display/display_renderer.h"
#include "trace_canvas.h"

namespace
{

static constexpr int MOVE_THRESHOLD = 20;
static constexpr uint32_t NOW_MILLISECONDS = 1000;

struct SimulatedState
{
    SimulatedState()
        : role(ROLE_FRONT), rollDegrees(0.0f), pitchDegrees(0.0f),
          isImuValid(true), batteryVolts{13.3f, 13.3f},
          isFrontPresent(true), isLeftPresent(true), isRightPresent(true), actuators{}
    {
        for (ActuatorId actuatorId = ActuatorId::FLHY;
             actuatorId < ActuatorId::ActuatorCount;
             ++actuatorId)
            actuators[actuatorId] = ActuatorGlyph::Hold;
    }

    BoardRole role;
    float rollDegrees;
    float pitchDegrees;
    bool isImuValid;
    float batteryVolts[2];
    bool isFrontPresent;
    bool isLeftPresent;
    bool isRightPresent;
    ActuatorGlyph actuators[ActuatorId::ActuatorCount];
};

BoardRole parseRole(const char *name)
{
    if (strcmp(name, "FRONT") == 0) return ROLE_FRONT;
    if (strcmp(name, "LEFT") == 0) return ROLE_LEFT;
    if (strcmp(name, "RIGHT") == 0) return ROLE_RIGHT;
    return ROLE_UNKNOWN;
}

bool parseGlyph(const char *name, ActuatorGlyph &out)
{
    if (strcmp(name, "hold") == 0) out = ActuatorGlyph::Hold;
    else if (strcmp(name, "extend") == 0) out = ActuatorGlyph::Extend;
    else if (strcmp(name, "retract") == 0) out = ActuatorGlyph::Retract;
    else if (strcmp(name, "disc") == 0) out = ActuatorGlyph::Disconnected;
    else if (strcmp(name, "unverified") == 0) out = ActuatorGlyph::Unverified;
    else return false;
    return true;
}

bool parseActuators(char *value, SimulatedState &state)
{
    char *cursor = value;
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        char *name = cursor;
        while (*cursor != '\0' && *cursor != ',' && *cursor != ';')
            ++cursor;
        const char separator = *cursor;
        if (separator != '\0')
            *cursor++ = '\0';
        if (!parseGlyph(name, state.actuators[actuatorId]))
            return false;

        const char expected = (actuatorId + 1) % 3 != 0
            ? ',' : (actuatorId + 1 != ActuatorId::ActuatorCount ? ';' : '\0');
        if (separator != expected)
            return false;
    }
    return *cursor == '\0';
}

bool parsePair(char *value, float (&out)[2])
{
    char *separator = strchr(value, ',');
    if (separator == NULL)
        return false;
    *separator = '\0';
    out[0] = static_cast<float>(atof(value));
    out[1] = static_cast<float>(atof(separator + 1));
    return true;
}

bool applyField(SimulatedState &state, const char *key, char *value)
{
    if (strcmp(key, "role") == 0) state.role = parseRole(value);
    else if (strcmp(key, "roll") == 0) state.rollDegrees = static_cast<float>(atof(value));
    else if (strcmp(key, "pitch") == 0) state.pitchDegrees = static_cast<float>(atof(value));
    else if (strcmp(key, "imu") == 0) state.isImuValid = atoi(value) != 0;
    else if (strcmp(key, "battery") == 0) return parsePair(value, state.batteryVolts);
    else if (strcmp(key, "front") == 0)
        state.isFrontPresent = atoi(value) != 0;
    else if (strcmp(key, "left") == 0)
        state.isLeftPresent = atoi(value) != 0;
    else if (strcmp(key, "right") == 0)
        state.isRightPresent = atoi(value) != 0;
    else if (strcmp(key, "legs") == 0) return parseActuators(value, state);
    else return false;
    return true;
}

ImuMeasurement measurementForTilt(const SimulatedState &state)
{
    ImuMeasurement measurement(state.isImuValid);
    if (!state.isImuValid)
        return measurement;

    const float roll = state.rollDegrees * static_cast<float>(M_PI) / 180.0f;
    const float pitch = state.pitchDegrees * static_cast<float>(M_PI) / 180.0f;
    const float yz = METERS_PER_SECOND_SQUARED_PER_G * cos(pitch);
    measurement.acceleration[0] = MetersPerSecondSquared(
        -METERS_PER_SECOND_SQUARED_PER_G * sin(pitch));
    measurement.acceleration[1] = MetersPerSecondSquared(yz * sin(roll));
    measurement.acceleration[2] = MetersPerSecondSquared(yz * cos(roll));
    return measurement;
}

DisplayFrame buildFrame(const SimulatedState &state)
{
    ControllerFreshnessTracker controllerFreshnessTrackers[BOARD_ROLE_COUNT];
    if (state.isFrontPresent)
        controllerFreshnessTrackers[ROLE_FRONT] =
            ControllerFreshnessTracker::seenAt(NOW_MILLISECONDS);
    if (state.isLeftPresent)
        controllerFreshnessTrackers[ROLE_LEFT] =
            ControllerFreshnessTracker::seenAt(NOW_MILLISECONDS);
    if (state.isRightPresent)
        controllerFreshnessTrackers[ROLE_RIGHT] =
            ControllerFreshnessTracker::seenAt(NOW_MILLISECONDS);

    ActuatorStatus actuators[ActuatorId::ActuatorCount];
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        ActuatorStatus &actuator = actuators[actuatorId];
        actuator.actuatorId = actuatorId;
        const ActuatorGlyph glyph = state.actuators[actuatorId];
        actuator.connectionState = glyph == ActuatorGlyph::Disconnected
            ? ActuatorConnection::Disconnected
            : glyph == ActuatorGlyph::Unverified
                ? ActuatorConnection::Unknown
                : ActuatorConnection::Connected;
        actuator.commandedPwm = glyph == ActuatorGlyph::Extend
            ? MOVE_THRESHOLD
            : glyph == ActuatorGlyph::Retract ? -MOVE_THRESHOLD : 0;
    }

    DisplayFrame frame = buildDisplayFrame(
        state.role,
        controllerFreshnessTrackers,
        actuators,
        measurementForTilt(state),
        NOW_MILLISECONDS,
        MOVE_THRESHOLD);
    const Volts batteryVoltage[2] = {
        Volts(state.batteryVolts[0]), Volts(state.batteryVolts[1])};
    setBatteryVoltages(frame, batteryVoltage);
    return frame;
}

}  // namespace

int main()
{
    TraceCanvas canvas(stdout);
    DisplayRenderer<TraceCanvas> renderer(canvas);
    SimulatedState state;
    bool hasPendingFrame = false;

    char lineBuffer[512];
    while (true)
    {
        const bool hasMoreInput =
            fgets(lineBuffer, sizeof(lineBuffer), stdin) != NULL;
        char *line = lineBuffer;
        if (hasMoreInput)
        {
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] == '#')
                continue;
        }

        if (!hasMoreInput || line[0] == '\0')
        {
            if (hasPendingFrame)
            {
                printf("frame\n");
                renderer.render(buildFrame(state));
                hasPendingFrame = false;
            }
            if (!hasMoreInput)
                break;
            continue;
        }

        char *separator = strchr(line, '=');
        if (separator == NULL)
        {
            fprintf(stderr, "oled_trace: expected key=value, got '%s'\n", line);
            return 1;
        }
        *separator = '\0';
        if (!applyField(state, line, separator + 1))
        {
            fprintf(stderr, "oled_trace: bad field '%s'\n", line);
            return 1;
        }
        hasPendingFrame = true;
    }
    return 0;
}

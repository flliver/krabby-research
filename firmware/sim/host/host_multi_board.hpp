#pragma once

class LinearActuator;

namespace krabby {
namespace sim {
namespace host {

void setThreeBoardFirmwareLoopEnabled(bool enabled);
bool threeBoardFirmwareLoopEnabled();

void setSingleLeaderEighteenJointEnabled(bool enabled);
bool singleLeaderEighteenJointEnabled();

void threeBoardFirmwareSetup();
void threeBoardFirmwareLoopOnce();
void singleLeaderEighteenFirmwareLoopOnce();
void threeBoardFirmwareResetLoopState();
/** True while the multi-board leader's robot calibration or any board's pass runs. */
bool threeBoardCalibrationActive();

/**
 * Most recent SENS state code ('?', 'P', 'e', 'x', 's', 'E') printed for a joint by
 * any board, or '\0' when none has been seen. Cleared by threeBoardFirmwareResetLoopState.
 */
char threeBoardSensorState(const char *name);

int threeBoardActuatorCommandPwm(const char *name);
LinearActuator *findThreeBoardActuator(const char *name);

/** Test hook: stop emitting follower uplink telemetry (simulates a missing board). */
void setThreeBoardFollowerTelemetryEnabled(bool leftEnabled, bool rightEnabled);

}  // namespace host
}  // namespace sim
}  // namespace krabby

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

int threeBoardActuatorCommandPwm(const char *name);
LinearActuator *findThreeBoardActuator(const char *name);

/** Test hook: stop emitting follower uplink telemetry (simulates a missing board). */
void setThreeBoardFollowerTelemetryEnabled(bool leftEnabled, bool rightEnabled);

}  // namespace host
}  // namespace sim
}  // namespace krabby

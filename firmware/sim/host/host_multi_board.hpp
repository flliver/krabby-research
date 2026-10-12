#pragma once

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

}  // namespace host
}  // namespace sim
}  // namespace krabby

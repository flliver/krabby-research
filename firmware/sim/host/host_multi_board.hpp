#pragma once

namespace krabby {
namespace sim {
namespace host {

void setThreeBoardFirmwareLoopEnabled(bool enabled);
bool threeBoardFirmwareLoopEnabled();

void threeBoardFirmwareSetup();
void threeBoardFirmwareLoopOnce();
void threeBoardFirmwareResetLoopState();

}  // namespace host
}  // namespace sim
}  // namespace krabby

#pragma once

#include "src/controller/board_role.h"
#include "src/display/display_frame_model.h"

namespace krabby {
namespace sim {
namespace host {

void hostDisplaySetup();
void hostDisplayReset();
void hostDisplayInvalidate();
void hostDisplayStep(unsigned long nowMs);

/** Leader OLED: ingest a forwarded follower telemetry line (same as ``arduino.ino``). */
void hostDisplayNoteTelemetryLine(const char *line, BoardRole boardRole);

uint32_t hostOledBufferCrc32();
ActuatorGlyph hostOledGlyphForActuator(ActuatorId actuatorId);
/** Last rendered health screen: is controller third shown as live (freshness timeout)? */
bool hostOledControllerShownFresh(BoardRole role, unsigned long nowMs);

}  // namespace host
}  // namespace sim
}  // namespace krabby

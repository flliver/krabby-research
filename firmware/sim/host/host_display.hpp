#pragma once

#include "src/display/display_frame_model.h"

namespace krabby {
namespace sim {
namespace host {

void hostDisplaySetup();
void hostDisplayReset();
void hostDisplayInvalidate();
void hostDisplayStep(unsigned long nowMs);

uint32_t hostOledBufferCrc32();
ActuatorGlyph hostOledGlyphForActuator(ActuatorId actuatorId);

}  // namespace host
}  // namespace sim
}  // namespace krabby

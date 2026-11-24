#pragma once

// Host-sim copy of arduino.ino's C command tokenizing (the replies and the
// dispatch are shared: src/actuator/calibration_command_dispatch.h).

#include "Arduino.h"
#include "actuator_manager.h"
#include "command.h"
#include "src/cli/actuator_calibration_command.h"

namespace krabby {
namespace sim {
namespace host {

// Tokenizes the payload after 'C' the way arduino.ino's handleCalibrationCommand does.
inline ActuatorCalibrationCommandKind parseHostCalibrationLine(
    const String &line,
    ActuatorCalibrationRequest &request)
{
    constexpr size_t kMaxTokens = 5;
    int index = 0;
    const int length = static_cast<int>(line.length());
    String storage[kMaxTokens];
    const char *tokens[kMaxTokens];
    size_t tokenCount = 0;
    while (tokenCount < kMaxTokens)
    {
        storage[tokenCount] = nextTok(line, index, length);
        if (storage[tokenCount].length() == 0) break;
        tokens[tokenCount] = storage[tokenCount].c_str();
        ++tokenCount;
    }
    return parseActuatorCalibrationCommand(tokenCount, tokens, request);
}

}  // namespace host
}  // namespace sim
}  // namespace krabby

#pragma once

#include "actuator_manager.h"

namespace krabby {
namespace sim {
namespace host {

ActuatorManager &actuatorManager();
void ensureActuatorsInitialized();
LinearActuator *find_actuator(const char *actuator_name);

}  // namespace host
}  // namespace sim
}  // namespace krabby

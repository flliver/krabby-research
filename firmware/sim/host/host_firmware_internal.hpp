#pragma once

#include "actuator_manager.h"

namespace krabby {
namespace sim {
namespace host {

ActuatorManager &actuatorManager();
void ensureActuatorsInitialized();

}  // namespace host
}  // namespace sim
}  // namespace krabby

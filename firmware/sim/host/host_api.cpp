#include "host_api.h"

#include "actuator_manager.h"
#include "pin_state.hpp"
#include "sim_clock.hpp"

namespace {

LinearActuator s_flhy("FLHY", 2, 3, 22, 6, 0, 0);
LinearActuator s_flhl("FLHL", 4, 5, 24, 7, 1, 1);
LinearActuator s_flkl("FLKL", 6, 7, 26, 8, 2, 2);
LinearActuator s_frhy("FRHY", 8, 9, 23, 9, 3, 3);
LinearActuator s_frhl("FRHL", 10, 11, 25, 10, 4, 4);
LinearActuator s_frkl("FRKL", 12, 13, 27, 11, 5, 5);

LinearActuator *s_actuators[] = {
    &s_flhy, &s_flhl, &s_flkl, &s_frhy, &s_frhl, &s_frkl,
};

ActuatorManager s_manager(s_actuators, sizeof(s_actuators) / sizeof(s_actuators[0]));
bool s_initialized = false;

void ensure_initialized()
{
    if (s_initialized) return;
    s_manager.initAll();
    s_initialized = true;
}

}  // namespace

extern "C" void krabby_host_reset(void)
{
    krabby::sim::resetHostSim();
    s_initialized = false;
}

extern "C" void krabby_host_handle_jog(const char *actuator_name, int pwm)
{
    if (actuator_name == nullptr) return;
    ensure_initialized();
    s_manager.handleJog(String(actuator_name), pwm);
}

extern "C" int krabby_host_pin_pwm(int pin)
{
    return krabby::sim::pinState().pwmDuty(pin);
}

extern "C" int krabby_host_pin_digital(int pin)
{
    return krabby::sim::pinState().digitalRead(pin);
}

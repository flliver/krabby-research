#include "host_api.h"

#include "EEPROM.h"
#include "actuator_manager.h"
#include "hall_sim.hpp"
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

LinearActuator *find_actuator(const char *actuator_name)
{
    if (actuator_name == nullptr) return nullptr;
    for (LinearActuator *actuator : s_actuators)
    {
        if (String(actuator->getName()) == actuator_name) return actuator;
    }
    return nullptr;
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

extern "C" void krabby_host_set_target(const char *actuator_name, float position)
{
    if (actuator_name == nullptr) return;
    ensure_initialized();
    for (LinearActuator *actuator : s_actuators)
    {
        if (String(actuator->getName()) == actuator_name)
        {
            actuator->setTarget(position);
            return;
        }
    }
}

extern "C" void krabby_host_set_actuator_stops(
    const char *actuator_name,
    int min_stop,
    int max_stop)
{
    if (actuator_name == nullptr) return;
    for (LinearActuator *actuator : s_actuators)
    {
        if (String(actuator->getName()) == actuator_name)
        {
            actuator->minStop = min_stop;
            actuator->maxStop = max_stop;
            return;
        }
    }
}

extern "C" void krabby_host_update_all(void)
{
    ensure_initialized();
    s_manager.updateAll();
}

extern "C" int krabby_host_pin_pwm(int pin)
{
    return krabby::sim::pinState().pwmDuty(pin);
}

extern "C" int krabby_host_pin_digital(int pin)
{
    return krabby::sim::pinState().digitalRead(pin);
}

extern "C" void krabby_host_set_analog_read(int pin, int value)
{
    krabby::sim::pinState().setAnalogInput(pin, value);
}

extern "C" int krabby_host_analog_read(int pin)
{
    return krabby::sim::pinState().analogInput(pin);
}

extern "C" void krabby_host_advance_millis(unsigned long delta_ms)
{
    krabby::sim::simClock().advance(delta_ms);
}

extern "C" unsigned long krabby_host_millis(void)
{
    return krabby::sim::simClock().nowMs();
}

extern "C" void krabby_host_set_hall_edges_per_full_travel(uint32_t edges)
{
    krabby::sim::setHallEdgesPerFullTravel(edges);
}

extern "C" void krabby_host_sync_hall_joint(
    uint8_t hall_slot,
    float angle_rad,
    float lower_rad,
    float upper_rad)
{
    krabby::sim::hallSimSyncJointAngle(hall_slot, angle_rad, lower_rad, upper_rad);
}

extern "C" int32_t krabby_host_hall_edge_count(uint8_t hall_slot)
{
    return static_cast<int32_t>(krabby::sim::hallSimEdgeCount(hall_slot));
}

extern "C" uint8_t krabby_host_eeprom_read(int addr)
{
    return EEPROM.read(addr);
}

extern "C" void krabby_host_eeprom_update(int addr, uint8_t value)
{
    EEPROM.update(addr, value);
}

extern "C" void krabby_host_save_calibration(void)
{
    ensure_initialized();
    s_manager.saveCalibration();
}

extern "C" void krabby_host_load_calibration(void)
{
    ensure_initialized();
    s_manager.loadCalibration();
}

extern "C" int krabby_host_actuator_min_stop(const char *actuator_name)
{
    LinearActuator *actuator = find_actuator(actuator_name);
    return actuator != nullptr ? actuator->minStop : -1;
}

extern "C" int krabby_host_actuator_max_stop(const char *actuator_name)
{
    LinearActuator *actuator = find_actuator(actuator_name);
    return actuator != nullptr ? actuator->maxStop : -1;
}

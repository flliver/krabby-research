#include "host_api.h"

#include <cstring>

#include "EEPROM.h"
#include "actuator_manager.h"
#include "hall_sim.hpp"
#include "pin_state.hpp"
#include "sim_clock.hpp"
#include "host_board_role.hpp"
#include "host_multi_board.hpp"
#include "sim_serial.hpp"

namespace krabby {
namespace sim {
namespace host {

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

void ensureActuatorsInitialized()
{
    if (s_initialized) return;
    s_manager.initAll();
    s_initialized = true;
}

ActuatorManager &actuatorManager()
{
    ensureActuatorsInitialized();
    return s_manager;
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

}  // namespace host
}  // namespace sim
}  // namespace krabby

using krabby::sim::host::actuatorManager;
using krabby::sim::host::ensureActuatorsInitialized;
using krabby::sim::host::find_actuator;
using krabby::sim::host::s_actuators;
using krabby::sim::host::s_initialized;
using krabby::sim::host::s_manager;

extern "C" void krabby_host_reset(void)
{
    krabby::sim::resetHostSim();
    krabby_host_firmware_reset_loop_state();
    krabby::sim::host::setThreeBoardFirmwareLoopEnabled(false);
    s_initialized = false;
    s_manager.calState = ActuatorManager::CAL_IDLE;
    s_manager.stateTimer = 0;
}

extern "C" void krabby_host_handle_jog(const char *actuator_name, int pwm)
{
    if (actuator_name == nullptr) return;
    ensureActuatorsInitialized();
    s_manager.handleJog(String(actuator_name), pwm);
}

extern "C" void krabby_host_set_target(const char *actuator_name, float position)
{
    if (actuator_name == nullptr) return;
    ensureActuatorsInitialized();
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
    ensureActuatorsInitialized();
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
    ensureActuatorsInitialized();
    s_manager.saveCalibration();
}

extern "C" void krabby_host_load_calibration(void)
{
    ensureActuatorsInitialized();
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

extern "C" void krabby_host_serial_usb_rx_write(const uint8_t *data, int length)
{
    if (data == nullptr || length <= 0) return;
    krabby::sim::serialPort(krabby::sim::SerialPortId::kUsb)
        .rxPush(data, static_cast<std::size_t>(length));
}

extern "C" int krabby_host_serial_usb_rx_available(void)
{
    return krabby::sim::serialPort(krabby::sim::SerialPortId::kUsb).rxAvailable();
}

extern "C" int krabby_host_serial_usb_rx_read(void)
{
    return krabby::sim::serialPort(krabby::sim::SerialPortId::kUsb).rxRead();
}

extern "C" int krabby_host_serial_usb_tx_available(void)
{
    return krabby::sim::serialPort(krabby::sim::SerialPortId::kUsb).txBytesAvailable();
}

extern "C" int krabby_host_serial_usb_tx_read(void)
{
    return krabby::sim::serialPort(krabby::sim::SerialPortId::kUsb).popTxByte();
}

extern "C" void krabby_host_serial_usb_println(const char *text)
{
    if (text == nullptr) return;
    Serial.println(text);
}

extern "C" void krabby_host_run_three_board_role_election(int roles_out[3])
{
    if (roles_out == nullptr) return;
    BoardRole roles[3] = {
        ROLE_UNKNOWN,
        ROLE_UNKNOWN,
        ROLE_UNKNOWN,
    };
    krabby::sim::runThreeBoardRoleElectionCooperative(roles);
    roles_out[0] = static_cast<int>(roles[0]);
    roles_out[1] = static_cast<int>(roles[1]);
    roles_out[2] = static_cast<int>(roles[2]);
}

extern "C" int krabby_host_leader_board_id(void)
{
    return static_cast<int>(krabby::sim::BoardId::kFront);
}

extern "C" int krabby_host_board_usb_tx_line_available(int board_id)
{
    if (board_id < 0 || board_id > 2) return 0;
    return krabby::sim::serialPort(
               static_cast<krabby::sim::BoardId>(board_id),
               krabby::sim::SerialPortId::kUsb)
                   .txLineAvailable()
        ? 1
        : 0;
}

extern "C" void krabby_host_enable_three_board_firmware_loop(int enabled)
{
    krabby::sim::host::setThreeBoardFirmwareLoopEnabled(enabled != 0);
}

extern "C" int krabby_host_three_board_actuator_command_pwm(const char *actuator_name)
{
    return krabby::sim::host::threeBoardActuatorCommandPwm(actuator_name);
}

extern "C" int krabby_host_board_usb_pop_tx_line(int board_id, char *buf, int buf_len)
{
    if (board_id < 0 || board_id > 2 || buf == nullptr || buf_len <= 0) return -1;
    krabby::sim::SimSerialPort &port = krabby::sim::serialPort(
        static_cast<krabby::sim::BoardId>(board_id),
        krabby::sim::SerialPortId::kUsb);
    if (!port.txLineAvailable()) return 0;
    const std::string line = port.popTxLine();
    const int copy_len = static_cast<int>(line.size());
    if (copy_len >= buf_len) return -1;
    std::memcpy(buf, line.data(), static_cast<std::size_t>(copy_len));
    buf[copy_len] = '\0';
    return copy_len;
}

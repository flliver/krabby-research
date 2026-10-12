#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void krabby_host_reset(void);
void krabby_host_handle_jog(const char *actuator_name, int pwm);
void krabby_host_set_target(const char *actuator_name, float position);
void krabby_host_set_actuator_stops(
    const char *actuator_name,
    int min_stop,
    int max_stop);
void krabby_host_update_all(void);
int krabby_host_pin_pwm(int pin);
int krabby_host_pin_digital(int pin);
void krabby_host_set_analog_read(int pin, int value);
int krabby_host_analog_read(int pin);

void krabby_host_advance_millis(unsigned long delta_ms);
unsigned long krabby_host_millis(void);

void krabby_host_set_hall_edges_per_full_travel(uint32_t edges);
void krabby_host_sync_hall_joint(
    uint8_t hall_slot,
    float angle_rad,
    float lower_rad,
    float upper_rad);
int32_t krabby_host_hall_edge_count(uint8_t hall_slot);

uint8_t krabby_host_eeprom_read(int addr);
void krabby_host_eeprom_update(int addr, uint8_t value);
void krabby_host_save_calibration(void);
void krabby_host_load_calibration(void);
int krabby_host_actuator_min_stop(const char *actuator_name);
int krabby_host_actuator_max_stop(const char *actuator_name);

void krabby_host_serial_usb_rx_write(const uint8_t *data, int length);
int krabby_host_serial_usb_rx_available(void);
int krabby_host_serial_usb_rx_read(void);
int krabby_host_serial_usb_tx_available(void);
int krabby_host_serial_usb_tx_read(void);
void krabby_host_serial_usb_println(const char *text);

void krabby_host_firmware_setup(void);
void krabby_host_firmware_loop_once(void);
void krabby_host_firmware_reset_loop_state(void);

#ifdef __cplusplus
}
#endif

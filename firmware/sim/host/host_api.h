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

#ifdef __cplusplus
}
#endif

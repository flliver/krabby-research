#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void krabby_host_reset(void);
void krabby_host_handle_jog(const char *actuator_name, int pwm);
int krabby_host_pin_pwm(int pin);
int krabby_host_pin_digital(int pin);

#ifdef __cplusplus
}
#endif

#pragma once

// Host Arduino output shim for firmware-in-Isaac simulation.
// Include this instead of AVR Arduino.h when compiling the actuator drive path on the host.

#include "arduino_io.hpp"
#include "pin_state.hpp"
#include "read_stubs.hpp"
#include "sim_clock.hpp"

#include <cstdlib>

#define INPUT 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define LOW 0
#define HIGH 1

#ifndef constrain
#define constrain(value, low, high) \
    ((value) < (low) ? (low) : ((value) > (high) ? (high) : (value)))
#endif

#ifndef abs
inline int abs(int j) { return j < 0 ? -j : j; }
#endif

inline unsigned long millis() { return krabby::sim::simClock().nowMs(); }

inline void pinMode(int pin, int mode) { krabby::sim::pinState().pinMode(pin, mode); }

inline void digitalWrite(int pin, int value) { krabby::sim::pinState().digitalWrite(pin, value); }

inline int digitalRead(int pin) { return krabby::sim::pinState().digitalRead(pin); }

inline int analogRead(int pin) { return krabby::sim::pinState().analogInput(pin); }

inline void analogWrite(int pin, int value) { krabby::sim::pinState().analogWrite(pin, value); }

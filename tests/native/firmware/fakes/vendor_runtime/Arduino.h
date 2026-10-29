#pragma once

// The Arduino surface the pinned SparkFun libraries need to compile on the host.
// Declarations match the suite environments that define them at link time.
#include <stddef.h>
#include <stdint.h>

static constexpr uint8_t INPUT = 0, OUTPUT = 1, INPUT_PULLUP = 2;
static constexpr uint8_t LOW = 0, HIGH = 1;

unsigned long millis();
void delay(unsigned long);
void pinMode(uint8_t, uint8_t);
void digitalWrite(uint8_t, uint8_t);

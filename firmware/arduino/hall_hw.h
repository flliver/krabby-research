#pragma once
#include <Arduino.h>
#include <stdint.h>

// Hall sensing — implementation varies by KRABBY_PIN_REV:
//   Rev 1: Port C PCINT1 (D37,D36,D35,D32,D33,D34)
//   Rev 2: No Hall sensors
//   Rev 3: HallA on Port B PCINT0 (D50,D51,D52) + Port K PCINT2 (A12,A13,A14);
//          HallB on PORTF A0-A5 (sampled for quadrature direction).

void hallHwInit();
uint32_t hallHwGetEdgeCount(uint8_t hallSlot);  // cumulative A edges (debug; telemetry sends the signed count)
int32_t  hallHwGetSignedCount(uint8_t hallSlot); // signed quadrature count = position
void     hallHwResetCount(uint8_t hallSlot);     // zero both counts for one slot

// Storm breaker (see the comment in hall_hw.cpp): a fast-shaft encoder can saturate the
// PCINT vector and freeze loop() entirely, so the ISRs self-limit by masking off a pin
// that produces too many edges between loop() passes.
void    hallHwLoopPet();                  // call once per loop() pass; zeroes the storm edge counter
uint8_t hallHwStormMask();                // bit per slot whose PCINT was tripped off by a storm
void    hallHwStormRearm(uint8_t hallSlot); // clear the storm flag + re-enable that slot's PCINT

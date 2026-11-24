#pragma once
#include <Arduino.h>
#include <stdint.h>

// HallA edge counting — implementation varies by KRABBY_PIN_REV:
//   Rev 1: Port C PCINT1 (D37,D36,D35,D32,D33,D34)
//   Rev 2: No Hall sensors
//   Rev 3: Port B PCINT0 (D50,D51,D52) + Port K PCINT2 (A12,A13,A14)

// Signed edge counts: Rev 3 only (quadrature). On hips and knees the count rises
// on retract and falls on extend; the FRHY yaw counts the other way. Rev 1 counts
// A edges without direction, so its counts carry no sign.
void hallHwInit();
int32_t hallHwGetEdgeCount(uint8_t hallSlot);

// True when the counts are signed quadrature counts (Rev 3). False for Rev 1
// (unsigned A counting) and Rev 2 (no halls).
bool hallHwHasQuadrature();

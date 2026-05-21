#pragma once
#include <Arduino.h>
#include <stdint.h>

// HallA edge counting — implementation varies by KRABBY_PIN_REV:
//   Rev 1: Port C PCINT1 (D37,D36,D35,D32,D33,D34)
//   Rev 2: No Hall sensors
//   Rev 3: Port B PCINT0 (D50,D51,D52) + Port K PCINT2 (A12,A13,A14)

void hallHwInit();
uint32_t hallHwGetEdgeCount(uint8_t hallSlot);

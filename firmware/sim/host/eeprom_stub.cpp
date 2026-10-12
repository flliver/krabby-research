#include "EEPROM.h"

FakeEEPROM EEPROM;

void resetHostEeprom() { EEPROM.clear(); }

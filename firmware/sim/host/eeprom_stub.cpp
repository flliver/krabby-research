#include "EEPROM.h"

HostEeprom EEPROM;

void resetHostEeprom()
{
    EEPROM.clear();
    EEPROM.setActiveBoard(krabby::sim::BoardId::kFront);
}

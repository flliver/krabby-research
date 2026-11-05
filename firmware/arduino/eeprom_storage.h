#pragma once

#include <stddef.h>
#include <stdint.h>
#include <EEPROM.h>

template <typename Record, int Address>
class EepromStorage
{
public:
    void load(Record &record)
    {
        EEPROM.get(Address, record);
    }

    void writeRecord(const Record &record)
    {
        EEPROM.put(Address, record);
    }

    void updateMagic(uint8_t magic)
    {
        static_assert(sizeof(((Record *)nullptr)->magic) == sizeof(uint8_t),
                      "EEPROM record magic must be one byte");
        EEPROM.update(Address + offsetof(Record, magic), magic);
    }
};

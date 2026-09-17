#pragma once

#include <cstring>

struct FakeEEPROM
{
    unsigned char bytes[1024] = {};
    unsigned int writes = 0;
    template <typename T> void put(int address, const T &value)
    {
        std::memcpy(bytes + address, &value, sizeof(value));
        ++writes;
    }
    template <typename T> void get(int address, T &value)
    {
        std::memcpy(&value, bytes + address, sizeof(value));
    }
};

extern FakeEEPROM EEPROM;

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

struct FakeEEPROM
{
    static constexpr size_t kSize = 4096;

    unsigned char bytes[kSize] = {};
    unsigned int writes = 0;

    void clear()
    {
        std::memset(bytes, 0, sizeof(bytes));
        writes = 0;
    }

    uint8_t read(int address) const
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return 0;
        return bytes[address];
    }

    void write(int address, uint8_t value)
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return;
        bytes[address] = value;
        ++writes;
    }

    void update(int address, uint8_t value)
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return;
        if (bytes[address] == value) return;
        bytes[address] = value;
        ++writes;
    }

    template <typename T> void put(int address, const T &value)
    {
        if (address < 0 || static_cast<size_t>(address) + sizeof(T) > kSize) return;
        std::memcpy(bytes + address, &value, sizeof(value));
        ++writes;
    }

    template <typename T> void get(int address, T &value) const
    {
        if (address < 0 || static_cast<size_t>(address) + sizeof(T) > kSize) return;
        std::memcpy(&value, bytes + address, sizeof(value));
    }
};

extern FakeEEPROM EEPROM;

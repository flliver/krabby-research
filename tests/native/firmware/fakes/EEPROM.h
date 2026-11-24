#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

struct FakeEEPROM
{
    static constexpr size_t kSize = 4096;

    unsigned char bytes[kSize] = {};
    unsigned int writes = 0;
    // Power-loss simulation for put(): -1 writes everything; otherwise at most
    // this many changed bytes are written (in address order, as the AVR core's
    // put updates byte by byte) and the rest are dropped and counted.
    int writeBudget = -1;
    unsigned int droppedBytes = 0;

    void clear()
    {
        std::memset(bytes, 0, sizeof(bytes));
        writes = 0;
        writeBudget = -1;
        droppedBytes = 0;
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
        const unsigned char *source = reinterpret_cast<const unsigned char *>(&value);
        for (size_t i = 0; i < sizeof(T); ++i)
        {
            unsigned char &target = bytes[static_cast<size_t>(address) + i];
            if (target == source[i]) continue;
            if (writeBudget == 0)
            {
                ++droppedBytes;
                continue;
            }
            if (writeBudget > 0) --writeBudget;
            target = source[i];
        }
        ++writes;
    }

    template <typename T> void get(int address, T &value) const
    {
        if (address < 0 || static_cast<size_t>(address) + sizeof(T) > kSize) return;
        std::memcpy(&value, bytes + address, sizeof(value));
    }
};

extern FakeEEPROM EEPROM;

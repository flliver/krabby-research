#pragma once

#include "sim_serial.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

// In-memory AVR-style EEPROM for host firmware builds (4 KB per simulated board).
struct FakeEEPROM
{
    static constexpr size_t kSize = 4096;

    unsigned char bytes[kSize] = {};

    void clear() { std::memset(bytes, 0, sizeof(bytes)); }

    uint8_t read(int address) const
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return 0;
        return bytes[address];
    }

    void write(int address, uint8_t value)
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return;
        bytes[address] = value;
    }

    void update(int address, uint8_t value)
    {
        if (address < 0 || static_cast<size_t>(address) >= kSize) return;
        if (bytes[address] == value) return;
        bytes[address] = value;
    }

    template <typename T> void put(int address, const T &value)
    {
        if (address < 0 || static_cast<size_t>(address) + sizeof(T) > kSize) return;
        std::memcpy(bytes + address, &value, sizeof(value));
    }

    template <typename T> void get(int address, T &value) const
    {
        if (address < 0 || static_cast<size_t>(address) + sizeof(T) > kSize) return;
        std::memcpy(&value, bytes + address, sizeof(value));
    }
};

struct HostEeprom
{
    FakeEEPROM boards[3];
    krabby::sim::BoardId activeBoard = krabby::sim::BoardId::kFront;
    unsigned int writes = 0;

    krabby::sim::BoardId activeBoardId() const { return activeBoard; }

    void setActiveBoard(krabby::sim::BoardId board) { activeBoard = board; }

    FakeEEPROM &active() { return boards[static_cast<int>(activeBoard)]; }

    const FakeEEPROM &active() const { return boards[static_cast<int>(activeBoard)]; }

    void clear()
    {
        for (FakeEEPROM &board : boards)
        {
            board.clear();
        }
        writes = 0;
    }

    uint8_t read(int address) const { return active().read(address); }

    void write(int address, uint8_t value)
    {
        active().write(address, value);
        ++writes;
    }

    void update(int address, uint8_t value)
    {
        if (active().read(address) == value) return;
        active().update(address, value);
        ++writes;
    }

    template <typename T> void put(int address, const T &value)
    {
        active().put(address, value);
        ++writes;
    }

    template <typename T> void get(int address, T &value) const
    {
        active().get(address, value);
    }
};

extern HostEeprom EEPROM;

void resetHostEeprom();

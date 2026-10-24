#pragma once

#include "sim_i2c.hpp"

#include <cstddef>
#include <cstdint>

class TwoWire
{
public:
    void begin()
    {
        krabby::sim::simI2cBus().clearTimeoutFlag();
    }

    void end() {}

    void setClock(uint32_t /*clockHz*/) {}

    void setWireTimeout(uint32_t /*timeoutMicroseconds*/, bool /*resetOnTimeout*/) {}

    void beginTransmission(uint8_t address)
    {
        krabby::sim::simI2cBus().beginTransmission(address);
    }

    std::size_t write(uint8_t data)
    {
        return krabby::sim::simI2cBus().write(data);
    }

    std::size_t write(const uint8_t *buffer, std::size_t size)
    {
        return krabby::sim::simI2cBus().write(buffer, size);
    }

    uint8_t endTransmission(bool sendStop = true)
    {
        return krabby::sim::simI2cBus().endTransmission(sendStop);
    }

    uint8_t requestFrom(uint8_t address, uint8_t quantity, uint8_t sendStop = true)
    {
        return krabby::sim::simI2cBus().requestFrom(address, quantity, sendStop != 0);
    }

    int available() { return krabby::sim::simI2cBus().available(); }

    int read() { return krabby::sim::simI2cBus().read(); }

    void clearWireTimeoutFlag() { krabby::sim::simI2cBus().clearTimeoutFlag(); }

    bool getWireTimeoutFlag() const { return krabby::sim::simI2cBus().timeoutFlag(); }
};

extern TwoWire Wire;

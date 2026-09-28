#pragma once

#include <Arduino.h>
#include <Wire.h>
#include <stdint.h>

// Wire.end() must release the pins before this object is used.
class ArduinoI2cBus
{
public:
    ArduinoI2cBus(uint32_t clockHz, uint32_t timeoutMicroseconds)
        : clockHz_(clockHz), timeoutMicroseconds_(timeoutMicroseconds)
    {
        releaseData();
        sclRelease();
    }

    bool isSdaHigh()
    {
        releaseData();
        return digitalRead(SDA) == HIGH;
    }

    void sclLow()
    {
        digitalWrite(SCL, LOW);
        pinMode(SCL, OUTPUT);
    }

    void sclRelease()
    {
        pinMode(SCL, INPUT_PULLUP);
    }

    void halfBit()
    {
        const uint32_t delayUs = 1000000UL / (2UL * clockHz_);
        delayMicroseconds(delayUs == 0 ? 1 : delayUs);
    }

    void sendStop()
    {
        // Open-drain STOP: release SDA rather than driving it high.
        digitalWrite(SDA, LOW);
        pinMode(SDA, OUTPUT);
        halfBit();
        sclRelease();
        halfBit();
        releaseData();
        halfBit();
    }

    void restart()
    {
        Wire.begin();
        Wire.setClock(clockHz_);
        Wire.setWireTimeout(timeoutMicroseconds_, true);
    }

private:
    void releaseData()
    {
        pinMode(SDA, INPUT_PULLUP);
    }

    uint32_t clockHz_;
    uint32_t timeoutMicroseconds_;
};

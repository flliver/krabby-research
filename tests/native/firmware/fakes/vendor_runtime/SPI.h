#pragma once

#include <stddef.h>
#include <stdint.h>

// The host build drives every device over I2C. SPI exists only so the vendor
// libraries compile; any call is a test-setup error and aborts.
static constexpr uint8_t LSBFIRST = 0, MSBFIRST = 1;
static constexpr uint8_t SPI_MODE0 = 0x00, SPI_MODE1 = 0x04, SPI_MODE2 = 0x08, SPI_MODE3 = 0x0C;

class SPISettings
{
public:
    SPISettings() {}
    SPISettings(uint32_t, uint8_t, uint8_t) {}
};

class SPIClass
{
public:
    void begin();
    void beginTransaction(SPISettings);
    void endTransaction();
    uint8_t transfer(uint8_t);
    uint16_t transfer16(uint16_t);
    void transfer(void *, size_t);
};

extern SPIClass SPI;

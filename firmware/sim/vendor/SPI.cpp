#include "SPI.h"

#include <cstdio>
#include <cstdlib>

namespace {
[[noreturn]] void unsupported(const char *operation)
{
    std::fprintf(stderr, "SPI %s called in the host build; devices are I2C only\n", operation);
    std::abort();
}
}

void SPIClass::begin() { unsupported("begin"); }
void SPIClass::beginTransaction(SPISettings) { unsupported("beginTransaction"); }
void SPIClass::endTransaction() { unsupported("endTransaction"); }
uint8_t SPIClass::transfer(uint8_t) { unsupported("transfer"); }
uint16_t SPIClass::transfer16(uint16_t) { unsupported("transfer16"); }
void SPIClass::transfer(void *, size_t) { unsupported("transfer"); }

SPIClass SPI;

#pragma once

#include <stdint.h>

inline int16_t decodeInt16LittleEndian(const uint8_t *bytes)
{
    return static_cast<int16_t>(
        static_cast<uint16_t>(bytes[0]) |
        (static_cast<uint16_t>(bytes[1]) << 8)
    );
}

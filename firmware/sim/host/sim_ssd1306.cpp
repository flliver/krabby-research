#include "sim_ssd1306.hpp"

namespace krabby {
namespace sim {

namespace {

SimSsd1306 g_ssd1306;

uint32_t crc32Update(uint32_t crc, uint8_t byte)
{
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit)
    {
        const uint32_t mask = static_cast<uint32_t>(-(crc & 1u));
        crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
    return crc;
}

}  // namespace

void SimSsd1306::reset()
{
    for (std::size_t i = 0; i < kSimSsd1306BufferBytes; ++i)
    {
        buffer_[i] = 0;
    }
    gdramOffset_ = 0;
    dataMode_ = false;
    totalDataBytes_ = 0;
}

void SimSsd1306::writeDataByte(uint8_t value)
{
    if (gdramOffset_ < kSimSsd1306BufferBytes)
    {
        buffer_[gdramOffset_++] = value;
    }
    ++totalDataBytes_;
}

void SimSsd1306::onI2cWrite(uint8_t /*address*/, const uint8_t *bytes, std::size_t length)
{
    if (bytes == nullptr || length == 0)
    {
        return;
    }

    std::size_t index = 0;
    if (length > 0 && (bytes[0] == 0x00 || bytes[0] == 0x40))
    {
        dataMode_ = bytes[0] == 0x40;
        if (dataMode_)
        {
            gdramOffset_ = 0;
        }
        index = 1;
    }

    for (; index < length; ++index)
    {
        if (dataMode_)
        {
            writeDataByte(bytes[index]);
        }
    }
}

uint32_t SimSsd1306::bufferCrc32() const
{
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < kSimSsd1306BufferBytes; ++i)
    {
        crc = crc32Update(crc, buffer_[i]);
    }
    return ~crc;
}

SimSsd1306 &simSsd1306() { return g_ssd1306; }

void resetSimSsd1306() { g_ssd1306.reset(); }

}  // namespace sim
}  // namespace krabby

#pragma once

#include <cstddef>
#include <cstdint>

namespace krabby {
namespace sim {

static constexpr std::size_t kSimSsd1306Width = 128;
static constexpr std::size_t kSimSsd1306Height = 64;
static constexpr std::size_t kSimSsd1306PageCount = kSimSsd1306Height / 8;
static constexpr std::size_t kSimSsd1306BufferBytes =
    kSimSsd1306Width * kSimSsd1306PageCount;

class SimSsd1306
{
public:
    void reset();

    void onI2cWrite(uint8_t address, const uint8_t *bytes, std::size_t length);

    const uint8_t *buffer() const { return buffer_; }

    std::size_t bufferSize() const { return kSimSsd1306BufferBytes; }

    uint32_t bufferCrc32() const;

    std::size_t totalDataBytesReceived() const { return totalDataBytes_; }

private:
    void writeDataByte(uint8_t value);

    uint8_t buffer_[kSimSsd1306BufferBytes];
    std::size_t gdramOffset_ = 0;
    bool dataMode_ = false;
    std::size_t totalDataBytes_ = 0;
};

SimSsd1306 &simSsd1306();

void resetSimSsd1306();

}  // namespace sim
}  // namespace krabby

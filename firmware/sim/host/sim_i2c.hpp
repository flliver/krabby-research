#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace krabby {
namespace sim {

// Shipping leader-bus addresses (see firmware/sim/i2c_shipping_scope.py).
constexpr uint8_t kI2cAddrLsm6dsoPrimary = 0x6Bu;
constexpr uint8_t kI2cAddrLsm6dsoAlternate = 0x6Au;
constexpr uint8_t kI2cAddrSsd1306Primary = 0x3Du;
constexpr uint8_t kI2cAddrSsd1306Alternate = 0x3Cu;
constexpr uint8_t kI2cAddrIna228Pack = 0x40u;
constexpr uint8_t kI2cAddrIna228Midpoint = 0x41u;

constexpr uint8_t kI2cNackAddress = 2u;  // AVR Wire / endTransmission address NACK

class SimI2cBus
{
public:
    void reset();

    void setDevicePresent(uint8_t address, bool present);

    bool devicePresent(uint8_t address) const;

    void beginTransmission(uint8_t address);

    std::size_t write(uint8_t byte);

    std::size_t write(const uint8_t *buffer, std::size_t size);

    /** @return 0 on ACK, non-zero on NACK (AVR Wire convention). */
    uint8_t endTransmission(bool sendStop);

    uint8_t requestFrom(uint8_t address, uint8_t quantity, bool sendStop);

    int available() const;

    int read();

    void clearTimeoutFlag();

    bool timeoutFlag() const;

private:
    struct DeviceState
    {
        bool present = false;
        uint8_t registerPointer = 0;
        std::vector<uint8_t> writeLog;
    };

    void ensureDefaultDevices();

    DeviceState *deviceForAddress(uint8_t address);

    const DeviceState *deviceForAddress(uint8_t address) const;

    void pushReadBytes(uint8_t address, uint8_t quantity);

    uint8_t readRegisterByte(uint8_t address, uint8_t reg) const;

    uint8_t txAddress_ = 0;
    std::vector<uint8_t> txBuffer_;
    std::deque<uint8_t> rxQueue_;
    bool timeoutFlag_ = false;
    DeviceState devices_[128];
};

SimI2cBus &simI2cBus();

void resetSimI2c();

}  // namespace sim
}  // namespace krabby

#include "sim_i2c.hpp"

#include "sim_imu_model.hpp"
#include "sim_ssd1306.hpp"

namespace krabby {
namespace sim {

namespace {

SimI2cBus g_sim_i2c;

void setDefaultPresent(SimI2cBus &bus, uint8_t address, bool present)
{
    bus.setDevicePresent(address, present);
}

}  // namespace

void SimI2cBus::reset()
{
    txAddress_ = 0;
    txBuffer_.clear();
    rxQueue_.clear();
    timeoutFlag_ = false;
    for (DeviceState &device : devices_)
    {
        device = DeviceState{};
    }
    ensureDefaultDevices();
}

void SimI2cBus::ensureDefaultDevices()
{
    setDefaultPresent(*this, kI2cAddrLsm6dsoPrimary, true);
    setDefaultPresent(*this, kI2cAddrLsm6dsoAlternate, true);
    setDefaultPresent(*this, kI2cAddrSsd1306Primary, true);
    setDefaultPresent(*this, kI2cAddrSsd1306Alternate, true);
    setDefaultPresent(*this, kI2cAddrIna228Pack, true);
    setDefaultPresent(*this, kI2cAddrIna228Midpoint, true);
}

void SimI2cBus::setDevicePresent(uint8_t address, bool present)
{
    devices_[address].present = present;
    devices_[address].registerPointer = 0;
    devices_[address].writeLog.clear();
}

bool SimI2cBus::devicePresent(uint8_t address) const
{
    return devices_[address].present;
}

SimI2cBus::DeviceState *SimI2cBus::deviceForAddress(uint8_t address)
{
    return &devices_[address];
}

const SimI2cBus::DeviceState *SimI2cBus::deviceForAddress(uint8_t address) const
{
    return &devices_[address];
}

void SimI2cBus::beginTransmission(uint8_t address)
{
    txAddress_ = address;
    txBuffer_.clear();
}

std::size_t SimI2cBus::write(uint8_t byte)
{
    txBuffer_.push_back(byte);
    return 1;
}

std::size_t SimI2cBus::write(const uint8_t *buffer, std::size_t size)
{
    if (buffer == nullptr || size == 0) return 0;
    txBuffer_.insert(txBuffer_.end(), buffer, buffer + size);
    return size;
}

uint8_t SimI2cBus::endTransmission(bool /*sendStop*/)
{
    if (!devicePresent(txAddress_))
    {
        return kI2cNackAddress;
    }

    DeviceState *device = deviceForAddress(txAddress_);
    if (device == nullptr) return kI2cNackAddress;

    for (std::size_t i = 0; i < txBuffer_.size(); ++i)
    {
        device->writeLog.push_back(txBuffer_[i]);
    }

    if (txAddress_ == kI2cAddrSsd1306Primary || txAddress_ == kI2cAddrSsd1306Alternate)
    {
        simSsd1306().onI2cWrite(txAddress_, txBuffer_.data(), txBuffer_.size());
    }

    if (!txBuffer_.empty())
    {
        device->registerPointer = txBuffer_[0];
    }

    txBuffer_.clear();
    return 0;
}

uint8_t SimI2cBus::readRegisterByte(uint8_t address, uint8_t reg) const
{
    if (address == kI2cAddrLsm6dsoPrimary || address == kI2cAddrLsm6dsoAlternate)
    {
        return lsm6dsoRegisterByte(reg);
    }
    return 0;
}

void SimI2cBus::pushReadBytes(uint8_t address, uint8_t quantity)
{
    const DeviceState *device = deviceForAddress(address);
    if (device == nullptr) return;

    uint8_t reg = device->registerPointer;
    for (uint8_t i = 0; i < quantity; ++i)
    {
        rxQueue_.push_back(readRegisterByte(address, reg));
        ++reg;
    }
}

uint8_t SimI2cBus::requestFrom(uint8_t address, uint8_t quantity, bool /*sendStop*/)
{
    if (!devicePresent(address) || quantity == 0)
    {
        return 0;
    }

    pushReadBytes(address, quantity);
    return quantity;
}

int SimI2cBus::available() const
{
    return static_cast<int>(rxQueue_.size());
}

int SimI2cBus::read()
{
    if (rxQueue_.empty()) return -1;
    const uint8_t value = rxQueue_.front();
    rxQueue_.pop_front();
    return static_cast<int>(value);
}

void SimI2cBus::clearTimeoutFlag() { timeoutFlag_ = false; }

bool SimI2cBus::timeoutFlag() const { return timeoutFlag_; }

SimI2cBus &simI2cBus() { return g_sim_i2c; }

void resetSimI2c() { g_sim_i2c.reset(); }

}  // namespace sim
}  // namespace krabby

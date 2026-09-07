#pragma once

#include <stdint.h>

// Retry scheduling and bus clearing for failed I2C devices. A Wire timeout
// resets the AVR peripheral but cannot release SDA held by a slave or fault.

struct I2cRecoveryLimits
{
    uint8_t badTicksBeforeRetry;   // consecutive failures before trying again
    uint32_t retryIntervalMs;      // and no more than one attempt this often
};

// Schedules I2C device retries without Wire or GPIO dependencies.
class I2cRecoveryPolicy
{
public:
    I2cRecoveryPolicy()
        : badTicks_(0), lastAttemptMs_(0), hasAttempted_(false)
    {
    }

    void noteSuccess()
    {
        badTicks_ = 0;
    }

    // True when the caller should attempt reinitialization.
    bool noteFailure(uint32_t nowMs, const I2cRecoveryLimits &limits)
    {
        if (badTicks_ < 255)
            ++badTicks_;
        if (badTicks_ < limits.badTicksBeforeRetry)
            return false;
        // Unsigned subtraction handles millis() rollover.
        if (hasAttempted_ &&
            static_cast<uint32_t>(nowMs - lastAttemptMs_) < limits.retryIntervalMs)
            return false;
        lastAttemptMs_ = nowMs;
        hasAttempted_ = true;
        badTicks_ = 0;
        return true;
    }

    uint8_t badTicks() const { return badTicks_; }

private:
    uint8_t badTicks_;        // saturates at 255
    uint32_t lastAttemptMs_;
    bool hasAttempted_;
};

// Eight data bits plus ACK.
static constexpr uint8_t I2C_BUS_CLEAR_PULSES = 9;

enum class I2cBusRecovery : uint8_t
{
    NotNeeded,   // SDA was free; the peripheral was restarted anyway
    Cleared,     // SDA was held, and clocking it released it
    Stuck,       // SDA still held after nine clocks -- not a software problem
};

// After a failed clear, suppress more clocks until SDA is released.
class I2cStuckBusLatch
{
public:
    I2cStuckBusLatch() : isWaitingForRelease_(false) {}

    bool mayAttempt(bool isSdaHigh)
    {
        if (!isWaitingForRelease_)
            return true;
        if (!isSdaHigh)
            return false;
        isWaitingForRelease_ = false;
        return true;
    }

    void noteResult(I2cBusRecovery result)
    {
        isWaitingForRelease_ = result == I2cBusRecovery::Stuck;
    }

    bool isWaitingForRelease() const { return isWaitingForRelease_; }

private:
    bool isWaitingForRelease_;
};

template <class Bus>
I2cBusRecovery recoverI2cBus(Bus &bus)
{
    if (bus.isSdaHigh())
    {
        bus.restart();
        return I2cBusRecovery::NotNeeded;
    }

    for (uint8_t pulse = 0; pulse < I2C_BUS_CLEAR_PULSES; ++pulse)
    {
        bus.sclLow();
        bus.halfBit();
        bus.sclRelease();
        bus.halfBit();
        if (bus.isSdaHigh())
            break;
    }

    if (!bus.isSdaHigh())
        return I2cBusRecovery::Stuck;

    // Return the released slave to idle.
    bus.sendStop();
    bus.restart();
    return I2cBusRecovery::Cleared;
}

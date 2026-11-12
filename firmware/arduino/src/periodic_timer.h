#pragma once

#include <stdint.h>

class PeriodicTimer
{
public:
    explicit PeriodicTimer(uint32_t intervalMilliseconds)
        : intervalMilliseconds_(intervalMilliseconds),
          startedAtMilliseconds_(0),
          isStarted_(false)
    {
    }

    void restart(uint32_t nowMilliseconds)
    {
        startedAtMilliseconds_ = nowMilliseconds;
        isStarted_ = true;
    }

    bool isExpired(uint32_t nowMilliseconds) const
    {
        return isStarted_ &&
               static_cast<uint32_t>(
                   nowMilliseconds - startedAtMilliseconds_) >=
                   intervalMilliseconds_;
    }

private:
    uint32_t intervalMilliseconds_;
    uint32_t startedAtMilliseconds_;
    bool isStarted_;
};

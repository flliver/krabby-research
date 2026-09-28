#pragma once

#include <stdint.h>

#include "board_role.h"

class ControllerFreshnessTracker
{
public:
    ControllerFreshnessTracker()
        : lastUpdateMilliseconds_(0), hasBeenSeen_(false)
    {
    }

    static ControllerFreshnessTracker seenAt(uint32_t nowMilliseconds)
    {
        ControllerFreshnessTracker tracker;
        tracker.lastUpdateMilliseconds_ = nowMilliseconds;
        tracker.hasBeenSeen_ = true;
        return tracker;
    }

    bool isFresh(
        uint32_t nowMilliseconds,
        uint32_t timeoutMilliseconds) const
    {
        return hasBeenSeen_ &&
               nowMilliseconds - lastUpdateMilliseconds_ < timeoutMilliseconds;
    }

private:
    uint32_t lastUpdateMilliseconds_;
    bool hasBeenSeen_;
};

#pragma once

#include "ina228_adapter.h"

class Ina228Provider
{
public:
    explicit Ina228Provider(uint8_t address) : monitor_(address) {}

    Ina228Provider(uint8_t address, float shuntResistanceOhms,
        float shuntMaxCurrentAmps, bool resetChargeOnBegin = false)
        : monitor_(address, shuntResistanceOhms,
                   shuntMaxCurrentAmps, resetChargeOnBegin) {}

    Ina228Adapter &get() { return monitor_; }

private:
    Ina228Adapter monitor_;
};

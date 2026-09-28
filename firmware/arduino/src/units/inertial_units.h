#pragma once

#include "base_units.h"

static constexpr float METERS_PER_SECOND_SQUARED_PER_G = 9.80665f;

class MetersPerSecondSquared
    : public LinearUnit<MetersPerSecondSquared, float>
{
public:
    using LinearUnit<MetersPerSecondSquared, float>::LinearUnit;
};

#pragma once

#include "base_units.h"

class Celsius : public UnitValue<float>
{
public:
    using UnitValue<float>::UnitValue;
};

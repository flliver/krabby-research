#pragma once

#include "base_units.h"

class Milliseconds : public LinearUnit<Milliseconds, float>
{
public:
    using LinearUnit<Milliseconds, float>::LinearUnit;
};

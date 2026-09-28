#pragma once

#include "base_units.h"

class MilliAmps;
class MilliWatts;

class Amps : public LinearUnit<Amps, float>
{
public:
    using LinearUnit<Amps, float>::LinearUnit;

    constexpr MilliAmps toMilliAmps() const;
};

class MilliAmps : public LinearUnit<MilliAmps, float>
{
public:
    using LinearUnit<MilliAmps, float>::LinearUnit;

    constexpr Amps toAmps() const;
};

class Watts : public LinearUnit<Watts, float>
{
public:
    using LinearUnit<Watts, float>::LinearUnit;

    constexpr MilliWatts toMilliWatts() const;
};

class MilliWatts : public LinearUnit<MilliWatts, float>
{
public:
    using LinearUnit<MilliWatts, float>::LinearUnit;

    constexpr Watts toWatts() const;
};

class Volts : public LinearUnit<Volts, float>
{
public:
    using LinearUnit<Volts, float>::LinearUnit;
};

class Coulombs : public LinearUnit<Coulombs, float>
{
public:
    using LinearUnit<Coulombs, float>::LinearUnit;
};

class Ohms : public LinearUnit<Ohms, float>
{
public:
    using LinearUnit<Ohms, float>::LinearUnit;
};

// -----------------------------------------------------------------------------
// Unit conversions
// -----------------------------------------------------------------------------

constexpr MilliAmps Amps::toMilliAmps() const
{
    return MilliAmps(value() * 1000.0f);
}

constexpr Amps MilliAmps::toAmps() const
{
    return Amps(value() / 1000.0f);
}

constexpr MilliWatts Watts::toMilliWatts() const
{
    return MilliWatts(value() * 1000.0f);
}

constexpr Watts MilliWatts::toWatts() const
{
    return Watts(value() / 1000.0f);
}

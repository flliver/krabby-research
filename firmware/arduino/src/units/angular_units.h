#pragma once

#include "base_units.h"

static constexpr float RADIANS_PER_DEGREE = 0.017453293f;

class DegreesPerSecond;
class Degrees;

class RadiansPerSecond
    : public LinearUnit<RadiansPerSecond, float>
{
public:
    using LinearUnit<RadiansPerSecond, float>::LinearUnit;

    constexpr DegreesPerSecond toDegreesPerSecond() const;
};

class DegreesPerSecond
    : public LinearUnit<DegreesPerSecond, float>
{
public:
    using LinearUnit<DegreesPerSecond, float>::LinearUnit;

    constexpr RadiansPerSecond toRadiansPerSecond() const;
};

class Radians : public LinearUnit<Radians, float>
{
public:
    using LinearUnit<Radians, float>::LinearUnit;

    Degrees toDegrees(Rounding rounding = Rounding::Exact) const;
};

class Degrees : public LinearUnit<Degrees, float>
{
public:
    using LinearUnit<Degrees, float>::LinearUnit;

    constexpr Radians toRadians() const;
};

// -----------------------------------------------------------------------------
// Unit conversions
// -----------------------------------------------------------------------------

constexpr Radians Degrees::toRadians() const
{
    return Radians(value() * RADIANS_PER_DEGREE);
}

constexpr RadiansPerSecond DegreesPerSecond::toRadiansPerSecond() const
{
    return RadiansPerSecond(value() * RADIANS_PER_DEGREE);
}

constexpr DegreesPerSecond RadiansPerSecond::toDegreesPerSecond() const
{
    return DegreesPerSecond(value() / RADIANS_PER_DEGREE);
}

inline Degrees Radians::toDegrees(Rounding rounding) const
{
    switch (rounding)
    {
    case Rounding::Exact:
        return Degrees(value() / RADIANS_PER_DEGREE);
    case Rounding::HalfAwayFromZero:
        return Degrees(
            roundHalfAwayFromZero(value() / RADIANS_PER_DEGREE));
    }
    return Degrees(value() / RADIANS_PER_DEGREE);
}

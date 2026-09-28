#pragma once

#include <stdint.h>

template <typename Representation>
class UnitValue
{
public:
    constexpr UnitValue()
        : value_()
    {
    }

    explicit constexpr UnitValue(Representation value)
        : value_(value)
    {
    }

    constexpr Representation value() const
    {
        return value_;
    }

private:
    Representation value_;
};

template <typename Unit, typename Representation>
class LinearUnit : public UnitValue<Representation>
{
public:
    using UnitValue<Representation>::UnitValue;

    constexpr Unit operator+(Unit other) const
    {
        return Unit(this->value() + other.value());
    }

    constexpr Unit operator-(Unit other) const
    {
        return Unit(this->value() - other.value());
    }

    constexpr Unit scalarMultiply(Representation dimensionlessScalar) const
    {
        return Unit(this->value() * dimensionlessScalar);
    }

    constexpr Unit scalarDivide(Representation dimensionlessScalar) const
    {
        return Unit(this->value() / dimensionlessScalar);
    }

    constexpr bool operator<(Unit other) const
    {
        return this->value() < other.value();
    }

    constexpr bool operator>(Unit other) const
    {
        return this->value() > other.value();
    }
};

enum class Rounding : uint8_t
{
    Exact,
    HalfAwayFromZero,
};

constexpr float roundHalfAwayFromZero(float value)
{
    return value >= 0.0f
        ? static_cast<float>(static_cast<long>(value + 0.5f))
        : static_cast<float>(static_cast<long>(value - 0.5f));
}
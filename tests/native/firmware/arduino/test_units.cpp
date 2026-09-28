#include <type_traits>
#include <stdint.h>

#include "src/units/angular_units.h"
#include "src/units/base_units.h"
#include "src/units/electrical_units.h"
#include "src/units/inertial_units.h"
#include "src/units/temperature_units.h"
#include "src/units/time_units.h"
#include "unity.h"

class EncoderCounts : public LinearUnit<EncoderCounts, int16_t>
{
public:
    using LinearUnit<EncoderCounts, int16_t>::LinearUnit;
};

template <typename Value>
class HasScalarMultiply
{
private:
    template <typename Candidate>
    static auto test(int) -> decltype(
        static_cast<void>(
            static_cast<const Candidate &>(Candidate()).scalarMultiply(1.0f)),
        std::true_type());

    template <typename>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<Value>(0))::value;
};

static_assert(
    !std::is_convertible<float, MetersPerSecondSquared>::value,
    "raw floats must not implicitly become acceleration");
static_assert(
    !std::is_assignable<MetersPerSecondSquared &, float>::value,
    "raw floats must not bypass the acceleration unit boundary");
static_assert(
    HasScalarMultiply<MetersPerSecondSquared>::value,
    "linear units must support dimensionless scaling");
static_assert(
    !HasScalarMultiply<Celsius>::value,
    "value-only units must not acquire linear operations");
static_assert(
    sizeof(MetersPerSecondSquared) == sizeof(float),
    "the unit abstraction must not add storage");
static_assert(
    sizeof(Celsius) == sizeof(float),
    "the value-only abstraction must not add storage");
static_assert(
    sizeof(EncoderCounts) == sizeof(int16_t),
    "unit storage must follow its selected representation");
static_assert(
    HasScalarMultiply<Volts>::value,
    "electrical quantities must expose linear operations");
static_assert(
    HasScalarMultiply<Milliseconds>::value,
    "time quantities must expose linear operations");

void setUp() {}
void tearDown() {}

static void test_same_unit_assignment_preserves_value()
{
    MetersPerSecondSquared destination;
    destination = MetersPerSecondSquared(9.8f);
    TEST_ASSERT_EQUAL_FLOAT(9.8f, destination.value());

    Celsius temperature;
    temperature = Celsius(25.0f);
    TEST_ASSERT_EQUAL_FLOAT(25.0f, temperature.value());
}

static void test_acceleration_and_angular_rate_support_vector_operations()
{
    TEST_ASSERT_EQUAL_FLOAT(
        -4.0f,
        (MetersPerSecondSquared(3.0f).scalarMultiply(-2.0f) -
         MetersPerSecondSquared(-2.0f)).value());
    TEST_ASSERT_EQUAL_FLOAT(
        1.0f,
        (RadiansPerSecond(3.0f) -
         RadiansPerSecond(2.0f)).value());
    TEST_ASSERT_EQUAL_FLOAT(
        -3.0f,
        RadiansPerSecond(1.5f).scalarMultiply(-2.0f).value());
}

static void test_degrees_per_second_supports_calibration_operations()
{
    DegreesPerSecond sum(1.0f);
    sum = sum + DegreesPerSecond(3.0f);

    TEST_ASSERT_EQUAL_FLOAT(2.0f, sum.scalarDivide(2.0f).value());
    TEST_ASSERT_TRUE(DegreesPerSecond(1.0f) < DegreesPerSecond(2.0f));
    TEST_ASSERT_TRUE(DegreesPerSecond(2.0f) > DegreesPerSecond(1.0f));
    TEST_ASSERT_EQUAL_FLOAT(
        1.0f,
        (DegreesPerSecond(3.0f) - DegreesPerSecond(2.0f)).value());
}

static void test_angular_rate_conversion_round_trips()
{
    const DegreesPerSecond degrees(90.0f);
    TEST_ASSERT_FLOAT_WITHIN(
        0.0001f,
        degrees.value(),
        degrees.toRadiansPerSecond().toDegreesPerSecond().value());
}

static void test_linear_unit_supports_an_integer_representation()
{
    const EncoderCounts total =
        EncoderCounts(6) + EncoderCounts(2);
    TEST_ASSERT_EQUAL_INT16(8, total.value());
    TEST_ASSERT_EQUAL_INT16(4, total.scalarDivide(2).value());
}

static void test_angle_conversions_support_exact_and_rounded_results()
{
    const Radians tenth(0.1f);
    TEST_ASSERT_FLOAT_WITHIN(
        0.0001f, 5.729577f, tenth.toDegrees().value());
    TEST_ASSERT_EQUAL_FLOAT(
        6.0f,
        tenth.toDegrees(Rounding::HalfAwayFromZero).value());
    TEST_ASSERT_FLOAT_WITHIN(
        0.0001f,
        90.0f,
        Degrees(90.0f).toRadians().toDegrees().value());
}

static void test_electrical_milli_conversions_round_trip()
{
    TEST_ASSERT_EQUAL_FLOAT(1.5f, MilliAmps(1500.0f).toAmps().value());
    TEST_ASSERT_EQUAL_FLOAT(1500.0f, Amps(1.5f).toMilliAmps().value());
    TEST_ASSERT_EQUAL_FLOAT(37.5f, MilliWatts(37500.0f).toWatts().value());
    TEST_ASSERT_EQUAL_FLOAT(37500.0f, Watts(37.5f).toMilliWatts().value());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_same_unit_assignment_preserves_value);
    RUN_TEST(test_acceleration_and_angular_rate_support_vector_operations);
    RUN_TEST(test_degrees_per_second_supports_calibration_operations);
    RUN_TEST(test_angular_rate_conversion_round_trips);
    RUN_TEST(test_linear_unit_supports_an_integer_representation);
    RUN_TEST(test_angle_conversions_support_exact_and_rounded_results);
    RUN_TEST(test_electrical_milli_conversions_round_trip);
    return UNITY_END();
}

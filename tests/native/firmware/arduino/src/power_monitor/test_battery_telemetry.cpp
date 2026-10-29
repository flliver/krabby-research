#include <stdio.h>

#include <string>
#include <type_traits>

#include "src/telemetry.h"
#include "unity.h"


static_assert(
    std::is_same<decltype(PowerMonitorMeasurement::voltage), Volts>::value,
    "pack voltage must remain strongly typed");
static_assert(
    std::is_same<decltype(PowerMonitorMeasurement::current), Amps>::value,
    "pack current must remain strongly typed");
static_assert(
    std::is_same<decltype(PowerMonitorMeasurement::power), Watts>::value,
    "pack power must remain strongly typed");
static_assert(
    std::is_same<decltype(PowerMonitorMeasurement::charge), Coulombs>::value,
    "pack charge must remain strongly typed");

void setUp() {}
void tearDown() {}

class StringOutput
{
public:
    void print(char value)
    {
        text_ += value;
    }

    void print(const char* value)
    {
        text_ += value;
    }

    void print(float value, int precision)
    {
        char buffer[40];
        snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
        text_ += buffer;
    }

    void print(int value)
    {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%d", value);
        text_ += buffer;
    }

    void print(uint8_t value)
    {
        char buffer[16];
        snprintf(buffer, sizeof(buffer), "%u", static_cast<unsigned>(value));
        text_ += buffer;
    }

    const std::string& text() const
    {
        return text_;
    }

private:
    std::string text_;
};

struct BatteryInputs
{
    PowerMonitorMeasurement pack;
    PowerMonitorMeasurement midpoint;
    Volts inferredB;
    bool isDiverged;
    uint8_t region;
};

static void serializeInputs(StringOutput &out, const BatteryInputs &inputs)
{
    appendBatteryTelemetry(out, inputs.pack, inputs.midpoint,
        inputs.inferredB, inputs.isDiverged, inputs.region);
}

static BatteryInputs distinctiveFrame(
    bool isDiverged, uint8_t packRegion,
    bool isPackValid = true, bool isMidpointValid = true)
{
    BatteryInputs frame;
    frame.pack.voltage = Volts(26.55f);
    frame.pack.current = Amps(-12.34f);
    frame.pack.power = Watts(327.6f);
    frame.pack.charge = Coulombs(1450.2f);
    frame.pack.isValid = isPackValid;
    frame.midpoint.voltage = Volts(13.35f);
    frame.inferredB = Volts(13.20f);
    frame.isDiverged = isDiverged;
    frame.midpoint.isValid = isMidpointValid;
    frame.region = packRegion;
    return frame;
}

static BatteryInputs frameWithCurrent(float amps)
{
    BatteryInputs frame = distinctiveFrame(false, 0);
    frame.pack.current = Amps(amps);
    return frame;
}

static BatteryInputs frameWithPower(float watts)
{
    BatteryInputs frame = distinctiveFrame(false, 0);
    frame.pack.power = Watts(watts);
    return frame;
}

static BatteryInputs frameWithCharge(float coulombs)
{
    BatteryInputs frame = distinctiveFrame(false, 0);
    frame.pack.charge = Coulombs(coulombs);
    return frame;
}

static BatteryInputs frameWithBatteryA(float volts)
{
    BatteryInputs frame = distinctiveFrame(false, 0);
    frame.midpoint.voltage = Volts(volts);
    return frame;
}

static BatteryInputs frameWithBatteryB(float volts)
{
    BatteryInputs frame = distinctiveFrame(false, 0);
    frame.inferredB = Volts(volts);
    return frame;
}

static void test_serializes_exact_tag_delimiters_order_units_and_precision()
{
    StringOutput out;

    serializeInputs(out, distinctiveFrame(false, 0));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1",
        out.text().c_str());
}

static void test_serializes_divergence_and_full_byte_state_fields()
{
    StringOutput out;

    serializeInputs(out, distinctiveFrame(true, 255));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 1 255 1 1",
        out.text().c_str());
}

static void test_serializes_positive_zero_and_negative_pack_current_without_sign_change()
{
    StringOutput positive;
    StringOutput zero;
    StringOutput negative;

    serializeInputs(positive, frameWithCurrent(12.34f));
    serializeInputs(zero, frameWithCurrent(0.0f));
    serializeInputs(negative, frameWithCurrent(-12.34f));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 12.34 327.6 1450.2 13.35 13.20 0 0 1 1",
        positive.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 0.00 327.6 1450.2 13.35 13.20 0 0 1 1",
        zero.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1",
        negative.text().c_str());
}

static void test_serializes_zero_fractional_and_positive_pack_power_in_watts()
{
    StringOutput zero;
    StringOutput fractional;
    StringOutput positive;

    serializeInputs(zero, frameWithPower(0.0f));
    serializeInputs(fractional, frameWithPower(0.26f));
    serializeInputs(positive, frameWithPower(327.64f));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 0.0 1450.2 13.35 13.20 0 0 1 1",
        zero.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 0.3 1450.2 13.35 13.20 0 0 1 1",
        fractional.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1",
        positive.text().c_str());
}

static void test_serializes_positive_zero_and_negative_accumulated_charge()
{
    StringOutput positive;
    StringOutput zero;
    StringOutput negative;

    serializeInputs(positive, frameWithCharge(1450.24f));
    serializeInputs(zero, frameWithCharge(0.0f));
    serializeInputs(negative, frameWithCharge(-1450.24f));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1",
        positive.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 0.0 13.35 13.20 0 0 1 1",
        zero.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 -1450.2 13.35 13.20 0 0 1 1",
        negative.text().c_str());
}

static void test_serializes_distinct_battery_a_voltages_as_the_fifth_payload()
{
    StringOutput low;
    StringOutput intermediate;
    StringOutput high;

    serializeInputs(low, frameWithBatteryA(12.0f));
    serializeInputs(intermediate, frameWithBatteryA(12.73f));
    serializeInputs(high, frameWithBatteryA(13.4f));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 12.00 13.20 0 0 1 1",
        low.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 12.73 13.20 0 0 1 1",
        intermediate.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.40 13.20 0 0 1 1",
        high.text().c_str());
}

static void test_serializes_distinct_battery_b_voltages_as_the_sixth_payload()
{
    StringOutput low;
    StringOutput intermediate;
    StringOutput high;

    serializeInputs(low, frameWithBatteryB(12.0f));
    serializeInputs(intermediate, frameWithBatteryB(12.68f));
    serializeInputs(high, frameWithBatteryB(13.4f));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 12.00 0 0 1 1",
        low.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 12.68 0 0 1 1",
        intermediate.text().c_str());
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.40 0 0 1 1",
        high.text().c_str());
}

static void assertSerializes(uint8_t region, const char* expected)
{
    StringOutput out;
    serializeInputs(out, distinctiveFrame(false, region));
    TEST_ASSERT_EQUAL_STRING(expected, out.text().c_str());
}

// 3g.8: the region is a measurement and the control state is what the MCU is
// doing about it. Every region is exercised against a fixed control state, so a
// serializer that swapped or merged the two fields cannot pass.
static void test_serializes_every_pack_region_as_the_eighth_payload()
{
    assertSerializes(PACK_REGION_NORMAL,
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1");
    assertSerializes(PACK_REGION_WARN,
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 1 1 1");
    assertSerializes(PACK_REGION_SOFT_CUT,
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 2 1 1");
    assertSerializes(PACK_REGION_HARD_CUT,
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 3 1 1");
    assertSerializes(PACK_REGION_OVER_VOLT,
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 4 1 1");
}

// The region byte is full width: an unknown value stays visible rather than
// being coerced onto one we know, so a newer firmware is still readable.
static void test_serializes_an_unknown_region_byte()
{
    assertSerializes(200, ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 200 1 1");
    assertSerializes(255, ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 255 1 1");
}

// TASK-3 §4 says to append the frame the same way as the Task 1 IMU segment, and
// TASK-1:105 defines that segment's failure convention as an in-band valid byte.
// Two bytes here because the monitors fail and recover independently.
static void test_serializes_each_monitor_validity_independently()
{
    StringOutput both;
    serializeInputs(both, distinctiveFrame(false, 0, true, true));
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1", both.text().c_str());

    StringOutput packDown;
    serializeInputs(packDown, distinctiveFrame(false, 0, false, true));
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 0 1", packDown.text().c_str());

    StringOutput midDown;
    serializeInputs(midDown, distinctiveFrame(false, 0, true, false));
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 0", midDown.text().c_str());

    StringOutput neither;
    serializeInputs(neither, distinctiveFrame(false, 0, false, false));
    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 0 0", neither.text().c_str());
}

// The case the old early-return discarded: four good Pack fields thrown away
// because a different monitor failed. 3g.1-3g.4 require them unconditionally.
static void test_a_dead_midpoint_still_carries_the_pack_fields()
{
    StringOutput out;
    serializeInputs(out, distinctiveFrame(false, 0, true, false));

    TEST_ASSERT_EQUAL_STRING(
        ";BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 0", out.text().c_str());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_serializes_exact_tag_delimiters_order_units_and_precision);
    RUN_TEST(test_serializes_divergence_and_full_byte_state_fields);
    RUN_TEST(test_serializes_positive_zero_and_negative_pack_current_without_sign_change);
    RUN_TEST(test_serializes_zero_fractional_and_positive_pack_power_in_watts);
    RUN_TEST(test_serializes_positive_zero_and_negative_accumulated_charge);
    RUN_TEST(test_serializes_distinct_battery_a_voltages_as_the_fifth_payload);
    RUN_TEST(test_serializes_distinct_battery_b_voltages_as_the_sixth_payload);
    RUN_TEST(test_serializes_every_pack_region_as_the_eighth_payload);
    RUN_TEST(test_serializes_an_unknown_region_byte);
    RUN_TEST(test_serializes_each_monitor_validity_independently);
    RUN_TEST(test_a_dead_midpoint_still_carries_the_pack_fields);
    return UNITY_END();
}

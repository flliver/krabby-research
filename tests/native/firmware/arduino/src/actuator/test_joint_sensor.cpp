#include "unity.h"
#include "src/actuator/joint_sensor.h"

void setUp() {}
void tearDown() {}

namespace
{
constexpr int8_t EXTEND = +1;  // +PWM
constexpr int8_t RETRACT = -1;
// Retract counts up, so a count change of +n is n drive edges for -PWM.
constexpr int32_t EDGE_SIGN_FOR_RETRACT = -ACTUATOR_HALL_EXTEND_COUNT_SIGN;

void assertHall(MoveFeedback expected, bool quadrature, int8_t pwmSign, int32_t countDelta)
{
    TEST_ASSERT_EQUAL_INT(
        (int)expected, (int)classifyHallMotion(quadrature, pwmSign, countDelta));
}

void assertPot(
    MoveFeedback expected,
    uint8_t requiredInBand,
    int8_t pwmSign,
    int32_t countDelta,
    int startSnap,
    int endSnap)
{
    TEST_ASSERT_EQUAL_INT(
        (int)expected,
        (int)classifyPotAtRest(requiredInBand, pwmSign, countDelta, startSnap, endSnap));
}
}

static void test_sensor_kinds_keep_their_eeprom_values()
{
    TEST_ASSERT_EQUAL_UINT8(0, (uint8_t)JointSensor::Unknown);
    TEST_ASSERT_EQUAL_UINT8(1, (uint8_t)JointSensor::Pot);
    TEST_ASSERT_EQUAL_UINT8(2, (uint8_t)JointSensor::Hall);
}

static void test_option_bits_are_valid_and_distinct_from_zero()
{
    TEST_ASSERT_NOT_EQUAL(0, CAL_OPT_KNEE_NUDGE);
    TEST_ASSERT_EQUAL_UINT8(CAL_OPT_KNEE_NUDGE, CAL_OPT_KNEE_NUDGE & CAL_OPT_ALL);
    TEST_ASSERT_EQUAL_UINT8(0, CAL_DEFAULT_OPTIONS & ~CAL_OPT_ALL);
    // The knee nudge is the shipped default (owner's decision, 2026-11-23).
    TEST_ASSERT_EQUAL_UINT8(CAL_OPT_KNEE_NUDGE, CAL_DEFAULT_OPTIONS);
}

static void test_hall_drive_edges_follow_the_count_sign()
{
    // Extend counts down, retract counts up.
    TEST_ASSERT_EQUAL_INT32(0, hallDriveEdges(EXTEND, 0));
    TEST_ASSERT_EQUAL_INT32(5, hallDriveEdges(EXTEND, -5 * EDGE_SIGN_FOR_RETRACT));
    TEST_ASSERT_EQUAL_INT32(-5, hallDriveEdges(EXTEND, 5 * EDGE_SIGN_FOR_RETRACT));
    TEST_ASSERT_EQUAL_INT32(5, hallDriveEdges(RETRACT, 5 * EDGE_SIGN_FOR_RETRACT));
    TEST_ASSERT_EQUAL_INT32(-5, hallDriveEdges(RETRACT, -5 * EDGE_SIGN_FOR_RETRACT));
    TEST_ASSERT_EQUAL_INT32(-1, ACTUATOR_HALL_EXTEND_COUNT_SIGN);
    // Spot values with the signs written out: +PWM, count -30 is 30 drive edges.
    TEST_ASSERT_EQUAL_INT32(30, hallDriveEdges(EXTEND, -30));
    TEST_ASSERT_EQUAL_INT32(30, hallDriveEdges(RETRACT, 30));
}

static void test_hall_motion_needs_a_quadrature_build()
{
    assertHall(MoveFeedback::None, false, EXTEND, -5000);
    assertHall(MoveFeedback::None, false, RETRACT, 5000);
    assertHall(MoveFeedback::None, false, EXTEND, 5000);
    assertHall(MoveFeedback::None, false, RETRACT, 0);
}

static void test_hall_motion_threshold_is_twelve_edges_in_each_direction()
{
    TEST_ASSERT_EQUAL_INT(12, CAL_HALL_DETECT_EDGES);
    // +PWM extends: the count falls.
    assertHall(MoveFeedback::None, true, EXTEND, 0);
    assertHall(MoveFeedback::None, true, EXTEND, -11);
    assertHall(MoveFeedback::Hall, true, EXTEND, -12);
    assertHall(MoveFeedback::Hall, true, EXTEND, -13);
    assertHall(MoveFeedback::None, true, EXTEND, 11);
    assertHall(MoveFeedback::HallWrongWay, true, EXTEND, 12);
    assertHall(MoveFeedback::HallWrongWay, true, EXTEND, 13);
    // -PWM retracts: the count rises.
    assertHall(MoveFeedback::None, true, RETRACT, 0);
    assertHall(MoveFeedback::None, true, RETRACT, 11);
    assertHall(MoveFeedback::Hall, true, RETRACT, 12);
    assertHall(MoveFeedback::Hall, true, RETRACT, 13);
    assertHall(MoveFeedback::None, true, RETRACT, -11);
    assertHall(MoveFeedback::HallWrongWay, true, RETRACT, -12);
    assertHall(MoveFeedback::HallWrongWay, true, RETRACT, -13);
}

static void test_hall_motion_handles_large_counts()
{
    assertHall(MoveFeedback::Hall, true, RETRACT, 20000);
    assertHall(MoveFeedback::HallWrongWay, true, RETRACT, -20000);
    assertHall(MoveFeedback::Hall, true, EXTEND, -2000000000L);
}

static void test_pot_band_edges()
{
    TEST_ASSERT_FALSE(isPotRestInBand(CAL_POT_DETECT_MIN_RAW - 1));
    TEST_ASSERT_TRUE(isPotRestInBand(CAL_POT_DETECT_MIN_RAW));
    TEST_ASSERT_TRUE(isPotRestInBand(CAL_POT_DETECT_MAX_RAW));
    TEST_ASSERT_FALSE(isPotRestInBand(CAL_POT_DETECT_MAX_RAW + 1));
    TEST_ASSERT_EQUAL_INT(31, CAL_POT_DETECT_MIN_RAW - 1);
    TEST_ASSERT_EQUAL_INT(991, CAL_POT_DETECT_MAX_RAW + 1);
    // Hall-B rest levels are outside the band.
    TEST_ASSERT_FALSE(isPotRestInBand(0));
    TEST_ASSERT_FALSE(isPotRestInBand(CAL_HALLB_REST_LOW_MAX));
    TEST_ASSERT_FALSE(isPotRestInBand(CAL_HALLB_REST_HIGH_MIN));
    TEST_ASSERT_FALSE(isPotRestInBand(1023));
}

static void test_pot_travel_threshold_without_a_band_rule()
{
    // requiredInBand 0: non-quadrature build, any readings, any count.
    assertPot(MoveFeedback::None, 0, EXTEND, 0, 500, 500 + 29);
    assertPot(MoveFeedback::Pot, 0, EXTEND, 0, 500, 500 + 30);
    assertPot(MoveFeedback::None, 0, EXTEND, 0, 500, 500 - 29);
    assertPot(MoveFeedback::PotWrongWay, 0, EXTEND, 0, 500, 500 - 30);
    assertPot(MoveFeedback::None, 0, RETRACT, 0, 500, 500 - 29);
    assertPot(MoveFeedback::Pot, 0, RETRACT, 0, 500, 500 - 30);
    assertPot(MoveFeedback::None, 0, RETRACT, 0, 500, 500 + 29);
    assertPot(MoveFeedback::PotWrongWay, 0, RETRACT, 0, 500, 500 + 30);
    // Rev 2 pots may span 3..1020: no band applies, and the count is ignored.
    assertPot(MoveFeedback::Pot, 0, EXTEND, 0, 3, 1020);
    assertPot(MoveFeedback::Pot, 0, RETRACT, 7, 1020, 3);
    assertPot(MoveFeedback::Pot, 0, EXTEND, -500, 3, 1020);
}

static void test_pot_classification_needs_both_snapshots_in_band()
{
    // requiredInBand 2: quadrature build, joint Unknown or Hall.
    assertPot(MoveFeedback::Pot, 2, EXTEND, 0, 100, 200);
    assertPot(MoveFeedback::Pot, 2, EXTEND, 0, CAL_POT_DETECT_MIN_RAW, CAL_POT_DETECT_MAX_RAW);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 31, 200);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 100, 991);
    assertPot(MoveFeedback::None, 2, RETRACT, 0, 991, 100);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 31, 991);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 0, 1023);
    // A wrong-way pot is only reported with both snapshots in band.
    assertPot(MoveFeedback::PotWrongWay, 2, EXTEND, 0, 300, 200);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 1000, 200);
    // Travel threshold inside the band.
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 100, 129);
    assertPot(MoveFeedback::Pot, 2, EXTEND, 0, 100, 130);
    assertPot(MoveFeedback::PotWrongWay, 2, EXTEND, 0, 130, 100);
    assertPot(MoveFeedback::Pot, 2, RETRACT, 0, 130, 100);
}

static void test_proven_pot_needs_one_snapshot_in_band()
{
    // requiredInBand 1: quadrature build, joint already Pot.
    assertPot(MoveFeedback::Pot, 1, EXTEND, 0, 20, 100);
    assertPot(MoveFeedback::Pot, 1, EXTEND, 0, 100, 1000);
    assertPot(MoveFeedback::Pot, 1, EXTEND, 0, 100, 200);
    assertPot(MoveFeedback::PotWrongWay, 1, EXTEND, 0, 100, 20);
    assertPot(MoveFeedback::None, 1, EXTEND, 0, 31, 991 + 50);
    assertPot(MoveFeedback::None, 1, EXTEND, 0, 5, 1015);
    assertPot(MoveFeedback::None, 1, RETRACT, 0, 1015, 5);
}

static void test_pot_never_counts_edges()
{
    for (uint8_t required = 1; required <= 2; ++required)
    {
        assertPot(MoveFeedback::Pot, required, EXTEND, 1, 100, 200);
        assertPot(MoveFeedback::Pot, required, EXTEND, -1, 100, 200);
        assertPot(MoveFeedback::Pot, required, EXTEND, 0, 100, 200);
        assertPot(MoveFeedback::None, required, EXTEND, 2, 100, 200);
        assertPot(MoveFeedback::None, required, EXTEND, -2, 100, 200);
        assertPot(MoveFeedback::None, required, RETRACT, 2, 200, 100);
        assertPot(MoveFeedback::None, required, RETRACT, -2, 200, 100);
    }
    TEST_ASSERT_EQUAL_INT(1, CAL_POT_DETECT_MAX_EDGES);
}

static void test_rail_flip_of_a_hall_b_line_is_never_pot_evidence()
{
    // Hall B flips from its high rest level to its low one between snapshots.
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 1015, 6);
    assertPot(MoveFeedback::None, 2, RETRACT, 0, 1015, 6);
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 6, 1015);
    assertPot(MoveFeedback::None, 2, RETRACT, 0, 6, 1015);
    assertPot(MoveFeedback::None, 1, EXTEND, 0, 1015, 6);
    assertPot(MoveFeedback::None, 1, RETRACT, 0, 6, 1015);
    // The same readings on a build with no hall-B pins would be pot travel.
    assertPot(MoveFeedback::Pot, 0, RETRACT, 0, 1015, 6);
    assertPot(MoveFeedback::PotWrongWay, 0, EXTEND, 0, 1015, 6);
    // A flip that lands inside the band for one snapshot only is still rejected.
    assertPot(MoveFeedback::None, 2, EXTEND, 0, 1015, 500);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_sensor_kinds_keep_their_eeprom_values);
    RUN_TEST(test_option_bits_are_valid_and_distinct_from_zero);
    RUN_TEST(test_hall_drive_edges_follow_the_count_sign);
    RUN_TEST(test_hall_motion_needs_a_quadrature_build);
    RUN_TEST(test_hall_motion_threshold_is_twelve_edges_in_each_direction);
    RUN_TEST(test_hall_motion_handles_large_counts);
    RUN_TEST(test_pot_band_edges);
    RUN_TEST(test_pot_travel_threshold_without_a_band_rule);
    RUN_TEST(test_pot_classification_needs_both_snapshots_in_band);
    RUN_TEST(test_proven_pot_needs_one_snapshot_in_band);
    RUN_TEST(test_pot_never_counts_edges);
    RUN_TEST(test_rail_flip_of_a_hall_b_line_is_never_pot_evidence);
    return UNITY_END();
}

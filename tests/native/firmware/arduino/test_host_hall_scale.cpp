#include "unity.h"

#include "hall_encoder_scale.hpp"
#include "hall_hw.h"
#include "hall_sim.hpp"
#include "pin_state.hpp"
#include "sim_clock.hpp"
#include "src/actuator/actuator_constants.h"

using krabby::sim::hallSimAddEdges;
using krabby::sim::hallSimEdgeCount;
using krabby::sim::hallSimEdgeCountBoard;
using krabby::sim::hallSimQuadrature;
using krabby::sim::hallSimRetargetJointAngle;
using krabby::sim::hallSimSetEdgeCount;
using krabby::sim::hallSimSetQuadrature;
using krabby::sim::hallSimSyncJointAngle;
using krabby::sim::pinState;
using krabby::sim::setHallEdgesPerFullTravel;

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static const float kLower = 0.0f;
static const float kUpper = 1.0f;

static void test_full_joint_span_accumulates_configured_edges()
{
    hallHwInit();
    hallSimSyncJointAngle(0, kLower, kLower, kUpper);
    hallSimSyncJointAngle(0, kUpper, kLower, kUpper);
    const int32_t edges = hallHwGetEdgeCount(0);
    // A rising angle is extension: the count falls by the extend sign.
    TEST_ASSERT_INT_WITHIN(
        2, ACTUATOR_HALL_EXTEND_COUNT_SIGN * static_cast<int>(kHallEdgesPerFullTravel), edges);
}

static void test_hall_count_holds_when_joint_stops()
{
    hallHwInit();
    const float lower = -0.4f;
    const float upper = 0.6f;
    hallSimSyncJointAngle(2, 0.1f, lower, upper);
    hallSimSyncJointAngle(2, 0.1f, lower, upper);
    const int32_t first = hallHwGetEdgeCount(2);
    hallSimSyncJointAngle(2, 0.1f, lower, upper);
    TEST_ASSERT_EQUAL_INT32(first, hallHwGetEdgeCount(2));
}

static void test_extension_counts_down_and_retraction_counts_up()
{
    setHallEdgesPerFullTravel(1000);
    hallHwInit();
    hallSimSyncJointAngle(1, 0.5f, kLower, kUpper);  // baseline
    hallSimSyncJointAngle(1, 0.6f, kLower, kUpper);  // extend by a tenth of the span
    TEST_ASSERT_INT_WITHIN(1, 100 * ACTUATOR_HALL_EXTEND_COUNT_SIGN, hallHwGetEdgeCount(1));
    hallSimSyncJointAngle(1, 0.4f, kLower, kUpper);  // retract by two tenths
    TEST_ASSERT_INT_WITHIN(1, 100 * -ACTUATOR_HALL_EXTEND_COUNT_SIGN, hallHwGetEdgeCount(1));
    hallSimSyncJointAngle(1, 0.5f, kLower, kUpper);  // back to the start
    TEST_ASSERT_INT_WITHIN(2, 0, hallHwGetEdgeCount(1));
}

static void test_sub_edge_jitter_counts_nothing_in_either_direction()
{
    setHallEdgesPerFullTravel(100);
    hallHwInit();
    hallSimSyncJointAngle(0, 0.5f, kLower, kUpper);
    hallSimSyncJointAngle(0, 0.5f + 0.004f, kLower, kUpper);
    hallSimSyncJointAngle(0, 0.5f, kLower, kUpper);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));
}

static void test_degenerate_span_counts_nothing()
{
    hallHwInit();
    hallSimSyncJointAngle(0, 0.0f, 1.0f, 1.0f);
    hallSimSyncJointAngle(0, 0.5f, 1.0f, 1.0f);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));
}

static void test_retarget_sets_baseline_without_counting()
{
    setHallEdgesPerFullTravel(1000);
    hallHwInit();
    hallSimRetargetJointAngle(3, 0.2f, kLower, kUpper);
    hallSimSyncJointAngle(3, 0.2f, kLower, kUpper);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(3));
    hallSimRetargetJointAngle(3, 0.9f, kLower, kUpper);  // moved with the motor stopped
    hallSimSyncJointAngle(3, 0.9f, kLower, kUpper);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(3));
}

static void test_counts_are_per_board()
{
    setHallEdgesPerFullTravel(1000);
    pinState().setActiveBoard(1);
    hallHwInit();
    hallSimSyncJointAngle(0, 0.2f, kLower, kUpper);
    hallSimSyncJointAngle(0, 0.7f, kLower, kUpper);
    const int32_t left = hallHwGetEdgeCount(0);
    TEST_ASSERT_INT_WITHIN(2, 500 * ACTUATOR_HALL_EXTEND_COUNT_SIGN, left);

    pinState().setActiveBoard(0);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));
    pinState().setActiveBoard(2);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));
    TEST_ASSERT_EQUAL_INT32(left, hallSimEdgeCountBoard(1, 0));
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(0, 0));
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(2, 0));

    // The same slot on another board keeps its own baseline and count.
    hallSimSyncJointAngle(0, 0.9f, kLower, kUpper);
    hallSimSyncJointAngle(0, 0.8f, kLower, kUpper);
    TEST_ASSERT_INT_WITHIN(2, 100 * -ACTUATOR_HALL_EXTEND_COUNT_SIGN, hallHwGetEdgeCount(0));
    TEST_ASSERT_EQUAL_INT32(left, hallSimEdgeCountBoard(1, 0));
}

static void test_init_zeroes_only_the_active_board_and_keeps_the_scale()
{
    setHallEdgesPerFullTravel(200);
    for (int board = 0; board < 3; ++board)
    {
        hallSimSetEdgeCount(board, 4, 100 + board);
        hallSimSetEdgeCount(board, 5, -(50 + board));
    }
    pinState().setActiveBoard(1);
    hallHwInit();
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(1, 4));
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(1, 5));
    TEST_ASSERT_EQUAL_INT32(100, hallSimEdgeCountBoard(0, 4));
    TEST_ASSERT_EQUAL_INT32(-50, hallSimEdgeCountBoard(0, 5));
    TEST_ASSERT_EQUAL_INT32(102, hallSimEdgeCountBoard(2, 4));
    TEST_ASSERT_EQUAL_INT32(-52, hallSimEdgeCountBoard(2, 5));

    // The scale survives the init: a full span still counts 200 edges.
    hallSimSyncJointAngle(0, kLower, kLower, kUpper);
    hallSimSyncJointAngle(0, kUpper, kLower, kUpper);
    TEST_ASSERT_INT_WITHIN(1, 200 * ACTUATOR_HALL_EXTEND_COUNT_SIGN, hallHwGetEdgeCount(0));
}

static void test_init_drops_the_baseline_of_the_active_board_only()
{
    setHallEdgesPerFullTravel(1000);
    hallSimSyncJointAngle(0, 0.1f, kLower, kUpper);  // board 0 baseline
    pinState().setActiveBoard(1);
    hallSimSyncJointAngle(0, 0.1f, kLower, kUpper);  // board 1 baseline
    hallHwInit();                                     // board 1 restarts
    hallSimSyncJointAngle(0, 0.6f, kLower, kUpper);  // first sample after init: baseline only
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));

    pinState().setActiveBoard(0);
    hallSimSyncJointAngle(0, 0.6f, kLower, kUpper);  // board 0 kept its baseline
    TEST_ASSERT_INT_WITHIN(1, 500 * ACTUATOR_HALL_EXTEND_COUNT_SIGN, hallHwGetEdgeCount(0));
}

static void test_reset_clears_every_board_the_scale_and_the_quadrature_flags()
{
    setHallEdgesPerFullTravel(7);
    for (int board = 0; board < 3; ++board)
    {
        hallSimSetEdgeCount(board, 0, 40 + board);
        hallSimSetQuadrature(board, true);
    }
    krabby::sim::resetHallSim();
    for (int board = 0; board < 3; ++board)
    {
        TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(board, 0));
        TEST_ASSERT_FALSE(hallSimQuadrature(board));
    }
    hallSimSyncJointAngle(0, kLower, kLower, kUpper);
    hallSimSyncJointAngle(0, kUpper, kLower, kUpper);
    TEST_ASSERT_INT_WITHIN(
        2, ACTUATOR_HALL_EXTEND_COUNT_SIGN * static_cast<int>(kHallEdgesPerFullTravel),
        hallHwGetEdgeCount(0));
}

static void test_quadrature_defaults_off_and_follows_the_active_board()
{
    TEST_ASSERT_FALSE(hallHwHasQuadrature());
    hallSimSetQuadrature(1, true);
    TEST_ASSERT_FALSE(hallHwHasQuadrature());  // board 0 is still active
    pinState().setActiveBoard(1);
    TEST_ASSERT_TRUE(hallHwHasQuadrature());
    TEST_ASSERT_TRUE(hallSimQuadrature(1));
    TEST_ASSERT_FALSE(hallSimQuadrature(0));
    TEST_ASSERT_FALSE(hallSimQuadrature(2));
    pinState().setActiveBoard(2);
    TEST_ASSERT_FALSE(hallHwHasQuadrature());

    hallSimSetQuadrature(1, false);
    pinState().setActiveBoard(1);
    TEST_ASSERT_FALSE(hallHwHasQuadrature());
}

static void test_init_keeps_the_quadrature_flag()
{
    hallSimSetQuadrature(0, true);
    hallHwInit();
    TEST_ASSERT_TRUE(hallHwHasQuadrature());
}

static void test_set_add_and_read_edges_per_board_are_signed()
{
    hallSimSetEdgeCount(2, 3, -120);
    TEST_ASSERT_EQUAL_INT32(-120, hallSimEdgeCountBoard(2, 3));
    hallSimAddEdges(2, 3, 150);
    TEST_ASSERT_EQUAL_INT32(30, hallSimEdgeCountBoard(2, 3));
    hallSimAddEdges(2, 3, -100);
    TEST_ASSERT_EQUAL_INT32(-70, hallSimEdgeCountBoard(2, 3));
    pinState().setActiveBoard(2);
    TEST_ASSERT_EQUAL_INT32(-70, hallHwGetEdgeCount(3));
    TEST_ASSERT_EQUAL_INT32(-70, hallSimEdgeCount(3));
}

static void test_add_edges_saturates_at_the_int32_range()
{
    hallSimSetEdgeCount(0, 0, INT32_MAX - 5);
    hallSimAddEdges(0, 0, 100);
    TEST_ASSERT_EQUAL_INT32(INT32_MAX, hallSimEdgeCountBoard(0, 0));
    hallSimSetEdgeCount(0, 1, INT32_MIN + 5);
    hallSimAddEdges(0, 1, -100);
    TEST_ASSERT_EQUAL_INT32(INT32_MIN, hallSimEdgeCountBoard(0, 1));
}

static void test_out_of_range_board_or_slot_reads_zero_and_writes_nothing()
{
    hallSimSetEdgeCount(0, 5, 9);
    hallSimSetEdgeCount(3, 0, 5);
    hallSimSetEdgeCount(-1, 0, 5);
    hallSimSetEdgeCount(0, 6, 5);
    hallSimAddEdges(3, 0, 5);
    hallSimAddEdges(0, 6, 5);
    hallSimSetQuadrature(3, true);
    hallSimSetQuadrature(-1, true);
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(3, 0));
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(-1, 0));
    TEST_ASSERT_EQUAL_INT32(0, hallSimEdgeCountBoard(0, 6));
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(6));
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(255));
    TEST_ASSERT_EQUAL_INT32(9, hallHwGetEdgeCount(5));
    TEST_ASSERT_FALSE(hallSimQuadrature(3));
    TEST_ASSERT_FALSE(hallSimQuadrature(-1));
    // A sync on an out-of-range slot is ignored.
    hallSimSyncJointAngle(6, 0.0f, kLower, kUpper);
    hallSimSyncJointAngle(6, 1.0f, kLower, kUpper);
    hallSimRetargetJointAngle(6, 1.0f, kLower, kUpper);
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(6));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_full_joint_span_accumulates_configured_edges);
    RUN_TEST(test_hall_count_holds_when_joint_stops);
    RUN_TEST(test_extension_counts_down_and_retraction_counts_up);
    RUN_TEST(test_sub_edge_jitter_counts_nothing_in_either_direction);
    RUN_TEST(test_degenerate_span_counts_nothing);
    RUN_TEST(test_retarget_sets_baseline_without_counting);
    RUN_TEST(test_counts_are_per_board);
    RUN_TEST(test_init_zeroes_only_the_active_board_and_keeps_the_scale);
    RUN_TEST(test_init_drops_the_baseline_of_the_active_board_only);
    RUN_TEST(test_reset_clears_every_board_the_scale_and_the_quadrature_flags);
    RUN_TEST(test_quadrature_defaults_off_and_follows_the_active_board);
    RUN_TEST(test_init_keeps_the_quadrature_flag);
    RUN_TEST(test_set_add_and_read_edges_per_board_are_signed);
    RUN_TEST(test_add_edges_saturates_at_the_int32_range);
    RUN_TEST(test_out_of_range_board_or_slot_reads_zero_and_writes_nothing);
    return UNITY_END();
}

#include "unity.h"

#include "hall_encoder_scale.hpp"
#include "hall_hw.h"
#include "hall_sim.hpp"
#include "sim_clock.hpp"

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_full_joint_span_accumulates_configured_edges()
{
    hallHwInit();
    const float lower = 0.0f;
    const float upper = 1.0f;
    krabby::sim::hallSimSyncJointAngle(0, lower, lower, upper);
    krabby::sim::hallSimSyncJointAngle(0, upper, lower, upper);
    const int32_t edges = hallHwGetEdgeCount(0);
    TEST_ASSERT_INT_WITHIN(2, static_cast<int>(kHallEdgesPerFullTravel), edges);
}

static void test_hall_count_holds_when_joint_stops()
{
    hallHwInit();
    const float lower = -0.4f;
    const float upper = 0.6f;
    krabby::sim::hallSimSyncJointAngle(2, 0.1f, lower, upper);
    krabby::sim::hallSimSyncJointAngle(2, 0.1f, lower, upper);
    const int32_t first = hallHwGetEdgeCount(2);
    krabby::sim::hallSimSyncJointAngle(2, 0.1f, lower, upper);
    TEST_ASSERT_EQUAL_INT32(first, hallHwGetEdgeCount(2));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_full_joint_span_accumulates_configured_edges);
    RUN_TEST(test_hall_count_holds_when_joint_stops);
    return UNITY_END();
}

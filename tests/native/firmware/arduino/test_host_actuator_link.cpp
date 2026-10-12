#include "unity.h"

#include "Arduino.h"
#include "actuator_manager.h"
#include "hall_hw.h"

using krabby::sim::pinState;

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_shipping_actuator_manual_drive_records_pins()
{
    LinearActuator actuator("FLHY", 2, 3, 22, 6, 0, 0);
    actuator.init();
    actuator.manualDrive(100);

    TEST_ASSERT_TRUE(pinState().isDigitalHigh(22));
    TEST_ASSERT_EQUAL_INT(100, pinState().pwmDuty(2));
    TEST_ASSERT_EQUAL_INT(0, pinState().pwmDuty(3));
    TEST_ASSERT_EQUAL_INT(100, pinState().signedPwmCommand(2, 3, 22));
}

static void test_read_and_hall_stubs_are_available_at_link_time()
{
    TEST_ASSERT_EQUAL_INT(512, analogRead(0));
    TEST_ASSERT_EQUAL_INT(0, digitalRead(22));
    TEST_ASSERT_EQUAL_INT32(0, hallHwGetEdgeCount(0));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_shipping_actuator_manual_drive_records_pins);
    RUN_TEST(test_read_and_hall_stubs_are_available_at_link_time);
    return UNITY_END();
}

#include "unity.h"

#include "Arduino.h"

using krabby::sim::pinState;
using krabby::sim::simClock;

void setUp() { krabby::sim::resetHostSim(); }
void tearDown() {}

static void test_millis_uses_sim_clock()
{
    simClock().set(42);
    TEST_ASSERT_EQUAL_UINT32(42, millis());
    simClock().advance(8);
    TEST_ASSERT_EQUAL_UINT32(50, millis());
}

static void test_pin_mode_and_pwm_writes_are_recorded()
{
    pinMode(2, OUTPUT);
    pinMode(22, OUTPUT);
    TEST_ASSERT_EQUAL_INT(OUTPUT, pinState().getMode(2));
    TEST_ASSERT_EQUAL_INT(OUTPUT, pinState().getMode(22));

    digitalWrite(22, HIGH);
    analogWrite(2, 122);
    analogWrite(3, 0);
    TEST_ASSERT_TRUE(pinState().isDigitalHigh(22));
    TEST_ASSERT_EQUAL_INT(122, pinState().pwmDuty(2));
    TEST_ASSERT_EQUAL_INT(0, pinState().pwmDuty(3));
}

static void test_analog_write_clamps_duty()
{
    analogWrite(10, 300);
    TEST_ASSERT_EQUAL_INT(255, pinState().pwmDuty(10));
    analogWrite(10, -1);
    TEST_ASSERT_EQUAL_INT(0, pinState().pwmDuty(10));
}

static void test_signed_pwm_matches_drive_actuator_pattern()
{
    const int pwm_r = 2;
    const int pwm_l = 3;
    const int en = 22;

    digitalWrite(en, HIGH);
    analogWrite(pwm_r, 200);
    analogWrite(pwm_l, 0);
    TEST_ASSERT_EQUAL_INT(200, pinState().signedPwmCommand(pwm_r, pwm_l, en));

    analogWrite(pwm_r, 0);
    analogWrite(pwm_l, 150);
    TEST_ASSERT_EQUAL_INT(-150, pinState().signedPwmCommand(pwm_r, pwm_l, en));

    digitalWrite(en, LOW);
    TEST_ASSERT_EQUAL_INT(0, pinState().signedPwmCommand(pwm_r, pwm_l, en));
}

static void test_constrain_and_abs_macros()
{
    TEST_ASSERT_EQUAL_INT(5, constrain(5, 0, 10));
    TEST_ASSERT_EQUAL_INT(0, constrain(-3, 0, 10));
    TEST_ASSERT_EQUAL_INT(10, constrain(99, 0, 10));
    TEST_ASSERT_EQUAL_INT(7, abs(-7));
    TEST_ASSERT_EQUAL_INT(4, abs(4));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_millis_uses_sim_clock);
    RUN_TEST(test_pin_mode_and_pwm_writes_are_recorded);
    RUN_TEST(test_analog_write_clamps_duty);
    RUN_TEST(test_signed_pwm_matches_drive_actuator_pattern);
    RUN_TEST(test_constrain_and_abs_macros);
    return UNITY_END();
}

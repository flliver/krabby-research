#include <stdio.h>

#include "unity.h"

#include "board_pins.h"

void setUp() {}
void tearDown() {}

// Every output pin belongs here for collision checking.
static const int drivenPins[] = {
    PIN_S0_PWMR, PIN_S0_PWML, PIN_S1_PWMR, PIN_S1_PWML,
    PIN_S2_PWMR, PIN_S2_PWML, PIN_S3_PWMR, PIN_S3_PWML,
    PIN_S4_PWMR, PIN_S4_PWML, PIN_S5_PWMR, PIN_S5_PWML,
    PIN_S0_EN,   PIN_S1_EN,   PIN_S2_EN,
    PIN_S3_EN,   PIN_S4_EN,   PIN_S5_EN,
    STATUS_LED_PIN,
};

// This translation unit checks the selected pin revision.
static void test_no_two_subsystems_drive_the_same_pin()
{
    const size_t count = sizeof(drivenPins) / sizeof(drivenPins[0]);
    for (size_t i = 0; i < count; ++i)
    {
        for (size_t j = i + 1; j < count; ++j)
        {
            if (drivenPins[i] != drivenPins[j]) continue;

            char message[64];
            snprintf(message, sizeof(message),
                     "rev %d: pin %d is driven by two subsystems",
                     KRABBY_PIN_REV, drivenPins[i]);
            TEST_FAIL_MESSAGE(message);
        }
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_no_two_subsystems_drive_the_same_pin);
    return UNITY_END();
}

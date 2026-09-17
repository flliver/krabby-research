#include <string.h>

#include "src/actuator/actuator_identity.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

static const char *ACTUATOR_NAMES[ActuatorId::ActuatorCount] = {
    "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL",
    "MLHY", "MLHL", "MLKL", "MRHY", "MRHL", "MRKL",
    "RLHY", "RLHL", "RLKL", "RRHY", "RRHL", "RRKL",
};

static void test_actuator_ids_iterate_in_index_order()
{
    size_t expected = 0;
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        TEST_ASSERT_EQUAL_UINT(
            expected, static_cast<size_t>(actuatorId));
        ++expected;
    }
    TEST_ASSERT_EQUAL_UINT(
        static_cast<size_t>(ActuatorId::ActuatorCount), expected);
}

static void test_every_actuator_wire_name_is_unique_and_four_characters()
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        const char *name = ACTUATOR_NAMES[actuatorId];
        TEST_ASSERT_EQUAL_UINT(4, strlen(name));

        for (ActuatorId otherActuatorId = ActuatorId::FLHY;
             otherActuatorId < ActuatorId::ActuatorCount;
             ++otherActuatorId)
        {
            if (otherActuatorId >= actuatorId)
                continue;
            TEST_ASSERT_NOT_EQUAL(
                0, strcmp(name, ACTUATOR_NAMES[otherActuatorId]));
        }
    }
}

static void test_every_actuator_wire_name_parses_back_to_its_id()
{
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
        TEST_ASSERT_EQUAL(actuatorId, parseActuatorId(ACTUATOR_NAMES[actuatorId]));
}

static void test_invalid_actuator_names_map_to_the_invalid_sentinel()
{
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId(nullptr));
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId(""));
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId("F"));
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId("FLH"));
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId("FLXX"));
    TEST_ASSERT_EQUAL(ActuatorId::ActuatorCount, parseActuatorId("XXXX"));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_actuator_ids_iterate_in_index_order);
    RUN_TEST(test_every_actuator_wire_name_is_unique_and_four_characters);
    RUN_TEST(test_every_actuator_wire_name_parses_back_to_its_id);
    RUN_TEST(test_invalid_actuator_names_map_to_the_invalid_sentinel);
    return UNITY_END();
}

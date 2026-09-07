#include "src/controller/board_role.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

static void test_eeprom_role_values_remain_compatible()
{
    TEST_ASSERT_EQUAL_INT(0, ROLE_UNKNOWN);
    TEST_ASSERT_EQUAL_INT(1, ROLE_FRONT);
    TEST_ASSERT_EQUAL_INT(2, ROLE_LEFT);
    TEST_ASSERT_EQUAL_INT(3, ROLE_RIGHT);
}

static void test_role_iteration_contains_only_assigned_roles()
{
    TEST_ASSERT_EQUAL_UINT(3, sizeof(ALL_BOARD_ROLES) / sizeof(ALL_BOARD_ROLES[0]));
    TEST_ASSERT_EQUAL_INT(ROLE_FRONT, ALL_BOARD_ROLES[0]);
    TEST_ASSERT_EQUAL_INT(ROLE_LEFT, ALL_BOARD_ROLES[1]);
    TEST_ASSERT_EQUAL_INT(ROLE_RIGHT, ALL_BOARD_ROLES[2]);
}

static void test_role_labels_match_the_wire_protocol()
{
    TEST_ASSERT_EQUAL_STRING("UNKWN", boardRoleLabel(ROLE_UNKNOWN));
    TEST_ASSERT_EQUAL_STRING("FRONT", boardRoleLabel(ROLE_FRONT));
    TEST_ASSERT_EQUAL_STRING("LEFT", boardRoleLabel(ROLE_LEFT));
    TEST_ASSERT_EQUAL_STRING("RIGHT", boardRoleLabel(ROLE_RIGHT));
}

static void test_each_actuator_maps_to_its_physical_board()
{
    const BoardRole expected[ActuatorId::ActuatorCount] = {
        ROLE_FRONT, ROLE_FRONT, ROLE_FRONT,
        ROLE_FRONT, ROLE_FRONT, ROLE_FRONT,
        ROLE_LEFT, ROLE_LEFT, ROLE_LEFT,
        ROLE_RIGHT, ROLE_RIGHT, ROLE_RIGHT,
        ROLE_LEFT, ROLE_LEFT, ROLE_LEFT,
        ROLE_RIGHT, ROLE_RIGHT, ROLE_RIGHT,
    };
    for (ActuatorId actuatorId = ActuatorId::FLHY;
         actuatorId < ActuatorId::ActuatorCount;
         ++actuatorId)
    {
        TEST_ASSERT_EQUAL_INT(
            expected[actuatorId], getBoardRoleForActuator(actuatorId));
    }
    TEST_ASSERT_EQUAL_INT(
        ROLE_UNKNOWN, getBoardRoleForActuator(ActuatorId::ActuatorCount));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_eeprom_role_values_remain_compatible);
    RUN_TEST(test_role_iteration_contains_only_assigned_roles);
    RUN_TEST(test_role_labels_match_the_wire_protocol);
    RUN_TEST(test_each_actuator_maps_to_its_physical_board);
    return UNITY_END();
}

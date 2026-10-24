#include <string.h>

#include "src/actuator/actuator_role_names.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

static void assert_names(
    const char *const *names,
    const char *n0,
    const char *n1,
    const char *n2,
    const char *n3,
    const char *n4,
    const char *n5)
{
    TEST_ASSERT_EQUAL_STRING(n0, names[0]);
    TEST_ASSERT_EQUAL_STRING(n1, names[1]);
    TEST_ASSERT_EQUAL_STRING(n2, names[2]);
    TEST_ASSERT_EQUAL_STRING(n3, names[3]);
    TEST_ASSERT_EQUAL_STRING(n4, names[4]);
    TEST_ASSERT_EQUAL_STRING(n5, names[5]);
}

static void test_role_name_tables_match_board_joint_ids()
{
    assert_names(
        actuatorNamesForRole(ROLE_FRONT),
        "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL");
    assert_names(
        actuatorNamesForRole(ROLE_LEFT),
        "RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL");
    assert_names(
        actuatorNamesForRole(ROLE_RIGHT),
        "RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL");
    // Timeout / unknown still uses front actuator names (same as shipping sketch).
    assert_names(
        actuatorNamesForRole(ROLE_UNKNOWN),
        "FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL");
}

static void test_role_name_tables_are_six_unique_ids()
{
    TEST_ASSERT_EQUAL_UINT(6, ACTUATOR_ROLE_NAME_COUNT);
    const BoardRole roles[] = {ROLE_FRONT, ROLE_LEFT, ROLE_RIGHT};
    for (BoardRole role : roles)
    {
        const char *const *names = actuatorNamesForRole(role);
        for (size_t i = 0; i < ACTUATOR_ROLE_NAME_COUNT; ++i)
        {
            TEST_ASSERT_EQUAL_UINT(4, strlen(names[i]));
            for (size_t j = 0; j < i; ++j)
                TEST_ASSERT_NOT_EQUAL(0, strcmp(names[i], names[j]));
        }
    }
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_role_name_tables_match_board_joint_ids);
    RUN_TEST(test_role_name_tables_are_six_unique_ids);
    return UNITY_END();
}

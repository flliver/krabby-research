#include <limits>

#include "unity.h"

#include "src/cli/power_calibration_command.h"

void setUp() {}
void tearDown() {}

static void test_top_level_command_prefix_is_calibration()
{
    TEST_ASSERT_EQUAL_CHAR('C', CALIBRATION_COMMAND_PREFIX);
    TEST_ASSERT_EQUAL_STRING(
        "ACTUATOR", ACTUATOR_CALIBRATION_TARGET);
    TEST_ASSERT_EQUAL_STRING(
        "PWR_SENSE", POWER_SENSOR_CALIBRATION_TARGET);
}

static void test_actuator_calibration_accepts_only_bare_or_explicit_target()
{
    const char* explicitTarget[] = {"ACTUATOR"};
    const char* lowercaseTarget[] = {"actuator"};
    const char* extraToken[] = {"ACTUATOR", "extra"};
    const char* powerSensor[] = {"PWR_SENSE"};

    TEST_ASSERT_TRUE(isActuatorCalibrationCommand(0, nullptr));
    TEST_ASSERT_TRUE(isActuatorCalibrationCommand(1, explicitTarget));
    TEST_ASSERT_TRUE(isActuatorCalibrationCommand(1, lowercaseTarget));
    TEST_ASSERT_FALSE(isActuatorCalibrationCommand(1, nullptr));
    TEST_ASSERT_FALSE(isActuatorCalibrationCommand(2, extraToken));
    TEST_ASSERT_FALSE(isActuatorCalibrationCommand(1, powerSensor));
}

static void test_complete_tokens_select_each_action_case_insensitively()
{
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::CalibrateVoltageOffsets),
        static_cast<int>(parsePowerCalibrationAction("PWR_SENSE", "VOLTAGE")));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::CalibrateCurrentScale),
        static_cast<int>(parsePowerCalibrationAction("pwr_sense", "current")));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::ShowCalibration),
        static_cast<int>(parsePowerCalibrationAction("Pwr_Sense", "Show")));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::ShowHelp),
        static_cast<int>(parsePowerCalibrationAction("PWR_SENSE", "?")));
}

static void test_missing_wrong_and_partial_tokens_are_invalid()
{
    const char* invalid[][2] = {
        {"", "VOLTAGE"},
        {"POWER", "VOLTAGE"},
        {"PWR", ""},
        {"PWR_SENSE", "V"},
        {"PWR_SENSE", "CURRENTLY"},
        {"PWR_SENSE", "SHUNT"},
        {"PWR_SENSE", "LIST"},
        {nullptr, "SHOW"},
        {"PWR_SENSE", nullptr},
    };

    for (size_t index = 0; index < 9; ++index)
    {
        TEST_ASSERT_EQUAL_INT(
            static_cast<int>(PowerCalibrationAction::Invalid),
            static_cast<int>(parsePowerCalibrationAction(
                invalid[index][0], invalid[index][1])));
    }
}

static void test_token_comparison_rejects_nulls_and_mismatched_lengths()
{
    TEST_ASSERT_FALSE(powerCalibrationTokenEquals(nullptr, "SHOW"));
    TEST_ASSERT_FALSE(powerCalibrationTokenEquals("SHOW", nullptr));
    TEST_ASSERT_FALSE(powerCalibrationTokenEquals("SHOW", "SHOWING"));
    TEST_ASSERT_FALSE(powerCalibrationTokenEquals("SHOWING", "SHOW"));
}

static void test_valid_numbers_are_parsed_exactly()
{
    const char* tokens[] = {
        "0", "10", "-10", "+10.25", "25.84", "1e-2", "-3.5E+1",
    };
    const float expected[] = {
        0.0f, 10.0f, -10.0f, 10.25f, 25.84f, 0.01f, -35.0f,
    };

    for (size_t index = 0; index < 7; ++index)
    {
        float result = 99.0f;
        TEST_ASSERT_TRUE(parsePowerCalibrationNumber(tokens[index], result));
        TEST_ASSERT_FLOAT_WITHIN(0.00001f, expected[index], result);
    }
}

static void test_invalid_numbers_do_not_mutate_result()
{
    const char* invalid[] = {
        "",
        " ",
        "abc",
        "1abc",
        "1 2",
        "nan",
        "NaN",
        "inf",
        "-inf",
        "1e999",
        nullptr,
    };

    for (size_t index = 0; index < 11; ++index)
    {
        float result = 99.0f;
        TEST_ASSERT_FALSE(
            parsePowerCalibrationNumber(invalid[index], result));
        TEST_ASSERT_EQUAL_FLOAT(99.0f, result);
    }
}

static void test_number_too_large_for_float_is_rejected()
{
    float result = 99.0f;
    TEST_ASSERT_FALSE(parsePowerCalibrationNumber("1e39", result));
    TEST_ASSERT_EQUAL_FLOAT(99.0f, result);
}

static void test_complete_commands_require_exact_argument_counts()
{
    const char* voltage[] = {"PWR_SENSE", "VOLTAGE", "25.84", "12.91"};
    const char* current[] = {"PWR_SENSE", "CURRENT", "-10.0"};
    const char* show[] = {"PWR_SENSE", "SHOW"};
    const char* help[] = {"PWR_SENSE", "?"};
    PowerCalibrationCommand result = {
        PowerCalibrationAction::Invalid, 99.0f, 98.0f};

    TEST_ASSERT_TRUE(parsePowerCalibrationCommand(4, voltage, result));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::CalibrateVoltageOffsets),
        static_cast<int>(result.action));
    TEST_ASSERT_FLOAT_WITHIN(0.00001f, 25.84f, result.firstReference);
    TEST_ASSERT_FLOAT_WITHIN(0.00001f, 12.91f, result.secondReference);

    TEST_ASSERT_TRUE(parsePowerCalibrationCommand(3, current, result));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::CalibrateCurrentScale),
        static_cast<int>(result.action));
    TEST_ASSERT_FLOAT_WITHIN(0.00001f, -10.0f, result.firstReference);

    TEST_ASSERT_TRUE(parsePowerCalibrationCommand(2, show, result));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::ShowCalibration),
        static_cast<int>(result.action));
    TEST_ASSERT_TRUE(parsePowerCalibrationCommand(2, help, result));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(PowerCalibrationAction::ShowHelp),
        static_cast<int>(result.action));
}

static void test_invalid_complete_commands_preserve_prior_result()
{
    const char* missingVoltage[] = {"PWR_SENSE", "VOLTAGE", "25.84"};
    const char* extraVoltage[] = {
        "PWR_SENSE", "VOLTAGE", "25.84", "12.91", "extra"};
    const char* missingCurrent[] = {"PWR_SENSE", "CURRENT"};
    const char* extraCurrent[] = {"PWR_SENSE", "CURRENT", "10", "extra"};
    const char* extraShow[] = {"PWR_SENSE", "SHOW", "extra"};
    const char* badNumber[] = {"PWR_SENSE", "CURRENT", "10A"};
    const char** invalid[] = {
        missingVoltage,
        extraVoltage,
        missingCurrent,
        extraCurrent,
        extraShow,
        badNumber,
    };
    const size_t counts[] = {3, 5, 2, 4, 3, 3};

    for (size_t index = 0; index < 6; ++index)
    {
        PowerCalibrationCommand result = {
            PowerCalibrationAction::ShowCalibration, 99.0f, 98.0f};
        TEST_ASSERT_FALSE(parsePowerCalibrationCommand(
            counts[index], invalid[index], result));
        TEST_ASSERT_EQUAL_INT(
            static_cast<int>(PowerCalibrationAction::ShowCalibration),
            static_cast<int>(result.action));
        TEST_ASSERT_EQUAL_FLOAT(99.0f, result.firstReference);
        TEST_ASSERT_EQUAL_FLOAT(98.0f, result.secondReference);
    }

    PowerCalibrationCommand result = {
        PowerCalibrationAction::ShowCalibration, 99.0f, 98.0f};
    TEST_ASSERT_FALSE(parsePowerCalibrationCommand(0, nullptr, result));
    TEST_ASSERT_EQUAL_FLOAT(99.0f, result.firstReference);

    const char* oneToken[] = {"PWR_SENSE"};
    TEST_ASSERT_FALSE(parsePowerCalibrationCommand(1, oneToken, result));

    const char* badFirstVoltage[] = {
        "PWR_SENSE", "VOLTAGE", "bad", "12.0"};
    const char* badSecondVoltage[] = {
        "PWR_SENSE", "VOLTAGE", "25.0", "bad"};
    const char* invalidAction[] = {"PWR_SENSE", "RESET"};
    TEST_ASSERT_FALSE(parsePowerCalibrationCommand(
        4, badFirstVoltage, result));
    TEST_ASSERT_FALSE(parsePowerCalibrationCommand(
        4, badSecondVoltage, result));
    TEST_ASSERT_FALSE(parsePowerCalibrationCommand(
        2, invalidAction, result));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_top_level_command_prefix_is_calibration);
    RUN_TEST(test_actuator_calibration_accepts_only_bare_or_explicit_target);
    RUN_TEST(test_complete_tokens_select_each_action_case_insensitively);
    RUN_TEST(test_missing_wrong_and_partial_tokens_are_invalid);
    RUN_TEST(test_token_comparison_rejects_nulls_and_mismatched_lengths);
    RUN_TEST(test_valid_numbers_are_parsed_exactly);
    RUN_TEST(test_invalid_numbers_do_not_mutate_result);
    RUN_TEST(test_number_too_large_for_float_is_rejected);
    RUN_TEST(test_complete_commands_require_exact_argument_counts);
    RUN_TEST(test_invalid_complete_commands_preserve_prior_result);
    return UNITY_END();
}

#include <initializer_list>
#include <sstream>
#include <string>

#include "unity.h"

#include "src/cli/actuator_calibration_command.h"

namespace
{

struct FakeOutput final : public CalibrationTextSink
{
    std::string text;

    void printFlash(const char *value) override { text += value; }
    void printText(const char *value) override { text += value; }
    void printNumber(long value) override { text += std::to_string(value); }
    void printChar(char value) override { text += value; }
    void endLine() override { text += '\n'; }
};

ActuatorCalibrationCommandKind parse(
    std::initializer_list<const char *> tokens,
    ActuatorCalibrationRequest &request)
{
    return parseActuatorCalibrationCommand(tokens.size(), tokens.begin(), request);
}

void assertRun(std::initializer_list<const char *> tokens, uint8_t passes, int pwm)
{
    ActuatorCalibrationRequest request = {0, 0};
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorCalibrationCommandKind::Run),
        static_cast<int>(parse(tokens, request)));
    TEST_ASSERT_EQUAL_UINT(passes, request.passes);
    TEST_ASSERT_EQUAL_INT(pwm, request.pwm);
}

void assertKind(std::initializer_list<const char *> tokens, ActuatorCalibrationCommandKind kind)
{
    ActuatorCalibrationRequest request = {0, 7};
    TEST_ASSERT_EQUAL_INT(static_cast<int>(kind), static_cast<int>(parse(tokens, request)));
    TEST_ASSERT_EQUAL_INT(7, request.pwm);  // untouched unless Run
}

}  // namespace

void setUp() {}
void tearDown() {}

static void test_bare_and_explicit_forms_run()
{
    ActuatorCalibrationRequest request = {0, 0};
    TEST_ASSERT_EQUAL_INT(static_cast<int>(ActuatorCalibrationCommandKind::Run),
        static_cast<int>(parseActuatorCalibrationCommand(0, nullptr, request)));
    TEST_ASSERT_EQUAL_UINT(CAL_PASS_ALL, request.passes);
    TEST_ASSERT_EQUAL_INT(CAL_DEFAULT_PWM, request.pwm);

    assertRun({"120"}, CAL_PASS_ALL, 120);
    assertRun({"SEEK"}, CAL_PASS_SEEK, CAL_DEFAULT_PWM);
    assertRun({"seek", "110"}, CAL_PASS_SEEK, 110);
    assertRun({"SWEEP"}, CAL_PASS_SWEEP, CAL_DEFAULT_PWM);
    assertRun({"Sweep", "150"}, CAL_PASS_SWEEP, 150);
    assertRun({"ACTUATOR"}, CAL_PASS_ALL, CAL_DEFAULT_PWM);
    assertRun({"actuator", "SEEK", "125"}, CAL_PASS_SEEK, 125);
    assertRun({"ACTUATOR", "0130"}, CAL_PASS_ALL, 130);
}

static void test_bad_pwm_and_extra_tokens_are_invalid()
{
    const ActuatorCalibrationCommandKind invalid = ActuatorCalibrationCommandKind::Invalid;
    assertKind({"109"}, invalid);
    assertKind({"40"}, invalid);  // the old minimum: stalls mid-stroke with no current signature
    assertKind({"80"}, invalid);
    assertKind({"151"}, invalid);
    assertKind({"0"}, invalid);
    assertKind({"-120"}, invalid);
    assertKind({"+120"}, invalid);
    assertKind({"120.5"}, invalid);
    assertKind({"12O"}, invalid);
    assertKind({"99999999999999999999"}, invalid);
    assertKind({"fast"}, invalid);
    assertKind({"SEEK", "SWEEP"}, invalid);
    assertKind({"SEEK", "120", "x"}, invalid);
    assertKind({"120", "SEEK"}, invalid);
    assertKind({"ACTUATOR", "ACTUATOR"}, invalid);
    assertKind({""}, invalid);
    ActuatorCalibrationRequest request = {0, 7};
    TEST_ASSERT_EQUAL_INT(static_cast<int>(invalid),
        static_cast<int>(parseActuatorCalibrationCommand(1, nullptr, request)));

    int pwm = 5;
    TEST_ASSERT_FALSE(parseActuatorCalibrationPwm(nullptr, pwm));
    TEST_ASSERT_FALSE(parseActuatorCalibrationPwm("", pwm));
    TEST_ASSERT_EQUAL_INT(5, pwm);
    TEST_ASSERT_TRUE(parseActuatorCalibrationPwm("110", pwm));
    TEST_ASSERT_EQUAL_INT(110, pwm);
}

// The calibration PWM floor is 110 (owner decision F4): 110 is accepted, 109 is
// not, and both ends are the constants, not copies of them.
static void test_pwm_minimum_is_110()
{
    TEST_ASSERT_EQUAL_INT(110, CAL_PWM_MIN);
    TEST_ASSERT_EQUAL_INT(150, CAL_PWM_MAX);

    int pwm = 5;
    TEST_ASSERT_TRUE(parseActuatorCalibrationPwm("110", pwm));
    TEST_ASSERT_EQUAL_INT(110, pwm);
    pwm = 5;
    TEST_ASSERT_FALSE(parseActuatorCalibrationPwm("109", pwm));
    TEST_ASSERT_EQUAL_INT(5, pwm);

    assertRun({"110"}, CAL_PASS_ALL, 110);
    assertRun({"SWEEP", "110"}, CAL_PASS_SWEEP, 110);
    assertKind({"109"}, ActuatorCalibrationCommandKind::Invalid);
    assertKind({"SEEK", "109"}, ActuatorCalibrationCommandKind::Invalid);
}

static void test_power_sensor_commands_are_left_to_the_power_parser()
{
    assertKind({"PWR_SENSE"}, ActuatorCalibrationCommandKind::PowerSensor);
    assertKind({"pwr_sense", "SHOW"}, ActuatorCalibrationCommandKind::PowerSensor);
    assertKind({"PWR_SENSE", "VOLTAGE", "26.5", "13.2"}, ActuatorCalibrationCommandKind::PowerSensor);
}

static void test_usage_text_names_the_bounds()
{
    FakeOutput out;
    printActuatorCalibrationUsage(out);
    TEST_ASSERT_EQUAL_STRING("usage: C [SEEK|SWEEP] [pwm], pwm 110-150", out.text.c_str());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_bare_and_explicit_forms_run);
    RUN_TEST(test_bad_pwm_and_extra_tokens_are_invalid);
    RUN_TEST(test_pwm_minimum_is_110);
    RUN_TEST(test_power_sensor_commands_are_left_to_the_power_parser);
    RUN_TEST(test_usage_text_names_the_bounds);
    return UNITY_END();
}

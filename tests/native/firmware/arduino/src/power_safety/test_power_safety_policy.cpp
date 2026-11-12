#include <type_traits>

#include "unity.h"

#include "src/power_monitor/voltage_validity.h"
#include "src/power_safety/power_safety_policy.h"
#include "src/telemetry.h"

static_assert(!std::is_convertible<float, Volts>::value,
              "raw floats must not implicitly become volts");

void setUp() {}
void tearDown() {}

static PowerSafetyEvaluation evaluate(
    PowerSafetyState state,
    PackVoltageRegion region,
    uint16_t softTicks = 0,
    bool isValid = true)
{
    return evaluatePowerSafetyAction(
        state, region, isValid, softTicks);
}

static PowerSafetyEvaluation repeat(
    PowerSafetyState state, PackVoltageRegion region, int count)
{
    PowerSafetyEvaluation result = {PowerSafetyAction::None, 0};
    for (int tick = 0; tick < count; ++tick)
        result = evaluate(state, region, result.belowSoftTicks);
    return result;
}

static void test_strict_thresholds_leave_exact_values_in_the_lower_region()
{
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::HardCut,
        classifyPackVoltage(PACK_HARD_CUT_THRESHOLD));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::SoftCut,
        classifyPackVoltage(Volts(PACK_HARD_CUT_THRESHOLD.value() + 0.01f)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::SoftCut,
        classifyPackVoltage(PACK_SOFT_CUT_THRESHOLD));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Warn,
        classifyPackVoltage(PACK_WARNING_THRESHOLD));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Normal,
        classifyPackVoltage(PACK_RECOVERY_THRESHOLD));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::AboveRecovery,
        classifyPackVoltage(Volts(PACK_RECOVERY_THRESHOLD.value() + 0.01f)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::AboveRecovery,
        classifyPackVoltage(PACK_OVER_VOLT_THRESHOLD));
}

static void test_crossing_each_threshold_enters_the_upper_region()
{
    const float delta = 0.01f;
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::SoftCut,
        classifyPackVoltage(Volts(PACK_HARD_CUT_THRESHOLD.value() + delta)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Warn,
        classifyPackVoltage(Volts(PACK_SOFT_CUT_THRESHOLD.value() + delta)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Normal,
        classifyPackVoltage(Volts(PACK_WARNING_THRESHOLD.value() + delta)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::AboveRecovery,
        classifyPackVoltage(Volts(PACK_RECOVERY_THRESHOLD.value() + delta)));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::OverVolt,
        classifyPackVoltage(Volts(PACK_OVER_VOLT_THRESHOLD.value() + delta)));
}

static void test_soft_cut_is_debounced_and_hard_cut_is_immediate()
{
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        repeat(
            PowerSafetyState::Operating,
            PackVoltageRegion::SoftCut,
            POWER_CUT_DEBOUNCE_TICKS - 1).action);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginGracefulShutdown,
        repeat(
            PowerSafetyState::Operating,
            PackVoltageRegion::SoftCut,
            POWER_CUT_DEBOUNCE_TICKS).action);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginHardShutdown,
        evaluate(
            PowerSafetyState::Operating,
            PackVoltageRegion::HardCut).action);
}

static void test_hard_cut_preempts_soft_cut_debounce()
{
    PowerSafetyEvaluation result = repeat(
        PowerSafetyState::Operating,
        PackVoltageRegion::SoftCut,
        POWER_CUT_DEBOUNCE_TICKS - 1);
    result = evaluate(
        PowerSafetyState::Operating,
        PackVoltageRegion::HardCut,
        result.belowSoftTicks);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginHardShutdown, result.action);
    TEST_ASSERT_EQUAL_UINT16(0, result.belowSoftTicks);

    result = evaluate(
        PowerSafetyState::Operating,
        PackVoltageRegion::Warn,
        result.belowSoftTicks);
    TEST_ASSERT_EQUAL_UINT16(0, result.belowSoftTicks);
}

static void test_powering_down_is_not_cancelled_but_can_escalate()
{
    const PackVoltageRegion safeRegions[] = {
        PackVoltageRegion::Normal,
        PackVoltageRegion::Warn,
        PackVoltageRegion::SoftCut,
        PackVoltageRegion::AboveRecovery,
    };
    for (const PackVoltageRegion region : safeRegions)
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyAction::None,
            evaluate(PowerSafetyState::PoweringDown, region).action);

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginHardShutdown,
        evaluate(
            PowerSafetyState::PoweringDown,
            PackVoltageRegion::HardCut).action);
}

static void test_sleep_resumes_only_above_the_recovery_threshold()
{
    const PackVoltageRegion heldRegions[] = {
        PackVoltageRegion::Normal,
        PackVoltageRegion::Warn,
        PackVoltageRegion::SoftCut,
        PackVoltageRegion::HardCut,
    };
    for (const PackVoltageRegion region : heldRegions)
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyAction::None,
            evaluate(PowerSafetyState::Sleep, region).action);

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        evaluate(
            PowerSafetyState::Sleep,
            PackVoltageRegion::AboveRecovery).action);
}

static void test_resuming_completes_or_reenters_safety_actions()
{
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CompleteResume,
        evaluate(
            PowerSafetyState::Resuming,
            PackVoltageRegion::Normal).action);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginGracefulShutdown,
        repeat(
            PowerSafetyState::Resuming,
            PackVoltageRegion::SoftCut,
            POWER_CUT_DEBOUNCE_TICKS).action);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginHardShutdown,
        evaluate(
            PowerSafetyState::Resuming,
            PackVoltageRegion::HardCut).action);
}

static void test_over_voltage_is_immediate_except_after_lockout()
{
    const PowerSafetyState states[] = {
        PowerSafetyState::Operating,
        PowerSafetyState::PoweringDown,
        PowerSafetyState::Sleep,
        PowerSafetyState::Resuming,
    };
    for (const PowerSafetyState state : states)
        TEST_ASSERT_EQUAL_UINT8(
            PowerSafetyAction::BeginOverVoltageShutdown,
            evaluate(state, PackVoltageRegion::OverVolt).action);

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        evaluate(
            PowerSafetyState::OverVoltageLockout,
            PackVoltageRegion::AboveRecovery).action);
}

static void test_invalid_samples_clear_debounce_without_an_action()
{
    const PowerSafetyEvaluation result = evaluate(
        PowerSafetyState::PoweringDown,
        PackVoltageRegion::HardCut,
        3,
        false);
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyAction::None, result.action);
    TEST_ASSERT_EQUAL_UINT16(0, result.belowSoftTicks);
}

static void test_cutoff_policy_tracks_power_ack_and_rollover()
{
    TEST_ASSERT_EQUAL_UINT32(ORIN_FORCE_OFF_MS, orinForceOffDelay(false));
    TEST_ASSERT_EQUAL_UINT32(ORIN_ACKED_OFF_MS, orinForceOffDelay(true));

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        evaluateOrinCutoffAction(UINT32_MAX, 0, false, true, false));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        evaluateOrinCutoffAction(UINT32_MAX, 0, true, false, false));

    const uint32_t armedAt = 100;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        evaluateOrinCutoffAction(
            armedAt + ORIN_FORCE_OFF_MS - 1,
            armedAt,
            true,
            true,
            false));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        evaluateOrinCutoffAction(
            armedAt + ORIN_FORCE_OFF_MS,
            armedAt,
            true,
            true,
            false));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        evaluateOrinCutoffAction(
            armedAt + ORIN_ACKED_OFF_MS,
            armedAt,
            true,
            true,
            true));

    const uint32_t rolloverArmedAt = UINT32_MAX - 100;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        evaluateOrinCutoffAction(
            rolloverArmedAt + ORIN_ACKED_OFF_MS,
            rolloverArmedAt,
            true,
            true,
            true));
}

static void test_action_resolution_preserves_safety_priority()
{
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginOverVoltageShutdown,
        resolvePowerSafetyActions(
            PowerSafetyAction::BeginOverVoltageShutdown,
            PowerSafetyAction::CutOrinPower));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        resolvePowerSafetyActions(
            PowerSafetyAction::BeginResume,
            PowerSafetyAction::CutOrinPower));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        resolvePowerSafetyActions(
            PowerSafetyAction::BeginHardShutdown,
            PowerSafetyAction::CutOrinPower));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginGracefulShutdown,
        resolvePowerSafetyActions(
            PowerSafetyAction::BeginGracefulShutdown,
            PowerSafetyAction::None));
}

static void test_enum_codes_are_stable()
{
    TEST_ASSERT_EQUAL_UINT8(0, PackVoltageRegion::Normal);
    TEST_ASSERT_EQUAL_UINT8(1, PackVoltageRegion::Warn);
    TEST_ASSERT_EQUAL_UINT8(2, PackVoltageRegion::SoftCut);
    TEST_ASSERT_EQUAL_UINT8(3, PackVoltageRegion::HardCut);
    TEST_ASSERT_EQUAL_UINT8(4, PackVoltageRegion::OverVolt);
    TEST_ASSERT_EQUAL_UINT8(5, PackVoltageRegion::AboveRecovery);
    TEST_ASSERT_EQUAL_UINT8(0, PowerSafetyState::Operating);
    TEST_ASSERT_EQUAL_UINT8(1, PowerSafetyState::PoweringDown);
    TEST_ASSERT_EQUAL_UINT8(2, PowerSafetyState::Sleep);
    TEST_ASSERT_EQUAL_UINT8(3, PowerSafetyState::Resuming);
    TEST_ASSERT_EQUAL_UINT8(4, PowerSafetyState::OverVoltageLockout);
}

static void test_power_policy_matches_the_bench_contract()
{
    TEST_ASSERT_EQUAL_FLOAT(24.8f, PACK_WARNING_THRESHOLD.value());
    TEST_ASSERT_EQUAL_FLOAT(24.0f, PACK_SOFT_CUT_THRESHOLD.value());
    TEST_ASSERT_EQUAL_FLOAT(22.4f, PACK_HARD_CUT_THRESHOLD.value());
    TEST_ASSERT_EQUAL_FLOAT(29.6f, PACK_OVER_VOLT_THRESHOLD.value());
    TEST_ASSERT_EQUAL_FLOAT(26.4f, PACK_RECOVERY_THRESHOLD.value());
    TEST_ASSERT_EQUAL_UINT8(4, POWER_CUT_DEBOUNCE_TICKS);
    TEST_ASSERT_GREATER_OR_EQUAL_UINT32(
        100, TELEMETRY_INTERVAL_MS * POWER_CUT_DEBOUNCE_TICKS);
    TEST_ASSERT_LESS_OR_EQUAL_UINT32(
        1000, TELEMETRY_INTERVAL_MS * POWER_CUT_DEBOUNCE_TICKS);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_strict_thresholds_leave_exact_values_in_the_lower_region);
    RUN_TEST(test_crossing_each_threshold_enters_the_upper_region);
    RUN_TEST(test_soft_cut_is_debounced_and_hard_cut_is_immediate);
    RUN_TEST(test_hard_cut_preempts_soft_cut_debounce);
    RUN_TEST(test_powering_down_is_not_cancelled_but_can_escalate);
    RUN_TEST(test_sleep_resumes_only_above_the_recovery_threshold);
    RUN_TEST(test_resuming_completes_or_reenters_safety_actions);
    RUN_TEST(test_over_voltage_is_immediate_except_after_lockout);
    RUN_TEST(test_invalid_samples_clear_debounce_without_an_action);
    RUN_TEST(test_cutoff_policy_tracks_power_ack_and_rollover);
    RUN_TEST(test_action_resolution_preserves_safety_priority);
    RUN_TEST(test_enum_codes_are_stable);
    RUN_TEST(test_power_policy_matches_the_bench_contract);
    return UNITY_END();
}

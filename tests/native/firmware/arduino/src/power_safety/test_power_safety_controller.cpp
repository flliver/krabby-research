#include "unity.h"

#include "src/power_safety/power_safety_controller.h"

struct FakeStorage
{
    PowerSafetyLatchRecord record = {0xFF, 0xFF, 0xFF};
    uint16_t writes = 0;

    void load(PowerSafetyLatchRecord &destination)
    {
        destination = record;
    }

    void writeRecord(const PowerSafetyLatchRecord &source)
    {
        record = source;
        ++writes;
    }

    void updateMagic(uint8_t magic)
    {
        record.magic = magic;
        ++writes;
    }
};

static FakeStorage storage;

static PowerSafetyLatchRecord latchRecord(PowerSafetyLatchValue value)
{
    return {
        EEPROM_POWER_SAFETY_MAGIC,
        EEPROM_POWER_SAFETY_SCHEMA,
        static_cast<uint8_t>(value),
    };
}

void setUp()
{
    storage = FakeStorage{};
}
void tearDown() {}

static void test_controller_records_only_valid_regions()
{
    PowerSafetyController controller;
    controller.determinePowerSafetyAction(
        Volts(24.5f), true, 0);
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Warn, controller.lastPackVoltageRegion());

    controller.determinePowerSafetyAction(
        Volts(0.0f), false, 0);
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::Warn, controller.lastPackVoltageRegion());
    TEST_ASSERT_EQUAL_UINT16(0, controller.belowSoftTicks());
}

static void test_lifecycle_operations_are_guarded()
{
    PowerSafetyController controller;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Rejected,
        controller.completeShutdown(storage, 0));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Rejected,
        controller.completeResume(storage));

    controller.beginPoweringDown(storage);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyState::PoweringDown, controller.state());
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.completeShutdown(storage, 0));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Sleep, controller.state());

    controller.beginResuming(storage);
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Resuming, controller.state());
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.completeResume(storage));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Operating, controller.state());
}

static void test_controller_owns_state_to_latch_mapping()
{
    PowerSafetyController controller;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.beginPoweringDown(storage));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchValue::Sleep, storage.record.value);
    const uint16_t lowVoltageWrites = storage.writes;

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.completeShutdown(storage, 0));
    TEST_ASSERT_EQUAL_UINT16(lowVoltageWrites, storage.writes);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.beginResuming(storage));
    TEST_ASSERT_EQUAL_UINT16(lowVoltageWrites, storage.writes);

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        controller.completeResume(storage));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchValue::None, storage.record.value);

    PowerSafetyController overVoltage;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Applied,
        overVoltage.enterOverVoltageLockout(storage));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchValue::Sleep, storage.record.value);
}

static void test_rejected_transition_does_not_write()
{
    PowerSafetyController controller;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Rejected,
        controller.completeResume(storage));
    TEST_ASSERT_EQUAL_UINT16(0, storage.writes);
}

static void test_restore_maps_owned_latch_to_stable_state()
{
    PowerSafetyController empty;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchLoadResult::Empty,
        empty.restorePowerSafetyState(storage, 100));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Operating, empty.state());
    TEST_ASSERT_TRUE(empty.isOrinPowerCommandedOn());

    storage = FakeStorage{};
    storage.record = latchRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyController sleeping;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchLoadResult::Loaded,
        sleeping.restorePowerSafetyState(storage, 100));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Sleep, sleeping.state());
    TEST_ASSERT_FALSE(sleeping.isOrinPowerCommandedOn());

    storage = FakeStorage{};
    storage.record = latchRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyController locked;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchLoadResult::Loaded,
        locked.restorePowerSafetyState(storage, 100));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyState::Sleep, locked.state());
    TEST_ASSERT_FALSE(locked.isOrinPowerCommandedOn());
}

static void test_corrupt_restore_repairs_to_fail_safe_sleep()
{
    storage.record = {
        EEPROM_POWER_SAFETY_MAGIC,
        EEPROM_POWER_SAFETY_SCHEMA,
        99,
    };
    PowerSafetyController controller;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchLoadResult::Corrupt,
        controller.restorePowerSafetyState(storage, 0));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Sleep, controller.state());
    TEST_ASSERT_FALSE(controller.isOrinPowerCommandedOn());
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchValue::Sleep, storage.record.value);
}

static void test_reset_reconciliation_uses_one_recovery_path()
{
    storage.record = latchRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyController sleeping;
    sleeping.restorePowerSafetyState(storage, 0);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        sleeping.reconcilePowerSafetyAfterReset(
            Volts(26.5f), true, 1));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Sleep, sleeping.state());

    storage = FakeStorage{};
    storage.record = latchRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyController unsafeReset;
    unsafeReset.restorePowerSafetyState(storage, 0);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginOverVoltageShutdown,
        unsafeReset.reconcilePowerSafetyAfterReset(
            Volts(29.7f), true, 1));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyState::Sleep,
        unsafeReset.state());

    storage = FakeStorage{};
    storage.record = latchRecord(PowerSafetyLatchValue::Sleep);
    PowerSafetyController safeReset;
    safeReset.restorePowerSafetyState(storage, 0);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        safeReset.reconcilePowerSafetyAfterReset(
            Volts(26.5f), true, 1));
    TEST_ASSERT_EQUAL_UINT8(PowerSafetyState::Sleep, safeReset.state());
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyLatchValue::Sleep, storage.record.value);
}

static void test_over_voltage_lockout_cannot_be_cleared()
{
    PowerSafetyController controller;
    controller.enterOverVoltageLockout(storage);
    controller.enterSleep(storage, 0);
    controller.beginResuming(storage);
    controller.beginPoweringDown(storage);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Rejected,
        controller.completeResume(storage));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyTransitionResult::Rejected,
        controller.completeShutdown(storage, 0));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyState::OverVoltageLockout, controller.state());
}

static void test_shutdown_ack_requires_a_live_graceful_transaction()
{
    PowerSafetyController controller;
    controller.armOrinCutoff(100);
    TEST_ASSERT_FALSE(controller.acceptShutdownAckIfExpected(1));

    controller.beginPoweringDown(storage);
    controller.markOrinPowerCut();
    TEST_ASSERT_FALSE(controller.acceptShutdownAckIfExpected(2));
    controller.markOrinPowerRestored();
    TEST_ASSERT_FALSE(controller.acceptShutdownAckIfExpected(3));
    controller.armOrinCutoff(100);
    TEST_ASSERT_TRUE(controller.acceptShutdownAckIfExpected(4));
    TEST_ASSERT_FALSE(controller.acceptShutdownAckIfExpected(5));
}

static void test_park_gate_tracks_lifecycle_state()
{
    PowerSafetyController controller;
    TEST_ASSERT_FALSE(controller.areActuatorsParked());
    controller.beginPoweringDown(storage);
    TEST_ASSERT_TRUE(controller.areActuatorsParked());
    controller.enterSleep(storage, 0);
    TEST_ASSERT_TRUE(controller.areActuatorsParked());
    controller.beginResuming(storage);
    TEST_ASSERT_FALSE(controller.areActuatorsParked());
    controller.enterOverVoltageLockout(storage);
    TEST_ASSERT_TRUE(controller.areActuatorsParked());
}

static void test_low_power_predicates_track_lifecycle_state()
{
    PowerSafetyController controller;
    TEST_ASSERT_FALSE(controller.isNormalOperationSuspended());
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(UINT32_MAX));

    controller.beginPoweringDown(storage);
    TEST_ASSERT_FALSE(controller.isNormalOperationSuspended());
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(UINT32_MAX));

    controller.enterSleep(storage, 0);
    TEST_ASSERT_TRUE(controller.isNormalOperationSuspended());
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(
        POWER_RECOVERY_POLL_INTERVAL_MS - 1));
    TEST_ASSERT_TRUE(controller.shouldCheckForRecovery(
        POWER_RECOVERY_POLL_INTERVAL_MS));

    controller.beginResuming(storage);
    TEST_ASSERT_FALSE(controller.isNormalOperationSuspended());
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(UINT32_MAX));

    controller.enterOverVoltageLockout(storage);
    TEST_ASSERT_TRUE(controller.isNormalOperationSuspended());
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(UINT32_MAX));
}

static void test_recovery_sample_restarts_poll_interval()
{
    const uint32_t sleepAt = 100;
    const uint32_t firstPollAt =
        sleepAt + POWER_RECOVERY_POLL_INTERVAL_MS;
    PowerSafetyController controller;
    controller.enterSleep(storage, sleepAt);

    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(firstPollAt - 1));
    TEST_ASSERT_TRUE(controller.shouldCheckForRecovery(firstPollAt));

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(
            Volts(0.0f), false, firstPollAt));
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(
        firstPollAt + POWER_RECOVERY_POLL_INTERVAL_MS - 1));
    TEST_ASSERT_TRUE(controller.shouldCheckForRecovery(
        firstPollAt + POWER_RECOVERY_POLL_INTERVAL_MS));

    const uint32_t secondPollAt =
        firstPollAt + POWER_RECOVERY_POLL_INTERVAL_MS;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(
            Volts(24.5f), true, secondPollAt));
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(
        secondPollAt + POWER_RECOVERY_POLL_INTERVAL_MS - 1));
}

static void test_recovery_poll_interval_handles_rollover()
{
    const uint32_t sleepAt = UINT32_MAX - 100;
    const uint32_t pollAt =
        sleepAt + POWER_RECOVERY_POLL_INTERVAL_MS;
    PowerSafetyController controller;
    controller.enterSleep(storage, sleepAt);

    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(pollAt - 1));
    TEST_ASSERT_TRUE(controller.shouldCheckForRecovery(pollAt));
}

static void test_recovered_sample_stops_recovery_checks_when_resuming()
{
    PowerSafetyController controller;
    controller.enterSleep(storage, 0);
    const uint32_t pollAt = POWER_RECOVERY_POLL_INTERVAL_MS;

    TEST_ASSERT_TRUE(controller.shouldCheckForRecovery(pollAt));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        controller.determinePowerSafetyAction(
            Volts(26.5f), true, pollAt));
    controller.beginResuming(storage);
    TEST_ASSERT_FALSE(controller.shouldCheckForRecovery(UINT32_MAX));
}

static void test_controller_records_the_latest_pack_sample()
{
    PowerSafetyController controller;
    controller.determinePowerSafetyAction(Volts(26.5f), true, 0);

    TEST_ASSERT_EQUAL_FLOAT(26.5f, controller.lastPackVoltage().value());
    TEST_ASSERT_TRUE(controller.isLastPackReadingValid());

    controller.determinePowerSafetyAction(Volts(0.0f), false, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, controller.lastPackVoltage().value());
    TEST_ASSERT_FALSE(controller.isLastPackReadingValid());
}

static void test_cutoff_deadline_tracks_ack_and_power_state()
{
    PowerSafetyController controller;
    controller.beginPoweringDown(storage);
    controller.armOrinCutoffIfNeeded(100);
    controller.armOrinCutoffIfNeeded(200);

    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(ORIN_FORCE_OFF_MS + 99));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        controller.determinePowerSafetyAction(ORIN_FORCE_OFF_MS + 100));

    controller.armOrinCutoff(100);
    const uint32_t acknowledgedAt = 20100;
    TEST_ASSERT_TRUE(controller.acceptShutdownAckIfExpected(acknowledgedAt));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(
            acknowledgedAt + ORIN_ACKED_OFF_MS - 1));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        controller.determinePowerSafetyAction(
            acknowledgedAt + ORIN_ACKED_OFF_MS));

    controller.markOrinPowerCut();
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(UINT32_MAX));

    PowerSafetyController rollover;
    rollover.beginPoweringDown(storage);
    rollover.armOrinCutoff(UINT32_MAX - 200);
    const uint32_t rolloverAck = UINT32_MAX - 100;
    TEST_ASSERT_TRUE(rollover.acceptShutdownAckIfExpected(rolloverAck));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        rollover.determinePowerSafetyAction(
            rolloverAck + ORIN_ACKED_OFF_MS - 1));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        rollover.determinePowerSafetyAction(
            rolloverAck + ORIN_ACKED_OFF_MS));
}

static void test_voltage_actions_and_cutoff_deadline_resolve_once()
{
    const uint32_t armedAt = 100;
    const uint32_t dueAt = armedAt + ORIN_FORCE_OFF_MS;

    PowerSafetyController invalid;
    invalid.beginPoweringDown(storage);
    invalid.armOrinCutoff(armedAt);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        invalid.determinePowerSafetyAction(Volts(0.0f), false, dueAt));

    PowerSafetyController hardCut;
    hardCut.beginPoweringDown(storage);
    hardCut.armOrinCutoff(armedAt);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        hardCut.determinePowerSafetyAction(Volts(22.0f), true, dueAt));
    TEST_ASSERT_EQUAL_UINT8(
        PackVoltageRegion::HardCut, hardCut.lastPackVoltageRegion());

    PowerSafetyController recovery;
    recovery.enterSleep(storage, 0);
    recovery.armOrinCutoff(armedAt);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginResume,
        recovery.determinePowerSafetyAction(Volts(26.5f), true, dueAt));

    PowerSafetyController overVoltage;
    overVoltage.beginPoweringDown(storage);
    overVoltage.armOrinCutoff(armedAt);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginOverVoltageShutdown,
        overVoltage.determinePowerSafetyAction(Volts(29.61f), true, dueAt));
}

static void test_direct_hard_cut_keeps_its_cutoff_deadline()
{
    const uint32_t hardCutAt = 100;
    PowerSafetyController controller;
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::BeginHardShutdown,
        controller.determinePowerSafetyAction(
            Volts(22.0f), true, hardCutAt));

    controller.armOrinCutoffIfNeeded(hardCutAt);
    controller.enterSleep(storage, hardCutAt);
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::None,
        controller.determinePowerSafetyAction(
            hardCutAt + ORIN_FORCE_OFF_MS - 1));
    TEST_ASSERT_EQUAL_UINT8(
        PowerSafetyAction::CutOrinPower,
        controller.determinePowerSafetyAction(
            hardCutAt + ORIN_FORCE_OFF_MS));
}

static void test_controller_tracks_status_led_state()
{
    PowerSafetyController controller;
    controller.beginLowPowerIndicators();
    TEST_ASSERT_TRUE(controller.isStatusLedOn());

    controller.toggleStatusLed();
    TEST_ASSERT_FALSE(controller.isStatusLedOn());

    controller.clearStatusLed();
    TEST_ASSERT_FALSE(controller.isStatusLedOn());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_controller_records_only_valid_regions);
    RUN_TEST(test_lifecycle_operations_are_guarded);
    RUN_TEST(test_controller_owns_state_to_latch_mapping);
    RUN_TEST(test_rejected_transition_does_not_write);
    RUN_TEST(test_restore_maps_owned_latch_to_stable_state);
    RUN_TEST(test_corrupt_restore_repairs_to_fail_safe_sleep);
    RUN_TEST(test_reset_reconciliation_uses_one_recovery_path);
    RUN_TEST(test_over_voltage_lockout_cannot_be_cleared);
    RUN_TEST(test_shutdown_ack_requires_a_live_graceful_transaction);
    RUN_TEST(test_park_gate_tracks_lifecycle_state);
    RUN_TEST(test_low_power_predicates_track_lifecycle_state);
    RUN_TEST(test_recovery_sample_restarts_poll_interval);
    RUN_TEST(test_recovery_poll_interval_handles_rollover);
    RUN_TEST(test_recovered_sample_stops_recovery_checks_when_resuming);
    RUN_TEST(test_controller_records_the_latest_pack_sample);
    RUN_TEST(test_cutoff_deadline_tracks_ack_and_power_state);
    RUN_TEST(test_voltage_actions_and_cutoff_deadline_resolve_once);
    RUN_TEST(test_direct_hard_cut_keeps_its_cutoff_deadline);
    RUN_TEST(test_controller_tracks_status_led_state);
    return UNITY_END();
}

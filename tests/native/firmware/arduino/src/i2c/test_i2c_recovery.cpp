// Retry timing, and the clocked sequence that frees a held SDA line.
//
// Both are the kind of thing that is wrong by one and looks fine: a counter
// that never resets turns "consecutive" into "cumulative", and a clock loop
// that checks SDA at the wrong point either gives up a pulse early or spends
// nine when one would do.

#include <stdint.h>

#include "src/i2c/i2c_recovery.h"

#include "unity.h"

namespace
{

const I2cRecoveryLimits LIMITS = {3, 2000};

// SDA is held low until `releaseAfter` full clock pulses have been issued.
// releaseAfter == 0 means the line was never held.
class FakeBus
{
public:
    explicit FakeBus(uint8_t releaseAfter)
        : releaseAfter_(releaseAfter), pulses_(0), halfBits_(0), stops_(0),
          restarts_(0), isSclLow_(false)
    {
    }

    bool isSdaHigh() const { return pulses_ >= releaseAfter_; }

    void sclLow()
    {
        TEST_ASSERT_FALSE_MESSAGE(isSclLow_, "SCL driven low twice without release");
        isSclLow_ = true;
    }

    void sclRelease()
    {
        TEST_ASSERT_TRUE_MESSAGE(isSclLow_, "SCL released without being driven low");
        isSclLow_ = false;
        ++pulses_;
    }

    void halfBit() { ++halfBits_; }
    void sendStop() { ++stops_; }
    void restart() { ++restarts_; }

    uint8_t pulses() const { return pulses_; }
    int halfBits() const { return halfBits_; }
    int stops() const { return stops_; }
    int restarts() const { return restarts_; }
    void releaseSda() { releaseAfter_ = pulses_; }

private:
    uint8_t releaseAfter_;
    uint8_t pulses_;
    int halfBits_;
    int stops_;
    int restarts_;
    bool isSclLow_;
};

}  // namespace

void setUp() {}
void tearDown() {}

// ---- policy ----

void test_a_single_failure_is_not_evidence(void)
{
    I2cRecoveryPolicy policy;

    TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(1000, LIMITS));
    TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(1050, LIMITS));
    TEST_ASSERT_TRUE(policy.shouldAttemptRecovery(1100, LIMITS));
}

void test_a_success_clears_pending_failures(void)
{
    I2cRecoveryPolicy policy;

    policy.shouldAttemptRecovery(1000, LIMITS);
    policy.shouldAttemptRecovery(1050, LIMITS);
    // Without this, "consecutive" would silently mean "cumulative" and a device
    // with occasional transients would eventually re-initialise for no reason.
    policy.noteSuccess();
    TEST_ASSERT_EQUAL_UINT8(0, policy.badTicks());

    TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(1100, LIMITS));
    TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(1150, LIMITS));
    TEST_ASSERT_TRUE(policy.shouldAttemptRecovery(1200, LIMITS));
}

void test_attempts_are_rate_limited(void)
{
    I2cRecoveryPolicy policy;
    uint32_t now = 1000;

    for (int i = 0; i < 3; ++i)
        policy.shouldAttemptRecovery(now, LIMITS);          // fires at the third

    // Still failing, but inside the interval: no second attempt.
    for (int i = 0; i < 20; ++i)
        TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(now + 100 * i, LIMITS));

    // Once the interval is up the next failure fires at once: the device has
    // been failing continuously, so there is nothing to re-accumulate.
    now += LIMITS.retryIntervalMs;
    TEST_ASSERT_TRUE(policy.shouldAttemptRecovery(now, LIMITS));
}

void test_the_first_attempt_is_not_blocked_at_boot(void)
{
    // millis() == 0 must not read as "attempted at 0", which would hold the
    // first retry off for a whole interval after every power-on.
    I2cRecoveryPolicy policy;

    policy.shouldAttemptRecovery(0, LIMITS);
    policy.shouldAttemptRecovery(0, LIMITS);
    TEST_ASSERT_TRUE(policy.shouldAttemptRecovery(0, LIMITS));
}

void test_the_interval_survives_the_millis_rollover(void)
{
    I2cRecoveryPolicy policy;
    const uint32_t beforeRollover = 0xFFFFFF00UL;

    for (int i = 0; i < 3; ++i)
        policy.shouldAttemptRecovery(beforeRollover, LIMITS);

    // 0x100 ms elapsed across the wrap. Comparing timestamps directly would see
    // a huge jump backwards and let the attempt through early.
    const uint32_t afterRollover = 0x00000000UL;
    for (int i = 0; i < 3; ++i)
        TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(afterRollover, LIMITS));

    // 0x100 + 2000 ms elapsed: past the interval, so this one fires.
    TEST_ASSERT_TRUE(
        policy.shouldAttemptRecovery(afterRollover + LIMITS.retryIntervalMs, LIMITS));
}

void test_the_bad_tick_counter_saturates(void)
{
    I2cRecoveryPolicy policy;
    // One attempt, then an interval long enough that every later failure is
    // blocked -- which is the only way the counter climbs without being reset.
    const I2cRecoveryLimits slow = {1, 1000000};

    TEST_ASSERT_TRUE(policy.shouldAttemptRecovery(0, slow));
    for (int i = 0; i < 600; ++i)
        TEST_ASSERT_FALSE(policy.shouldAttemptRecovery(1, slow));

    // Wrapping to 0 would read as a healthy device.
    TEST_ASSERT_EQUAL_UINT8(255, policy.badTicks());
}

// ---- bus ----

void test_a_free_bus_is_restarted_without_being_clocked(void)
{
    FakeBus bus(0);

    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::NotNeeded),
        static_cast<int>(recoverI2cBus(bus)));
    // Clocking a healthy bus would corrupt a transfer that was merely slow.
    TEST_ASSERT_EQUAL_UINT8(0, bus.pulses());
    TEST_ASSERT_EQUAL_INT(0, bus.stops());
    TEST_ASSERT_EQUAL_INT(1, bus.restarts());
}

void test_a_held_line_is_clocked_free_and_stopped(void)
{
    FakeBus bus(4);

    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::Cleared),
        static_cast<int>(recoverI2cBus(bus)));
    // Stops as soon as SDA comes back, rather than always spending nine.
    TEST_ASSERT_EQUAL_UINT8(4, bus.pulses());
    TEST_ASSERT_EQUAL_INT(2 * 4, bus.halfBits());
    TEST_ASSERT_EQUAL_INT(1, bus.stops());
    TEST_ASSERT_EQUAL_INT(1, bus.restarts());
}

void test_a_slave_holding_through_a_whole_byte_is_still_freed(void)
{
    // Eight data bits plus the ACK is the worst case a slave can hold, and it
    // is the reason the limit is nine rather than eight.
    FakeBus bus(I2C_BUS_CLEAR_PULSES);

    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::Cleared),
        static_cast<int>(recoverI2cBus(bus)));
    TEST_ASSERT_EQUAL_UINT8(I2C_BUS_CLEAR_PULSES, bus.pulses());
    TEST_ASSERT_EQUAL_INT(1, bus.stops());
}

void test_a_line_that_never_releases_is_reported_stuck(void)
{
    FakeBus bus(200);

    // Reported rather than retried: clocking cannot fix a shorted line, and the
    // caller needs to stop instead of looping on it forever.
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::Stuck),
        static_cast<int>(recoverI2cBus(bus)));
    TEST_ASSERT_EQUAL_UINT8(I2C_BUS_CLEAR_PULSES, bus.pulses());
    TEST_ASSERT_EQUAL_INT(0, bus.stops());
    TEST_ASSERT_EQUAL_INT(0, bus.restarts());
}

void test_a_stuck_bus_is_clocked_once_then_polled_until_sda_releases(void)
{
    FakeBus bus(200);
    I2cStuckBusLatch latch;

    TEST_ASSERT_TRUE(latch.mayAttempt(bus.isSdaHigh()));
    const I2cBusRecovery first = recoverI2cBus(bus);
    latch.noteResult(first);
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::Stuck), static_cast<int>(first));
    TEST_ASSERT_TRUE(latch.isWaitingForRelease());
    TEST_ASSERT_EQUAL_UINT8(I2C_BUS_CLEAR_PULSES, bus.pulses());

    // Poll without issuing more clocks.
    for (int retry = 0; retry < 20; ++retry)
        TEST_ASSERT_FALSE(latch.mayAttempt(bus.isSdaHigh()));
    TEST_ASSERT_EQUAL_UINT8(I2C_BUS_CLEAR_PULSES, bus.pulses());

    // Resume recovery after SDA is released.
    bus.releaseSda();
    TEST_ASSERT_TRUE(latch.mayAttempt(bus.isSdaHigh()));
    const I2cBusRecovery afterRelease = recoverI2cBus(bus);
    latch.noteResult(afterRelease);
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(I2cBusRecovery::NotNeeded),
        static_cast<int>(afterRelease));
    TEST_ASSERT_FALSE(latch.isWaitingForRelease());
    TEST_ASSERT_EQUAL_UINT8(I2C_BUS_CLEAR_PULSES, bus.pulses());
    TEST_ASSERT_EQUAL_INT(1, bus.restarts());
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_a_single_failure_is_not_evidence);
    RUN_TEST(test_a_success_clears_pending_failures);
    RUN_TEST(test_attempts_are_rate_limited);
    RUN_TEST(test_the_first_attempt_is_not_blocked_at_boot);
    RUN_TEST(test_the_interval_survives_the_millis_rollover);
    RUN_TEST(test_the_bad_tick_counter_saturates);
    RUN_TEST(test_a_free_bus_is_restarted_without_being_clocked);
    RUN_TEST(test_a_held_line_is_clocked_free_and_stopped);
    RUN_TEST(test_a_slave_holding_through_a_whole_byte_is_still_freed);
    RUN_TEST(test_a_line_that_never_releases_is_reported_stuck);
    RUN_TEST(test_a_stuck_bus_is_clocked_once_then_polled_until_sda_releases);
    return UNITY_END();
}

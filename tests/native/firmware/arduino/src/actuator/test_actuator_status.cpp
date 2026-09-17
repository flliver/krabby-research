#include "src/actuator/actuator_status.h"
#include "src/actuator/actuator_pot_tracker.h"
#include "unity.h"

void setUp() {}
void tearDown() {}

static void test_default_status_is_unassigned_unknown_and_idle()
{
    const ActuatorStatus status;

    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorId::ActuatorCount, (int)status.actuatorId);
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Unknown,
        (int)status.connectionState);
    TEST_ASSERT_EQUAL_INT16(0, status.commandedPwm);
}

static void test_connection_state_wire_values_are_stable()
{
    TEST_ASSERT_EQUAL_INT(0, (int)ActuatorConnection::Unknown);
    TEST_ASSERT_EQUAL_INT(1, (int)ActuatorConnection::Connected);
    TEST_ASSERT_EQUAL_INT(2, (int)ActuatorConnection::Disconnected);
}

static void assertConnectionState(
    ActuatorConnection expected,
    bool isPositionValid,
    ActuatorCurrentEvidence currentEvidence)
{
    TEST_ASSERT_EQUAL_INT(
        (int)expected,
        (int)determineActuatorConnectionState(
            isPositionValid, currentEvidence));
}

static void test_invalid_position_is_disconnected_for_every_current_evidence()
{
    assertConnectionState(
        ActuatorConnection::Disconnected,
        false,
        ActuatorCurrentEvidence::Unknown);
    assertConnectionState(
        ActuatorConnection::Disconnected,
        false,
        ActuatorCurrentEvidence::CurrentPresent);
    assertConnectionState(
        ActuatorConnection::Disconnected,
        false,
        ActuatorCurrentEvidence::CurrentAbsent);
}

static void test_valid_position_and_unknown_current_remain_unknown()
{
    assertConnectionState(
        ActuatorConnection::Unknown,
        true,
        ActuatorCurrentEvidence::Unknown);
}

static void test_valid_position_and_present_current_are_connected()
{
    assertConnectionState(
        ActuatorConnection::Connected,
        true,
        ActuatorCurrentEvidence::CurrentPresent);
}

static void test_valid_position_and_absent_current_are_disconnected()
{
    assertConnectionState(
        ActuatorConnection::Disconnected,
        true,
        ActuatorCurrentEvidence::CurrentAbsent);
}

static void test_absent_current_does_not_invalidate_the_pot()
{
    ActuatorPotTracker actuatorPot;
    actuatorPot.reset(500);

    TEST_ASSERT_TRUE(actuatorPot.isValid());
    TEST_ASSERT_EQUAL_INT(
        (int)ActuatorConnection::Disconnected,
        (int)determineActuatorConnectionState(
            actuatorPot.isValid(), ActuatorCurrentEvidence::CurrentAbsent));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_default_status_is_unassigned_unknown_and_idle);
    RUN_TEST(test_connection_state_wire_values_are_stable);
    RUN_TEST(test_invalid_position_is_disconnected_for_every_current_evidence);
    RUN_TEST(test_valid_position_and_unknown_current_remain_unknown);
    RUN_TEST(test_valid_position_and_present_current_are_connected);
    RUN_TEST(test_valid_position_and_absent_current_are_disconnected);
    RUN_TEST(test_absent_current_does_not_invalidate_the_pot);
    return UNITY_END();
}

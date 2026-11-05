#include <unity.h>
#include <type_traits>

#include "src/power_safety/power_message.h"

static_assert(sizeof(PowerMessageType) == sizeof(uint8_t),
              "power-message type must remain one byte");

void setUp() {}
void tearDown() {}

static void accepts_only_the_exact_shutdown_ack_line()
{
    TEST_ASSERT_TRUE(isShutdownAckLine("PWR 1 SHUTDOWN_ACK"));
    TEST_ASSERT_TRUE(isShutdownAckLine("PWR 1 SHUTDOWN_ACK\r"));

    TEST_ASSERT_FALSE(isShutdownAckLine(nullptr));
    TEST_ASSERT_FALSE(isShutdownAckLine(""));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 2 SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 1 SHUTDOWN_ACK extra"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 1 POWERING_DOWN SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("XPWR 1 SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 1 SHUTDOWN_ACKNOWLEDGED"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR / SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR x SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 1XSHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckLine("PWR 256 SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckMessage(
        "PWR 1 SHUTDOWN_ACK", nullptr));
}

static void accepts_only_the_exact_payload_after_prefix_dispatch()
{
    TEST_ASSERT_TRUE(isShutdownAckPayload("WR 1 SHUTDOWN_ACK"));
    TEST_ASSERT_TRUE(isShutdownAckPayload("WR 1 SHUTDOWN_ACK\r"));

    TEST_ASSERT_FALSE(isShutdownAckPayload(nullptr));
    TEST_ASSERT_FALSE(isShutdownAckPayload("PWR 1 SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckPayload("WR 2 SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isShutdownAckPayload("WR 1 SHUTDOWN_ACK extra"));
    TEST_ASSERT_FALSE(isShutdownAckPayload(" CAL SHOW"));
}

static void protocol_tokens_match_the_wire_contract()
{
    TEST_ASSERT_EQUAL_UINT8(1, POWER_MSG_SCHEMA);
    TEST_ASSERT_EQUAL_STRING(
        "POWERING_DOWN",
        powerMessageTypeToken(PowerMessageType::PoweringDown));
    TEST_ASSERT_EQUAL_STRING(
        "SHUTDOWN_ACK",
        powerMessageTypeToken(PowerMessageType::ShutdownAck));
    TEST_ASSERT_EQUAL_STRING(
        "RESUMING",
        powerMessageTypeToken(PowerMessageType::Resuming));
    TEST_ASSERT_EQUAL_STRING(
        "EMERGENCY_SHUTDOWN",
        powerMessageTypeToken(PowerMessageType::EmergencyShutdown));
    TEST_ASSERT_EQUAL_STRING(
        "under_voltage_soft",
        powerReasonToken(PoweringDownReason::UnderVoltageSoft));
    TEST_ASSERT_EQUAL_STRING(
        "hard_cut",
        powerReasonToken(EmergencyShutdownReason::HardCut));
    TEST_ASSERT_EQUAL_STRING(
        "over_voltage",
        powerReasonToken(EmergencyShutdownReason::OverVoltage));
    TEST_ASSERT_EQUAL_STRING(
        "voltage_recovered",
        powerReasonToken(ResumingReason::VoltageRecovered));
}

static void exact_token_rejects_null_inputs()
{
    TEST_ASSERT_FALSE(isExactPowerToken(nullptr, "SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isExactPowerToken("SHUTDOWN_ACK", nullptr));
    TEST_ASSERT_FALSE(isExactPowerToken(
        "SHUTDOWN_ACKX", "SHUTDOWN_ACK"));
    TEST_ASSERT_FALSE(isExactPowerToken(
        "SHUTDOWN_ACX\r", "SHUTDOWN_ACK"));
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(accepts_only_the_exact_shutdown_ack_line);
    RUN_TEST(accepts_only_the_exact_payload_after_prefix_dispatch);
    RUN_TEST(protocol_tokens_match_the_wire_contract);
    RUN_TEST(exact_token_rejects_null_inputs);
    return UNITY_END();
}

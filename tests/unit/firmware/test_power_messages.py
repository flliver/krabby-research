"""Power-state control message protocol (M16 Task 4, firmware/interfaces/power_messages.py).

Naming: test_<unit>_<condition>_<expected>, arrange/act/assert body structure.
"""
import pytest

from firmware.interfaces.power_messages import (
    EmergencyShutdownReason,
    POWER_MSG_SCHEMA,
    PowerMessage,
    PowerMessageType,
    PoweringDownReason,
    ResumingReason,
)


class TestFormatParseRoundTrip:
    @pytest.mark.parametrize(
        "msg",
        [
            PowerMessage(
                PowerMessageType.POWERING_DOWN,
                PoweringDownReason.UNDER_VOLTAGE_SOFT,
            ),
            PowerMessage(
                PowerMessageType.EMERGENCY_SHUTDOWN,
                EmergencyShutdownReason.HARD_CUT,
            ),
            PowerMessage(
                PowerMessageType.EMERGENCY_SHUTDOWN,
                EmergencyShutdownReason.OVER_VOLTAGE,
            ),
            PowerMessage(
                PowerMessageType.RESUMING,
                ResumingReason.VOLTAGE_RECOVERED,
            ),
            PowerMessage(PowerMessageType.SHUTDOWN_ACK),  # no reason
        ],
        ids=lambda m: f"{m.type.name}-{m.reason.name if m.reason else 'none'}",
    )
    def test_format_then_parse_recovers_the_message(self, msg):
        assert PowerMessage.parse(msg.format_line()) == msg

    def test_powering_down_line_is_the_expected_wire_shape(self):
        line = PowerMessage(
            PowerMessageType.POWERING_DOWN,
            PoweringDownReason.UNDER_VOLTAGE_SOFT,
        ).format_line()

        assert line == f"PWR {POWER_MSG_SCHEMA} POWERING_DOWN under_voltage_soft"

    def test_ack_line_omits_the_reason(self):
        assert PowerMessage(PowerMessageType.SHUTDOWN_ACK).format_line() == (
            f"PWR {POWER_MSG_SCHEMA} SHUTDOWN_ACK"
        )

    def test_resuming_line_carries_its_typed_reason(self):
        assert PowerMessage(
            PowerMessageType.RESUMING,
            ResumingReason.VOLTAGE_RECOVERED,
        ).format_line() == (
            f"PWR {POWER_MSG_SCHEMA} RESUMING voltage_recovered"
        )

    def test_emergency_shutdown_line_carries_its_typed_reason(self):
        assert PowerMessage(
            PowerMessageType.EMERGENCY_SHUTDOWN,
            EmergencyShutdownReason.OVER_VOLTAGE,
        ).format_line() == (
            f"PWR {POWER_MSG_SCHEMA} EMERGENCY_SHUTDOWN over_voltage"
        )


class TestParseRejections:
    @pytest.mark.parametrize(
        "line",
        [
            "FRONT; FLHY 0.1 0 0 0 0 0 0 0",  # a telemetry line
            "VER 1.0 main abc",               # a version line
            "PWR",                            # prefix only
            "PWR 1",                          # no type
            "PWR 1 POWERING_DOWN",             # required reason missing
            "PWR 1 EMERGENCY_SHUTDOWN",        # required reason missing
            "PWR 1 SHUTDOWN_ACK extra",        # reasonless type has extra field
            "PWR 1 POWERING_DOWN operator_request extra",  # excess field
            "",
        ],
        ids=[
            "telemetry",
            "version",
            "prefix-only",
            "no-type",
            "powering-down-no-reason",
            "emergency-no-reason",
            "ack-extra-field",
            "powering-down-extra-field",
            "empty",
        ],
    )
    def test_non_power_or_incomplete_line_returns_none(self, line):
        assert PowerMessage.parse(line) is None

    @pytest.mark.parametrize(
        "schema_token",
        ["-1", "0", "2", "255", "256", "x"],
    )
    def test_unsupported_or_malformed_schema_is_dropped_not_guessed(
        self, schema_token
    ):
        # A future or malformed layout must not be interpreted under the
        # current field meanings.
        assert PowerMessage.parse(
            f"PWR {schema_token} POWERING_DOWN under_voltage_soft"
        ) is None

    def test_unknown_message_type_returns_none(self):
        assert PowerMessage.parse(f"PWR {POWER_MSG_SCHEMA} REBOOT_NOW") is None


class TestProducerSchema:
    @pytest.mark.parametrize(
        "schema_version",
        [-1, 0, 2, 255, 256, "1", True, None],
    )
    def test_unsupported_or_noninteger_schema_cannot_be_emitted(
        self, schema_version
    ):
        with pytest.raises(ValueError, match="unsupported power-message schema"):
            PowerMessage(
                PowerMessageType.SHUTDOWN_ACK,
                schema_version=schema_version,
            )

    def test_supported_schema_is_emitted_before_the_message_type(self):
        message = PowerMessage(PowerMessageType.SHUTDOWN_ACK)

        assert message.schema_version == POWER_MSG_SCHEMA
        assert message.format_line().split()[:3] == [
            "PWR",
            str(POWER_MSG_SCHEMA),
            "SHUTDOWN_ACK",
        ]


class TestReasonHierarchy:
    @pytest.mark.parametrize(
        "line",
        [
            "PWR 1 POWERING_DOWN over_voltage",
            "PWR 1 POWERING_DOWN hard_cut",
            "PWR 1 POWERING_DOWN voltage_recovered",
            "PWR 1 EMERGENCY_SHUTDOWN under_voltage_soft",
            "PWR 1 EMERGENCY_SHUTDOWN operator_request",
            "PWR 1 RESUMING over_voltage",
            "PWR 1 POWERING_DOWN brownout_future",
        ],
    )
    def test_unknown_or_cross_family_reason_is_rejected(self, line):
        assert PowerMessage.parse(line) is None

    @pytest.mark.parametrize(
        ("message_type", "reason"),
        [
            (PowerMessageType.POWERING_DOWN, None),
            (
                PowerMessageType.POWERING_DOWN,
                EmergencyShutdownReason.OVER_VOLTAGE,
            ),
            (
                PowerMessageType.EMERGENCY_SHUTDOWN,
                PoweringDownReason.UNDER_VOLTAGE_SOFT,
            ),
            (
                PowerMessageType.SHUTDOWN_ACK,
                PoweringDownReason.UNDER_VOLTAGE_SOFT,
            ),
            (PowerMessageType.RESUMING, EmergencyShutdownReason.HARD_CUT),
        ],
    )
    def test_invalid_constructed_message_is_rejected(self, message_type, reason):
        with pytest.raises(ValueError):
            PowerMessage(message_type, reason)

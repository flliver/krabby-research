"""SDK reader-loop dispatch of power-control lines (M16 Task 4, §6.1).

The Orin reads the MCU on one serial reader; power lines must be recognized
there and handed to a callback, never a second port. These run against a bare
KrabbyMCUSDK (no hal imports, so they run outside the Docker test image).
"""
import logging
from unittest.mock import Mock

from firmware.interfaces.power_messages import PowerMessage, PowerMessageType
from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.mcu_port import McuSerialDevice


def _bare_sdk() -> KrabbyMCUSDK:
    return KrabbyMCUSDK(target=McuSerialDevice("unused"))


def test_reader_dispatch_sets_last_message_and_invokes_callback():
    sdk = _bare_sdk()
    seen = []
    sdk.on_power_message = seen.append

    sdk._handle_power_line("PWR 1 POWERING_DOWN under_voltage_soft")

    assert sdk.last_power_message is not None
    assert sdk.last_power_message.type is PowerMessageType.POWERING_DOWN
    assert [m.type for m in seen] == [PowerMessageType.POWERING_DOWN]


def test_unrecognized_power_line_is_dropped():
    sdk = _bare_sdk()
    seen = []
    sdk.on_power_message = seen.append

    sdk._handle_power_line("PWR 99 POWERING_DOWN under_voltage_soft")  # unknown schema

    assert sdk.last_power_message is None
    assert seen == []


def test_power_line_without_registered_callback_still_stores_last_message():
    sdk = _bare_sdk()

    sdk._handle_power_line("PWR 1 RESUMING voltage_recovered")

    assert sdk.last_power_message.type is PowerMessageType.RESUMING


def test_raising_callback_does_not_propagate(caplog):
    sdk = _bare_sdk()

    def boom(msg):
        raise RuntimeError("handler bug")

    sdk.on_power_message = boom

    with caplog.at_level(logging.ERROR, logger="KrabbySDK"):
        sdk._handle_power_line("PWR 1 SHUTDOWN_ACK")  # must not raise

    assert any("callback raised" in r.getMessage() for r in caplog.records)


def test_send_power_message_writes_line_with_newline():
    # The Orin's only path to put SHUTDOWN_ACK on the wire. Verify the exact wire
    # line independently, then that send writes it newline-terminated + flushes.
    sdk = _bare_sdk()
    sdk.ser = Mock()
    sdk.ser.is_open = True
    msg = PowerMessage(PowerMessageType.SHUTDOWN_ACK)
    assert msg.format_line() == "PWR 1 SHUTDOWN_ACK"

    sdk.send_power_message(msg)

    sdk.ser.write.assert_called_once_with(b"PWR 1 SHUTDOWN_ACK\n")
    sdk.ser.flush.assert_called_once()


def test_send_power_message_closed_port_is_noop():
    # A closed/absent port is a silent no-op — no write, no exception.
    sdk = _bare_sdk()
    sdk.ser = Mock()
    sdk.ser.is_open = False

    sdk.send_power_message(PowerMessage(PowerMessageType.SHUTDOWN_ACK))

    sdk.ser.write.assert_not_called()


def test_reader_loop_routes_pwr_line_to_handler():
    # End-to-end: the reader loop's 'PWR '-prefix branch must route a PWR line to
    # _handle_power_line and thus to the registered callback (not just the direct
    # _handle_power_line path the other tests exercise).
    sdk = _bare_sdk()
    seen = []
    sdk.on_power_message = seen.append
    sdk.running = True

    ser = Mock()
    ser.is_open = True
    lines = iter([b"PWR 1 POWERING_DOWN under_voltage_soft\n"])

    def fake_readline():
        try:
            return next(lines)
        except StopIteration:
            sdk.running = False  # drain the queued line, then stop the loop
            return b""

    ser.readline.side_effect = fake_readline
    sdk.ser = ser

    sdk._reader_loop()

    assert [m.type for m in seen] == [PowerMessageType.POWERING_DOWN]

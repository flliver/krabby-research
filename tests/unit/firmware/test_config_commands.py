"""Unit tests for the M17 Task 1 SET/GET config command path (SDK + CLI)."""
import threading
import time

import pytest
from unittest.mock import Mock

import firmware.cli as cli_mod
from firmware.krabby_mcu import (
    KrabbyMCUSDK,
    build_get_line,
    build_set_line,
    parse_get_reply,
)


# --- build_set_line -------------------------------------------------------

class TestBuildSetLine:
    def test_front_default(self):
        assert build_set_line(None, [("role", "FRONT")]) == "SET role FRONT"

    def test_multiple_pairs_order_preserved(self):
        line = build_set_line(None, [("role", "FRONT"), ("serial", "FRT-0042")])
        assert line == "SET role FRONT serial FRT-0042"

    def test_left_and_right_get_suffix(self):
        assert build_set_line("left", [("role", "LEFT")]) == "SET_LEFT role LEFT"
        assert build_set_line("right", [("role", "RIGHT")]) == "SET_RIGHT role RIGHT"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown config key"):
            build_set_line(None, [("bogus", "x")])

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="invalid role"):
            build_set_line(None, [("role", "SIDEWAYS")])

    def test_invalid_board_raises(self):
        with pytest.raises(ValueError, match="invalid board"):
            build_set_line("up", [("role", "LEFT")])

    def test_serial_with_space_raises(self):
        with pytest.raises(ValueError, match="no spaces"):
            build_set_line(None, [("serial", "has space")])

    def test_serial_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            build_set_line(None, [("serial", "X" * 16)])

    def test_serial_at_max_len_ok(self):
        assert build_set_line(None, [("serial", "X" * 15)]) == "SET serial " + "X" * 15

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            build_set_line(None, [])

    def test_role_unknown_is_valid(self):
        assert build_set_line(None, [("role", "UNKNOWN")]) == "SET role UNKNOWN"


# --- build_get_line -------------------------------------------------------

class TestBuildGetLine:
    def test_front_default(self):
        assert build_get_line(None, ["role", "serial"]) == "GET role serial"

    def test_left_suffix(self):
        assert build_get_line("left", ["role"]) == "GET_LEFT role"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown config key"):
            build_get_line(None, ["bogus"])

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            build_get_line(None, [])


# --- parse_get_reply ------------------------------------------------------

class TestParseGetReply:
    def test_front_reply(self):
        assert parse_get_reply("GET role FRONT serial FRT-0042") == (
            "front", {"role": "FRONT", "serial": "FRT-0042"})

    def test_left_reply(self):
        assert parse_get_reply("GET_LEFT role LEFT") == ("left", {"role": "LEFT"})

    def test_right_reply(self):
        assert parse_get_reply("GET_RIGHT serial RGT-0019") == (
            "right", {"serial": "RGT-0019"})

    def test_non_get_line_returns_none(self):
        assert parse_get_reply("VER 1.0 main abc") is None
        assert parse_get_reply("FRONT; FLHY 0.5 ...") is None
        assert parse_get_reply("") is None

    def test_unset_serial_sentinel(self):
        # firmware prints "-" for an unset serial
        assert parse_get_reply("GET serial -") == ("front", {"serial": "-"})


# --- send_set / send_get (mock serial, no reader thread) ------------------

class TestSendSetGet:
    def _bare_sdk(self):
        sdk = object.__new__(KrabbyMCUSDK)
        sdk._last_get_line = None
        sdk.ser = Mock()
        sdk.ser.is_open = True
        return sdk

    def test_send_set_writes_wire_line(self):
        sdk = self._bare_sdk()
        sdk.send_set(role="FRONT", serial="FRT-0042")
        sdk.ser.write.assert_called_once_with(b"SET role FRONT serial FRT-0042\n")
        sdk.ser.flush.assert_called()

    def test_send_set_board_left_suffix(self):
        sdk = self._bare_sdk()
        sdk.send_set(board="left", role="LEFT")
        sdk.ser.write.assert_called_once_with(b"SET_LEFT role LEFT\n")

    def test_send_set_invalid_raises_before_write(self):
        sdk = self._bare_sdk()
        with pytest.raises(ValueError):
            sdk.send_set(role="NORTH")
        sdk.ser.write.assert_not_called()

    def test_send_get_returns_parsed_reply(self):
        sdk = self._bare_sdk()

        def deliver():
            time.sleep(0.05)
            sdk._last_get_line = "GET role FRONT serial FRT-0042"

        t = threading.Thread(target=deliver)
        t.start()
        result = sdk.send_get("role", "serial", timeout=0.5)
        t.join()

        assert result == {"role": "FRONT", "serial": "FRT-0042"}
        sdk.ser.write.assert_called_once_with(b"GET role serial\n")

    def test_send_get_times_out_returns_none(self):
        sdk = self._bare_sdk()
        assert sdk.send_get("role", timeout=0.1) is None

    def test_send_get_ignores_reply_for_other_board(self):
        sdk = self._bare_sdk()

        def deliver():
            time.sleep(0.05)
            sdk._last_get_line = "GET_LEFT role LEFT"  # we asked the front board

        t = threading.Thread(target=deliver)
        t.start()
        result = sdk.send_get("role", timeout=0.3)
        t.join()
        assert result is None


# --- CLI _parse_assignments ----------------------------------------------

class TestParseAssignments:
    def test_basic(self):
        assert cli_mod._parse_assignments(["role=FRONT", "serial=FRT-0042"]) == [
            ("role", "FRONT"), ("serial", "FRT-0042")]

    def test_value_with_equals_kept(self):
        # only the first '=' splits, so '=' in a value survives
        assert cli_mod._parse_assignments(["serial=A=B"]) == [("serial", "A=B")]

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            cli_mod._parse_assignments(["role"])

    def test_empty_key_or_value_raises(self):
        with pytest.raises(ValueError):
            cli_mod._parse_assignments(["=FRONT"])
        with pytest.raises(ValueError):
            cli_mod._parse_assignments(["role="])

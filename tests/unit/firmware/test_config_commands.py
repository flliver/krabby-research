"""Unit tests for the SET/GET board-config command path (SDK + CLI)."""
import threading
import time
from pathlib import Path

import pytest

import firmware.cli as cli_mod
from firmware.krabby_mcu import (
    build_get_line,
    build_set_line,
    parse_get_reply,
)

SKETCH = Path(__file__).resolve().parents[3] / "firmware" / "arduino" / "arduino.ino"


class TestBuildSetLine:
    def test_front_default(self):
        assert build_set_line(None, [("role", "FRONT")]) == "SET role FRONT"

    def test_left_and_right_get_suffix(self):
        assert build_set_line("left", [("role", "LEFT")]) == "SET_LEFT role LEFT"
        assert build_set_line("right", [("role", "RIGHT")]) == "SET_RIGHT role RIGHT"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown config key"):
            build_set_line(None, [("colour", "red")])

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError, match="invalid role"):
            build_set_line(None, [("role", "NORTH")])

    def test_invalid_board_raises(self):
        with pytest.raises(ValueError, match="invalid board"):
            build_set_line("middle", [("role", "LEFT")])

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError):
            build_set_line(None, [])

    def test_role_unknown_is_valid(self):
        assert build_set_line(None, [("role", "UNKNOWN")]) == "SET role UNKNOWN"

    def test_version_is_not_settable(self):
        with pytest.raises(ValueError, match="unknown config key"):
            build_set_line(None, [("version", "1.0")])


class TestBuildGetLine:
    def test_front_default(self):
        assert build_get_line(None, ["role", "version"]) == "GET role version"

    def test_left_suffix(self):
        assert build_get_line("left", ["role"]) == "GET_LEFT role"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown config key"):
            build_get_line(None, ["colour"])

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError):
            build_get_line(None, [])


class TestParseGetReply:
    def test_front_reply(self):
        assert parse_get_reply("GET role FRONT") == ("front", {"role": "FRONT"})

    def test_left_and_right_replies(self):
        assert parse_get_reply("GET_LEFT role LEFT") == ("left", {"role": "LEFT"})
        assert parse_get_reply("GET_RIGHT role RIGHT") == ("right", {"role": "RIGHT"})

    def test_non_get_line_returns_none(self):
        assert parse_get_reply("FRONT; FLHY 0.5 500 0 0 0 0 0 0") is None
        assert parse_get_reply("VER 1.0 main abc") is None
        assert parse_get_reply("") is None

    def test_version_reply_pipe_joined(self):
        assert parse_get_reply("GET version 0.2.16|main|abc123") == (
            "front", {"version": "0.2.16|main|abc123"})


class TestSendSetGet:
    def test_send_set_writes_wire_line(self, bare_sdk):
        bare_sdk.send_set(role="FRONT")
        bare_sdk.ser.write.assert_called_once_with(b"SET role FRONT\n")
        bare_sdk.ser.flush.assert_called()

    def test_send_set_board_left_suffix(self, bare_sdk):
        bare_sdk.send_set(board="left", role="LEFT")
        bare_sdk.ser.write.assert_called_once_with(b"SET_LEFT role LEFT\n")

    def test_send_set_invalid_raises_before_write(self, bare_sdk):
        with pytest.raises(ValueError):
            bare_sdk.send_set(role="NORTH")
        bare_sdk.ser.write.assert_not_called()

    def test_send_get_returns_parsed_reply(self, bare_sdk):
        def deliver():
            time.sleep(0.05)
            bare_sdk._last_get_line = "GET role FRONT version 0.2.16|main|abc"

        t = threading.Thread(target=deliver)
        t.start()
        result = bare_sdk.send_get("role", "version", timeout=0.5)
        t.join()
        assert result == {"role": "FRONT", "version": "0.2.16|main|abc"}
        bare_sdk.ser.write.assert_called_once_with(b"GET role version\n")

    def test_send_get_times_out_returns_none(self, bare_sdk):
        assert bare_sdk.send_get("role", timeout=0.1) is None

    def test_send_get_ignores_reply_for_other_board(self, bare_sdk):
        def deliver():
            time.sleep(0.05)
            bare_sdk._last_get_line = "GET_LEFT role LEFT"  # we asked the front board

        t = threading.Thread(target=deliver)
        t.start()
        result = bare_sdk.send_get("role", timeout=0.3)
        t.join()
        assert result is None


class TestParseAssignments:
    def test_basic(self):
        assert cli_mod._parse_assignments(["role=FRONT"]) == [("role", "FRONT")]

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            cli_mod._parse_assignments(["role"])

    def test_empty_key_or_value_raises(self):
        with pytest.raises(ValueError):
            cli_mod._parse_assignments(["=FRONT"])
        with pytest.raises(ValueError):
            cli_mod._parse_assignments(["role="])


class TestSketchRoleSource:
    """The role comes from EEPROM + SET, never from a boot-time election."""

    def test_no_sync_election(self):
        src = SKETCH.read_text()
        assert "SYNC" not in src
        assert "determineRole" not in src

    def test_boot_applies_eeprom_role(self):
        assert "applyRole(loadRole());" in SKETCH.read_text()

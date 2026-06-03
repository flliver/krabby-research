import threading
import time
import pytest
from unittest.mock import Mock

from firmware.krabby_mcu import KrabbyMCUSDK, parse_ver_reply


class TestParseVerReply:
    def test_single_board_reply(self):
        result = parse_ver_reply("VER dev-local dev-local dev-local")
        assert result == [("dev-local", "dev-local", "dev-local")]

    def test_three_board_combined_reply(self):
        result = parse_ver_reply("VER 1.0|2.0|3.0 main|feat|fix abc123|def456|ghi789")
        assert result == [
            ("1.0", "main", "abc123"),
            ("2.0", "feat", "def456"),
            ("3.0", "fix",  "ghi789"),
        ]

    def test_partial_boards_dashes(self):
        # Single connected board; left/right show "-"
        result = parse_ver_reply("VER dev-local|-|- dev-local|-|- dev-local|-|-")
        assert len(result) == 3
        assert result[0] == ("dev-local", "dev-local", "dev-local")
        assert result[1] == ("-", "-", "-")
        assert result[2] == ("-", "-", "-")

    def test_missing_branch_and_commit_fields(self):
        result = parse_ver_reply("VER 1.2.3")
        assert result == [("1.2.3", "-", "-")]

    def test_missing_commit_field(self):
        result = parse_ver_reply("VER 1.2.3 main")
        assert result == [("1.2.3", "main", "-")]

    def test_not_ver_line_returns_none(self):
        assert parse_ver_reply("FRONT; pos=0.5 ...") is None
        assert parse_ver_reply("") is None
        assert parse_ver_reply("VE") is None

    def test_ver_prefix_only_returns_none(self):
        assert parse_ver_reply("VER ") is None

    def test_trailing_whitespace_on_line(self):
        # readline() may leave trailing whitespace; parse_ver_reply should handle it
        result = parse_ver_reply("VER 1.0|2.0|3.0 main|feat|fix abc|def|ghi   ")
        assert result == [
            ("1.0", "main", "abc"),
            ("2.0", "feat", "def"),
            ("3.0", "fix",  "ghi"),
        ]


class TestReadVersion:
    def _bare_sdk(self):
        """KrabbyMCUSDK instance with a mock serial, no reader thread."""
        sdk = object.__new__(KrabbyMCUSDK)
        sdk._last_ver_line = None
        sdk.ser = Mock()
        sdk.ser.is_open = True
        return sdk

    def test_sends_v_command_and_returns_ver_line(self):
        sdk = self._bare_sdk()

        def deliver():
            time.sleep(0.05)
            sdk._last_ver_line = "VER dev-local dev-local dev-local"

        t = threading.Thread(target=deliver)
        t.start()
        result = sdk.read_version(timeout=0.5)
        t.join()

        assert result == "VER dev-local dev-local dev-local"
        sdk.ser.write.assert_called_once_with(b"V\n")
        sdk.ser.flush.assert_called()

    def test_returns_none_on_timeout(self):
        sdk = self._bare_sdk()
        result = sdk.read_version(timeout=0.1)
        assert result is None

    def test_returns_none_when_ser_is_none(self):
        sdk = self._bare_sdk()
        sdk.ser = None
        assert sdk.read_version() is None

    def test_returns_none_when_port_closed(self):
        sdk = self._bare_sdk()
        sdk.ser.is_open = False
        assert sdk.read_version() is None

    def test_clears_stale_ver_line_before_sending(self):
        sdk = self._bare_sdk()
        sdk._last_ver_line = "VER stale stale stale"

        def deliver():
            time.sleep(0.05)
            sdk._last_ver_line = "VER fresh fresh fresh"

        t = threading.Thread(target=deliver)
        t.start()
        result = sdk.read_version(timeout=0.5)
        t.join()

        assert result == "VER fresh fresh fresh"

import threading
import time
import pytest
from unittest.mock import Mock

from firmware import krabby_mcu
from firmware.interfaces.joint_telemetry import JointTelemetry
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


class _QuietSerial:
    """A port that never has data: the reader loop ends on its first read."""

    def __init__(self, owner):
        self._owner = owner
        self.is_open = True
        self.written = []

    def readline(self):
        self._owner.running = False
        return b""

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def cancel_read(self):
        pass

    def close(self):
        self.is_open = False


class TestConnectClearsHostState:
    """A new connection starts with no telemetry and no SENS states (design 8.2)."""

    def _connect(self, monkeypatch, sdk):
        monkeypatch.setattr(krabby_mcu.time, "sleep", lambda _s: None)
        port = _QuietSerial(sdk)
        sdk.target = Mock()
        sdk.target.open.return_value = port
        assert sdk.connect() is True
        sdk.thread.join(timeout=2.0)
        return port

    def test_joints_and_sensor_states_are_cleared(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        sdk.joints["FLHL"] = JointTelemetry(
            name="FLHL", pos=0.5, pot=500, current=0, en=(0, 0), pwm=(0, 0), saf=10
        )
        sdk.joint_sensor_state["FLHL"] = "E"
        sdk.imu = object()
        sdk.battery = object()
        self._connect(monkeypatch, sdk)
        assert sdk.joints == {}
        assert sdk.joint_sensor_state == {}
        assert sdk.sensor_state("FLHL") is None
        assert sdk.imu is None and sdk.battery is None

    def test_a_reconnect_does_not_keep_a_ready_joint(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        sdk.joints["FLHL"] = JointTelemetry(
            name="FLHL", pos=0.5, pot=500, current=0, en=(0, 0), pwm=(0, 0), saf=10
        )
        sdk.joint_sensor_state["FLHL"] = "E"
        assert sdk.position_ready("FLHL")
        self._connect(monkeypatch, sdk)
        assert not sdk.position_ready("FLHL")
        assert sdk.neutral_targets()[0] == {}

    def test_connect_still_sends_hold(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        port = self._connect(monkeypatch, sdk)
        assert port.written == [b"H\n"]

    def test_failed_connect_leaves_state_alone(self, monkeypatch):
        monkeypatch.setattr(krabby_mcu.time, "sleep", lambda _s: None)
        sdk = KrabbyMCUSDK(target=Mock())
        sdk.target.open.side_effect = OSError("no port")
        sdk.joint_sensor_state["FLHL"] = "E"
        assert sdk.connect() is False
        assert sdk.sensor_state("FLHL") == "E"

    def test_states_learned_after_connect_are_kept(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        self._connect(monkeypatch, sdk)
        sdk._record_sensor_line("SENS FRONT FLHL:e")
        assert sdk.sensor_state("FLHL") == "e"

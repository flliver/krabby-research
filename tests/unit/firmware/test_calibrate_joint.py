"""Unit tests for the per-joint calibration command path (M17 Task 2, 2g):
the SDK `calibrate_joint` (wire `K<name>`), the CLI `calibrate-joint`, and source
guards on the firmware state-machine wiring (which can't be bench-checked w/o a re-cal).
"""
import re
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

import firmware.cli as cli_mod
from firmware.krabby_mcu import ALL_JOINT_NAMES, KrabbyMCUSDK, parse_cal_reply

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


def _bare_sdk():
    sdk = object.__new__(KrabbyMCUSDK)
    sdk.ser = Mock()
    sdk.ser.is_open = True
    return sdk


class TestCalibrateJointSDK:
    def test_wire_format(self):
        sdk = _bare_sdk()
        sdk.calibrate_joint("FLHL")
        sdk.ser.write.assert_called_once_with(b"KFLHL\n")
        sdk.ser.flush.assert_called()

    def test_unknown_joint_raises_before_write(self):
        sdk = _bare_sdk()
        with pytest.raises(ValueError, match="unknown joint"):
            sdk.calibrate_joint("NOPE")
        sdk.ser.write.assert_not_called()

    def test_all_18_joints_valid_and_wire_clean(self):
        assert len(ALL_JOINT_NAMES) == 18
        for name in ALL_JOINT_NAMES:
            sdk = _bare_sdk()
            sdk.calibrate_joint(name)
            sdk.ser.write.assert_called_once_with(f"K{name}\n".encode())


class TestCalibrateJointCLI:
    def test_rejects_unknown_joint_before_connecting(self):
        # exits at client-side validation, before any port is opened
        with pytest.raises(SystemExit):
            cli_mod.cmd_calibrate_joint(None, "BOGUS")


@pytest.fixture(scope="module")
def actuator() -> str:
    return (ARDUINO / "actuator_manager.h").read_text()


@pytest.fixture(scope="module")
def ino() -> str:
    return (ARDUINO / "arduino.ino").read_text()


class TestCalibrateMachineFirmware:
    def test_entry_points_exist(self, actuator):
        assert re.search(r"void\s+calibrateJoint\s*\(", actuator)
        assert re.search(r"void\s+calibrateJointByName\s*\(", actuator)
        assert re.search(r"void\s+updateJointCal\s*\(", actuator)

    def test_updateall_routes_single_joint_cal(self, actuator):
        body = actuator[actuator.index("void updateAll"):actuator.index("void updateAll") + 300]
        assert re.search(r"if\s*\(\s*jcActive\s*\)", body)
        assert "updateJointCal" in body

    def test_save_persists_and_applies_immediately(self, actuator):
        body = actuator[actuator.index("void updateJointCal"):]
        assert "jointCalSave" in body and "applyJointCal" in body
        # the calibrated flag is range-gated; see TestRangeCheckFirmware
        assert "jc.calibrated" in body

    def test_motor_did_not_move_on_no_motion(self, actuator):
        assert "motor_did_not_move" in actuator

    def test_k_command_dispatched_and_forwarded(self, ino):
        assert re.search(r"cmdType\s*==\s*'K'", ino), "loop() must dispatch the K command"
        assert "calibrateJointByName" in ino

    def test_error_output_wired_to_main_serial(self, ino):
        assert "setErrorOutput" in ino, "applyRole must point the cal ERR sink at mainSerial"


# --- calibration read-back (Q<name> / get-calibration) -------------------

class TestParseCalReply:
    def test_basic(self):
        assert parse_cal_reply("CAL FLHL type POT rev 0 min 120 max 905 cal 1") == (
            "FLHL", {"type": "POT", "rev": "0", "min": "120", "max": "905", "cal": "1"})

    def test_non_cal_lines(self):
        assert parse_cal_reply("GET role FRONT") is None
        assert parse_cal_reply("") is None


class TestGetCalibrationSDK:
    def _bare(self):
        sdk = object.__new__(KrabbyMCUSDK)
        sdk._last_cal_line = None
        sdk.ser = Mock()
        sdk.ser.is_open = True
        return sdk

    def test_unknown_joint_raises(self):
        with pytest.raises(ValueError, match="unknown joint"):
            self._bare().get_calibration("NOPE")

    def test_returns_parsed_dict_and_sends_query(self):
        sdk = self._bare()

        def deliver():
            time.sleep(0.05)
            sdk._last_cal_line = "CAL FLHL type POT rev 0 min 120 max 905 cal 1"

        t = threading.Thread(target=deliver)
        t.start()
        result = sdk.get_calibration("FLHL", timeout=0.5)
        t.join()
        assert result == {"type": "POT", "rev": "0", "min": "120", "max": "905", "cal": "1"}
        sdk.ser.write.assert_called_once_with(b"QFLHL\n")

    def test_times_out_to_none(self):
        assert self._bare().get_calibration("FLHL", timeout=0.1) is None

    def test_ignores_reply_for_other_joint(self):
        sdk = self._bare()

        def deliver():
            time.sleep(0.05)
            sdk._last_cal_line = "CAL FRHL type POT rev 0 min 10 max 900 cal 1"

        t = threading.Thread(target=deliver)
        t.start()
        assert sdk.get_calibration("FLHL", timeout=0.3) is None
        t.join()


class TestFormatCal:
    def test_trusted_shows_span_no_warning(self):
        s = cli_mod._format_cal("FLHL", {"type": "POT", "rev": "0", "min": "120", "max": "905", "cal": "1"})
        assert "span=785" in s and "NOT TRUSTED" not in s

    def test_untrusted_is_flagged(self):
        s = cli_mod._format_cal("FLHL", {"type": "POT", "rev": "0", "min": "500", "max": "505", "cal": "0"})
        assert "NOT TRUSTED" in s

    def test_none_is_no_readback(self):
        assert "no calibration" in cli_mod._format_cal("FLHL", None).lower()


class TestRangeCheckFirmware:
    def test_span_check_gates_calibrated_flag(self, actuator):
        body = actuator[actuator.index("case JC_SAVE"):]
        assert "JC_POT_MIN_SPAN" in body, "JC_SAVE must check the swept span"
        assert "pot_value_invalid" in body, "a too-small pot span must emit pot_value_invalid"
        assert re.search(r"calibrated\s*=\s*ok\s*\?\s*1\s*:\s*0", body)

    def test_query_command_dispatched(self, actuator, ino):
        assert "queryCalByName" in actuator and "printJointCal" in actuator
        assert re.search(r"cmdType\s*==\s*'Q'", ino), "loop() must dispatch the Q (read cal) command"

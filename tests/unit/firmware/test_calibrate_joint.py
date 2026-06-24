"""Unit tests for the per-joint calibration command path (M17 Task 2, 2g):
the SDK `calibrate_joint` (wire `K<name>`), the CLI `calibrate-joint`, and source
guards on the firmware state-machine wiring (which can't be bench-checked w/o a re-cal).
"""
import re
from pathlib import Path
from unittest.mock import Mock

import pytest

import firmware.cli as cli_mod
from firmware.krabby_mcu import ALL_JOINT_NAMES, KrabbyMCUSDK

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
        assert "calibrated = 1" in body

    def test_motor_did_not_move_on_no_motion(self, actuator):
        assert "motor_did_not_move" in actuator

    def test_k_command_dispatched_and_forwarded(self, ino):
        assert re.search(r"cmdType\s*==\s*'K'", ino), "loop() must dispatch the K command"
        assert "calibrateJointByName" in ino

    def test_error_output_wired_to_main_serial(self, ino):
        assert "setErrorOutput" in ino, "applyRole must point the cal ERR sink at mainSerial"

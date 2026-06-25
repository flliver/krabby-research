### Unit tests for KrabbyMCUSDK (hal/server/jetson/krabby_mcusdk.py).
### Run: pytest tests/unit/hal/server/jetson/test_krabby_mcusdk.py -v

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from unittest.mock import Mock, patch

from hal.server.jetson.krabby_mcusdk import (
    FULL_CAL_STATE,
    JOINT_LIMIT_RAD,
    JOINT_NEUTRAL,
    KrabbyMCUSDK,
    _hal_to_firmware_name,
    _map_mcu_joints_to_normalized,
    _rad_to_pwm,
)
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


class TestHalToFirmwareName:
    def test_known_suffixes(self):
        assert _hal_to_firmware_name("FL_hip_yaw") == "FLHY"
        assert _hal_to_firmware_name("FL_hip_pitch") == "FLHL"
        assert _hal_to_firmware_name("FL_knee") == "FLKL"
        assert _hal_to_firmware_name("RR_hip_yaw") == "RRHY"

    def test_unknown_suffix_first_two_chars_upper(self):
        assert _hal_to_firmware_name("FL_ab") == "FLAB"

    def test_short_suffix_fallback(self):
        assert _hal_to_firmware_name("FL_x") == "FL??"


class TestRadToPwm:
    def test_zero_rad_gives_zero_pwm(self):
        assert _rad_to_pwm(0.0) == 0

    def test_positive_and_negative(self):
        assert _rad_to_pwm(0.1) == 51
        assert _rad_to_pwm(-0.1) == -51

    def test_clamp_at_limits(self):
        assert _rad_to_pwm(JOINT_LIMIT_RAD) == 255
        assert _rad_to_pwm(-JOINT_LIMIT_RAD) == -255
        assert _rad_to_pwm(1.0) == 255
        assert _rad_to_pwm(-1.0) == -255


class TestMapMcuJointsToNormalized:
    def test_firmware_keys_and_normalized_range(self):
        mcu_joints = ("FL_hip_yaw", "FL_hip_pitch")
        command = {"FL_hip_yaw": 0.0, "FL_hip_pitch": JOINT_LIMIT_RAD}
        out = _map_mcu_joints_to_normalized(command, mcu_joints)
        assert set(out.keys()) == {"FLHY", "FLHL"}
        assert out["FLHY"] == pytest.approx(JOINT_NEUTRAL)
        assert out["FLHL"] == pytest.approx(1.0)
        for v in out.values():
            assert 0.0 <= v <= 1.0

    def test_missing_joint_defaults_to_zero_rad(self):
        mcu_joints = ("FL_knee",)
        command = {}
        out = _map_mcu_joints_to_normalized(command, mcu_joints)
        assert out["FLKL"] == pytest.approx(JOINT_NEUTRAL)


class TestKrabbyMCUSDKInit:
    @patch("hal.server.jetson.krabby_mcusdk.FirmwareKrabbyMCUSDK", Mock())
    def test_init_raises_value_error_for_wrong_joint_count(self):
        with pytest.raises(ValueError, match="18 names.*got 17"):
            KrabbyMCUSDK(mcu_joints=("A",) * 17, auto_connect=False)
        with pytest.raises(ValueError, match="18 names.*got 19"):
            KrabbyMCUSDK(mcu_joints=("A",) * 19, auto_connect=False)
    def test_init_succeeds_with_18_joints(self):
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        assert len(mcu_joints) == 18
        sdk = KrabbyMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        assert sdk._mcu_joints == mcu_joints


class TestApplyCommandRouting:
    """apply_command routes FULLY-calibrated joints to closed-loop position
    targets and everything else to open-loop jog (Task 2 §6 hybrid)."""

    def _make_sdk(self, joints_telemetry=None):
        mcu_joints = KRABBY_HEX_DEFINITION.get_mcu_joints()
        sdk = KrabbyMCUSDK(mcu_joints=mcu_joints, auto_connect=False)
        sdk._connected = True
        sdk._mcu = Mock()
        sdk._mcu.running = True
        sdk._mcu.joints = joints_telemetry or {}
        return sdk

    def _make_command(self, positions):
        cmd = Mock()
        cmd.to_positions_dict.return_value = positions
        cmd.timestamp_ns = 0
        cmd.observation_timestamp_ns = 0
        return cmd

    def _neutral_positions(self, sdk):
        return {n: 0.0 for n in sdk._mcu_joints}

    def test_full_joint_targeted_others_jogged(self):
        sdk = self._make_sdk({"FLHL": Mock(cal_state=FULL_CAL_STATE)})
        positions = self._neutral_positions(sdk)
        positions["FL_knee"] = 0.1  # non-neutral, uncalibrated -> jog with PWM
        sdk.apply_command(self._make_command(positions))

        sdk._mcu.send_command_joints.assert_called_once()
        targets = sdk._mcu.send_command_joints.call_args[0][0]
        assert set(targets) == {"FLHL"}
        assert targets["FLHL"] == pytest.approx(JOINT_NEUTRAL)  # 0.0 rad -> 0.5

        sdk._mcu.send_commands_jog.assert_called_once()
        jogs = sdk._mcu.send_commands_jog.call_args[0][0]
        assert "FLHL" not in jogs  # the FULL joint is NOT also jogged
        assert jogs["FLKL"] == _rad_to_pwm(0.1)
        assert jogs["FLHY"] == 0  # neutral -> 0 PWM

    def test_partial_joint_is_jogged_not_targeted(self):
        # PARTIAL (Hall, unanchored) is not yet trustworthy for absolute position.
        sdk = self._make_sdk({"FLHL": Mock(cal_state=1)})
        sdk.apply_command(self._make_command(self._neutral_positions(sdk)))
        sdk._mcu.send_command_joints.assert_not_called()
        jogs = sdk._mcu.send_commands_jog.call_args[0][0]
        assert "FLHL" in jogs

    def test_all_uncalibrated_only_jog(self):
        sdk = self._make_sdk({})  # no telemetry -> everything jogs
        sdk.apply_command(self._make_command(self._neutral_positions(sdk)))
        sdk._mcu.send_command_joints.assert_not_called()
        sdk._mcu.send_commands_jog.assert_called_once()

    def test_all_full_only_targets(self):
        sdk = self._make_sdk()
        sdk._mcu.joints = {
            _hal_to_firmware_name(n): Mock(cal_state=FULL_CAL_STATE)
            for n in sdk._mcu_joints
        }
        sdk.apply_command(self._make_command(self._neutral_positions(sdk)))
        sdk._mcu.send_command_joints.assert_called_once()
        targets = sdk._mcu.send_command_joints.call_args[0][0]
        assert len(targets) == 18
        sdk._mcu.send_commands_jog.assert_not_called()

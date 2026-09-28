### Unit tests for KrabbyMCUSDK (hal/server/jetson/krabby_mcusdk.py).
### Run: pytest tests/unit/hal/server/jetson/test_krabby_mcusdk.py -v

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[5]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pytest
from unittest.mock import Mock, patch

from firmware.interfaces.joint_telemetry import JointTelemetry
from hal.server.jetson.krabby_mcusdk import (
    CONTACT_FULLSCALE,
    CONTACT_LEGS,
    JOINT_LIMIT_RAD,
    JOINT_NEUTRAL,
    KrabbyMCUSDK,
    _hal_to_firmware_name,
    _map_mcu_joints_to_normalized,
    _normalized_to_rad,
    _rad_to_pwm,
    contact_forces_from_joints,
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


def _joint(name: str, current: int) -> JointTelemetry:
    return JointTelemetry(name=name, pos=0.5, pot=500, current=current, en=(0, 0), pwm=(0, 0), saf=0)


class TestContactForcesFromJoints:
    def test_no_telemetry_reads_unknown(self):
        assert contact_forces_from_joints({}) == [0.0] * 5

    def test_leg_current_is_summed_and_scaled(self):
        half = CONTACT_FULLSCALE / 2
        joints = {"FLHY": _joint("FLHY", 0), "FLHL": _joint("FLHL", half / 2), "FLKL": _joint("FLKL", half / 2)}
        assert contact_forces_from_joints(joints)[CONTACT_LEGS.index("FL")] == pytest.approx(0.0)

    def test_clipped_to_model_range(self):
        joints = {"RRHL": _joint("RRHL", 0), "MLKL": _joint("MLKL", 10 * CONTACT_FULLSCALE)}
        forces = contact_forces_from_joints(joints)
        assert forces[CONTACT_LEGS.index("RR")] == -0.5
        assert forces[CONTACT_LEGS.index("ML")] == 0.5

    def test_dropped_middle_right_leg_is_ignored(self):
        assert "MR" not in CONTACT_LEGS
        assert contact_forces_from_joints({"MRKL": _joint("MRKL", 999)}) == [0.0] * 5


class TestJointState:
    def _sdk(self):
        with patch("hal.server.jetson.krabby_mcusdk.FirmwareKrabbyMCUSDK", Mock()):
            sdk = KrabbyMCUSDK(mcu_joints=KRABBY_HEX_DEFINITION.get_mcu_joints(), auto_connect=False)
        sdk._mcu.joints = {}
        return sdk

    def test_normalized_to_rad_inverts_command_mapping(self):
        for rad in (-JOINT_LIMIT_RAD, -0.2, 0.0, 0.3, JOINT_LIMIT_RAD):
            n = _map_mcu_joints_to_normalized({"FL_knee": rad}, ("FL_knee",))["FLKL"]
            assert _normalized_to_rad(n) == pytest.approx(rad)

    def test_only_connected_joints_are_reported(self):
        sdk = self._sdk()
        sdk._mcu.joints = {"FLKL": _joint("FLKL", 0), "FRKL": JointTelemetry(
            name="FRKL", pos=float("nan"), pot=0, current=0, en=(0, 0), pwm=(0, 0), saf=0)}
        positions, velocities = sdk.joint_state(now=1.0)
        assert positions == {"FL_knee": pytest.approx(0.0)}
        assert velocities == {"FL_knee": 0.0}

    def test_velocity_is_smoothed_derivative(self):
        sdk = self._sdk()
        sdk._mcu.joints = {"FLKL": _joint("FLKL", 0)}
        sdk.joint_state(now=0.0)
        sdk._mcu.joints = {"FLKL": JointTelemetry(
            name="FLKL", pos=0.6, pot=0, current=0, en=(0, 0), pwm=(0, 0), saf=0)}
        _, velocities = sdk.joint_state(now=0.1)
        raw = _normalized_to_rad(0.6) / 0.1
        assert velocities["FL_knee"] == pytest.approx(0.2 * raw)

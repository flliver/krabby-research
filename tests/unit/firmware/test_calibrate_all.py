"""Unit tests for the whole-board calibration sequence (M17 Task 3): the SDK
`calibrate_all` (wire `CALL`), the firmware executor wiring, and the local per-leg
sequence it generates. The full 18-joint / multi-board run is chassis-gated; these
guard the buildable logic."""
import re
from pathlib import Path
from unittest.mock import Mock

from firmware.krabby_mcu import KrabbyMCUSDK

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


def _actuator() -> str:
    return (ARDUINO / "actuator_manager.h").read_text()


def _ino() -> str:
    return (ARDUINO / "arduino.ino").read_text()


def _bare_sdk():
    sdk = object.__new__(KrabbyMCUSDK)
    sdk.ser = Mock()
    sdk.ser.is_open = True
    return sdk


class TestCalibrateAllSDK:
    def test_default_sends_call(self):
        sdk = _bare_sdk()
        sdk.calibrate_all()
        sdk.ser.write.assert_called_once_with(b"CALL\n")

    def test_no_yaw_sends_call_noyaw(self):
        sdk = _bare_sdk()
        sdk.calibrate_all(include_yaw=False)
        sdk.ser.write.assert_called_once_with(b"CALL noyaw\n")

    def test_noop_when_closed(self):
        sdk = _bare_sdk()
        sdk.ser.is_open = False
        sdk.calibrate_all()
        sdk.ser.write.assert_not_called()


class TestExecutorFirmware:
    def test_entry_points_exist(self):
        a = _actuator()
        assert re.search(r"void\s+calibrateAll\s*\(", a)
        assert re.search(r"void\s+updateCalibrateAll\s*\(", a)
        assert re.search(r"void\s+buildLocalSequence\s*\(", a)
        assert "enum CalStepOp" in a and "CA_CAL" in a and "CA_MOVE" in a

    def test_updateall_routes_sequence(self, ):
        a = _actuator()
        body = a[a.index("void updateAll"):a.index("void updateAll") + 400]
        assert re.search(r"else if\s*\(\s*caActive\s*\)", body)
        assert "updateCalibrateAll" in body

    def test_halts_and_holds_on_failure(self):
        # 3f: a failed cal step stops the sequence and parks every motor.
        a = _actuator()
        body = a[a.index("void updateCalibrateAll"):a.index("void updateAll")]
        assert "jcLastFailed" in body, "executor must read the per-step cal result"
        assert "caActive = false" in body and "holdAll()" in body

    def test_cal_result_signal_wired(self):
        # calibrateJoint clears jcLastFailed; abort + failing SAVE set it.
        a = _actuator()
        assert "jcLastFailed = false" in a   # cleared at cal start
        assert "jcLastFailed = true" in a    # set on abort (motor_did_not_move / mismatch)
        assert "jcLastFailed = (failCode != nullptr)" in a  # set on a failing SAVE

    def test_move_step_is_closed_loop_with_settle(self):
        # CA_MOVE is the moveJointTo primitive: setTarget + atTarget settle / timeout.
        a = _actuator()
        body = a[a.index("void updateCalibrateAll"):a.index("void updateAll")]
        assert "setTarget" in body and "atTarget" in body
        assert "CA_MOVE_SETTLE_MS" in body and "CA_MOVE_TIMEOUT_MS" in body


class TestLocalSequence:
    def test_standard_seven_step_order(self):
        # §3 order: hip retract, knee retract, knee extend, [yaw left/right/center], hip extend.
        a = _actuator()
        body = a[a.index("void buildLocalSequence"):a.index("void buildLocalSequence") + 1200]
        # the four linear cal steps in order
        order = [m.group(0) for m in re.finditer(
            r"CA_CAL,\s+\w+,\s+CAL_DIR_(RETRACT|EXTEND)", body)]
        # first three linear: hl retract, kl retract, kl extend; last: hl extend
        assert "CAL_DIR_RETRACT" in order[0]      # hip min
        assert order[1].endswith("RETRACT")       # knee min
        assert order[2].endswith("EXTEND")        # knee max
        assert order[-1].endswith("EXTEND")       # hip max (step 7)

    def test_yaw_steps_gated_on_includeyaw(self):
        a = _actuator()
        body = a[a.index("void buildLocalSequence"):a.index("void buildLocalSequence") + 1200]
        assert "if (includeYaw)" in body, "yaw steps must be skippable for the bench fallback"

    def test_returns_leg_to_neutral(self):
        a = _actuator()
        body = a[a.index("void buildLocalSequence"):a.index("void buildLocalSequence") + 1200]
        assert body.count("CA_MOVE") >= 2, "each leg returns hip+knee to mid-travel"


class TestCallWireDispatch:
    def test_call_distinguished_from_legacy_c(self):
        ino = _ino()
        body = ino[ino.index("cmdType == 'C'"):ino.index("cmdType == 'C'") + 700]
        assert 'startsWith("CALL")' in body, "CALL must be split out from the legacy C autocal"
        assert "calibrateAll" in body
        assert 'indexOf("noyaw")' in body, "noyaw arg selects the yaw-skipping run"

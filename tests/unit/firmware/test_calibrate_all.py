"""Unit tests for the whole-board calibration sequence (M17 Task 3): the SDK
`calibrate_all` (wire `CALL`), the firmware executor wiring, and the local per-leg
sequence it generates. The full 18-joint / multi-board run is chassis-gated; these
guard the buildable logic."""
import re
from pathlib import Path
from unittest.mock import Mock

import time

from firmware.krabby_mcu import KrabbyMCUSDK, parse_cal_line, parse_cal_reply

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

    def test_skip_validation_sends_skipval(self):
        sdk = _bare_sdk()
        sdk.calibrate_all(validate=False)
        sdk.ser.write.assert_called_once_with(b"CALL skipval\n")

    def test_no_yaw_and_skip_validation(self):
        sdk = _bare_sdk()
        sdk.calibrate_all(include_yaw=False, validate=False)
        sdk.ser.write.assert_called_once_with(b"CALL noyaw skipval\n")

    def test_validate_current_sense_sends_valonly(self):
        sdk = _bare_sdk()
        sdk.validate_current_sense()
        sdk.ser.write.assert_called_once_with(b"CALL valonly\n")

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


class TestCurrentSenseValidation:
    """M17 Task 4: neutral pose (4b) + per-leg current-sense lift validation (4c/4d/4e)."""

    def test_new_step_ops(self):
        a = _actuator()
        for op in ("CA_NEUTRAL", "CA_RECORD_UNLOADED", "CA_EVAL"):
            assert op in a, op

    def test_validate_entry_points(self):
        a = _actuator()
        assert re.search(r"void\s+validateCurrentSense\s*\(", a)
        assert re.search(r"void\s+appendNeutralAndValidation\s*\(", a)
        # calibrateAll gained the validate flag
        assert re.search(r"void\s+calibrateAll\s*\(\s*bool\s+includeYaw[^)]*bool\s+validate", a)

    def test_neutral_pose_targets_only_full_joints(self):
        # 4b: every calibrated joint -> 0.5; uncalibrated joints (e.g. an unwired yaw) skipped.
        a = _actuator()
        body = a[a.index("bool neutral = (s.op == CA_NEUTRAL)"):a.index("bool neutral") + 900]
        assert "CAL_STATE_FULL" in body and "setTarget" in body

    def test_lift_sequence_shape(self):
        # 4c: lift hip (unload) -> record -> extend knee (load) -> eval -> lower.
        a = _actuator()
        body = a[a.index("void appendNeutralAndValidation"):
                 a.index("void appendNeutralAndValidation") + 900]
        assert "CA_NEUTRAL" in body
        assert "CA_RECORD_UNLOADED" in body and "CA_EVAL" in body
        assert "90" in body and "100" in body  # lift hip to 0.9, load knee to 1.0

    def test_eval_emits_current_sense_codes(self):
        # 4d: no delta -> no_signal; below floor / above ceiling -> no_spike.
        a = _actuator()
        body = a[a.index("void evalCurrentSense"):a.index("void evalCurrentSense") + 500]
        assert "current_sense_no_signal" in body and "current_sense_no_spike" in body
        assert "avgIS" not in body  # reads come from the caller; eval works on the values
        assert "800" in body and "100" in body  # ceiling + spike floor (inline literals, 4d)


class TestCallWireDispatch:
    def test_call_distinguished_from_legacy_c(self):
        ino = _ino()
        body = ino[ino.index("cmdType == 'C'"):ino.index("cmdType == 'C'") + 900]
        assert 'startsWith("CALL")' in body, "CALL must be split out from the legacy C autocal"
        assert "calibrateAll" in body
        assert 'indexOf("noyaw")' in body, "noyaw arg selects the yaw-skipping run"
        assert 'indexOf("skipval")' in body, "skipval arg skips Task-4 validation"
        assert 'indexOf("valonly")' in body and "validateCurrentSense" in body


class TestParseCalLine:
    """Task 4 §4: the positional whole-board GET-calibration line parser."""

    def test_positional_parse(self):
        assert parse_cal_line("CAL FLHL POT 200 820 1") == (
            "FLHL", {"type": "POT", "min": "200", "max": "820", "reversed": "1"})

    def test_hall_parse(self):
        assert parse_cal_line("CAL FLHY HALL -512 511 0") == (
            "FLHY", {"type": "HALL", "min": "-512", "max": "511", "reversed": "0"})

    def test_rejects_labelled_q_reply(self):
        # the richer per-joint Q reply has 'type' as its 3rd token — not this format
        assert parse_cal_line("CAL FLHL type POT rev 0 min 200 max 820 cal 1 state FULL") is None

    def test_rejects_wrong_arity(self):
        assert parse_cal_line("CAL FLHL POT 200 820") is None
        assert parse_cal_line("VER 1 a b") is None

    def test_parsers_dont_collide(self):
        # the Q parser and the whole-board parser recognize disjoint shapes
        q = "CAL FLHL type POT rev 0 min 200 max 820 cal 1 state FULL"
        board = "CAL FLHL POT 200 820 1"
        assert parse_cal_reply(q) is not None and parse_cal_line(q) is None
        assert parse_cal_line(board) is not None


class TestGetAllCalibrationSDK:
    def _sdk(self):
        sdk = object.__new__(KrabbyMCUSDK)
        sdk.ser = Mock()
        sdk.ser.is_open = True
        sdk._board_cals = {}
        return sdk

    def test_front_sends_get_calibration(self):
        sdk = self._sdk()
        sdk.get_all_calibration(timeout=0.05)
        sdk.ser.write.assert_called_once_with(b"GET calibration\n")

    def test_left_sends_get_left(self):
        sdk = self._sdk()
        sdk.get_all_calibration(board="left", timeout=0.05)
        sdk.ser.write.assert_called_once_with(b"GET_LEFT calibration\n")

    def test_right_sends_get_right(self):
        sdk = self._sdk()
        sdk.get_all_calibration(board="right", timeout=0.05)
        sdk.ser.write.assert_called_once_with(b"GET_RIGHT calibration\n")

    def test_returns_accumulated_board_cals(self):
        sdk = self._sdk()

        def deliver():
            time.sleep(0.02)
            sdk._board_cals = {f"J{i}": {"type": "POT"} for i in range(6)}

        import threading
        t = threading.Thread(target=deliver)
        t.start()
        out = sdk.get_all_calibration(timeout=0.5)
        t.join()
        assert len(out) == 6


class TestGetCalibrationFirmware:
    def test_printallcal_positional_format(self):
        a = _actuator()
        body = a[a.index("void printAllCal"):a.index("void printAllCal") + 700]
        # positional, leaner than printJointCal: no "type"/"rev"/"cal"/"state" labels
        assert '"CAL "' in body and "sensorReversed" in body
        assert '" type"' not in body and '" cal "' not in body

    def test_get_calibration_wired_in_config(self):
        ino = _ino()
        # local dump + follower forwarding both special-case the "calibration" key
        assert ino.count('payload == "calibration"') >= 2
        assert "printAllCal" in ino

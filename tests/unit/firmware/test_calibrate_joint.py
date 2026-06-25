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


class TestDirectionalCalSDK:
    """M17 Task 2: `K <name> <direction>` — linear joints take extend/retract, yaw
    joints (name ends in 'Y') take left/right. A mismatch is a client bug → raise."""

    def test_linear_extend_retract_wire(self):
        for d in ("extend", "retract"):
            sdk = _bare_sdk()
            sdk.calibrate_joint("FLHL", d)  # FLHL = linear
            sdk.ser.write.assert_called_once_with(f"KFLHL {d}\n".encode())

    def test_yaw_left_right_wire(self):
        for d in ("left", "right"):
            sdk = _bare_sdk()
            sdk.calibrate_joint("FLHY", d)  # FLHY = yaw
            sdk.ser.write.assert_called_once_with(f"KFLHY {d}\n".encode())

    def test_none_is_full_sweep_no_token(self):
        sdk = _bare_sdk()
        sdk.calibrate_joint("FLHL", None)
        sdk.ser.write.assert_called_once_with(b"KFLHL\n")

    def test_extend_rejected_on_yaw(self):
        sdk = _bare_sdk()
        with pytest.raises(ValueError, match="yaw joint"):
            sdk.calibrate_joint("FLHY", "extend")
        sdk.ser.write.assert_not_called()

    def test_left_rejected_on_linear(self):
        sdk = _bare_sdk()
        with pytest.raises(ValueError, match="linear joint"):
            sdk.calibrate_joint("FLKL", "left")
        sdk.ser.write.assert_not_called()

    def test_unknown_direction_rejected(self):
        sdk = _bare_sdk()
        with pytest.raises(ValueError):
            sdk.calibrate_joint("FLHL", "sideways")

    def test_every_joint_typed_correctly(self):
        # knees/hip-lifts are linear (extend/retract), hip-yaws are yaw (left/right)
        for name in ALL_JOINT_NAMES:
            is_yaw = name[3] == "Y"
            ok, bad = (("left", "extend") if is_yaw else ("extend", "left"))
            assert KrabbyMCUSDK._validate_cal_direction(name, ok) == ok
            with pytest.raises(ValueError):
                KrabbyMCUSDK._validate_cal_direction(name, bad)


class TestCalibrateJointCLI:
    def test_rejects_unknown_joint_before_connecting(self):
        # exits at client-side validation, before any port is opened
        with pytest.raises(SystemExit):
            cli_mod.cmd_calibrate_joint(None, "BOGUS")

    def test_rejects_bad_direction_before_connecting(self):
        with pytest.raises(SystemExit):
            cli_mod.cmd_calibrate_joint(None, "FLHY", "extend")  # extend on a yaw joint

    def test_full_sweep_does_not_break_on_initial_uncal(self, monkeypatch):
        # Regression: a never-calibrated joint reports UNCAL from the start, so the poll
        # must NOT treat UNCAL as "done" — it has to wait for FULL (or an ERR). Otherwise
        # the call returns the stale pre-cal read while the sweep is still running.
        states = ["UNCAL", "UNCAL", "UNCAL", "FULL"]  # what get_calibration yields over time
        calls = {"get_cal": 0}

        class FakeSDK:
            port = "/dev/fake"
            _validate_cal_direction = staticmethod(KrabbyMCUSDK._validate_cal_direction)
            def __init__(self, *a, **k): pass
            def connect(self, *a, **k): return True
            def clear_errors(self): pass
            def calibrate_joint(self, name, direction=None): pass
            def get_errors(self): return []
            def get_calibration(self, name, timeout=1.0):
                i = min(calls["get_cal"], len(states) - 1)
                calls["get_cal"] += 1
                return {"type": "HALL", "rev": "0", "min": "0", "max": "1075",
                        "cal": "1", "state": states[i]}
            def close(self): pass

        monkeypatch.setattr(cli_mod, "KrabbyMCUSDK", FakeSDK)
        monkeypatch.setattr(cli_mod.time, "sleep", lambda *_: None)
        cli_mod.cmd_calibrate_joint(None, "FLKL")  # fresh joint → boots UNCAL
        assert calls["get_cal"] >= 4, "poll broke early on the initial UNCAL instead of waiting for FULL"


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

    def test_partial_shows_state_and_anchor_hint(self):
        s = cli_mod._format_cal("FLHL", {"type": "HALL", "rev": "0", "min": "0",
                                         "max": "1077", "cal": "1", "state": "PARTIAL"})
        assert "state=PARTIAL" in s and "end-stop" in s

    def test_full_state_shown(self):
        s = cli_mod._format_cal("FLHL", {"type": "HALL", "rev": "0", "min": "0",
                                         "max": "1077", "cal": "1", "state": "FULL"})
        assert "state=FULL" in s


class TestSelfHeal:
    """M17 Task 2 §6.5 / 2h: Hall boot-state + end-stop self-heal anchor."""

    def test_boot_state_split(self, actuator):
        body = actuator[actuator.index("void applyJointCal"):]
        assert "SENSOR_POT || liveFrame" in body, "pot/fresh-cal → FULL; EEPROM-loaded Hall → PARTIAL"
        assert "CAL_STATE_PARTIAL" in body

    def test_self_heal_snaps_offset_on_partial_stall(self, actuator):
        body = actuator[actuator.index("void update()"):actuator.index("void update()") + 2000]
        assert "CAL_STATE_PARTIAL" in body and "hallOffset" in body
        assert "CAL_STATE_FULL" in body, "anchoring must flip the joint to FULLY_CALIBRATED"

    def test_getrawpos_applies_offset(self, actuator):
        body = actuator[actuator.index("int32_t getRawPos"):actuator.index("int32_t getRawPos") + 300]
        assert "hallOffset" in body, "Hall getRawPos must add the anchor offset"

    def test_partial_joint_rejects_position_targets(self, actuator):
        body = actuator[actuator.index("void applyCommands"):]
        assert "CAL_STATE_PARTIAL" in body and "not_calibrated" in body

    def test_state_in_cal_readback(self, actuator):
        body = actuator[actuator.index("void printJointCal"):]
        assert "calStateName" in body, "the CAL reply must include the runtime state"


class TestRangeCheckFirmware:
    def test_span_check_gates_calibrated_flag(self, actuator):
        body = actuator[actuator.index("case JC_SAVE"):]
        assert "JC_POT_MIN_SPAN" in body, "JC_SAVE must check the swept span"
        assert "pot_value_invalid" in body, "a too-small pot span must emit pot_value_invalid"
        assert re.search(r"calibrated\s*=\s*ok\s*\?\s*1\s*:\s*0", body)

    def test_query_command_dispatched(self, actuator, ino):
        assert "queryCalByName" in actuator and "printJointCal" in actuator
        assert re.search(r"cmdType\s*==\s*'Q'", ino), "loop() must dispatch the Q (read cal) command"

    def test_hall_detect_enabled_and_checked_first(self, actuator):
        assert re.search(r"JC_HALL_DETECT\s*=\s*true", actuator), \
            "Hall auto-detect must be on now that quadrature is real"
        # In jcEvalNudge, the HALL branch must come before the POT branch (shared A1 pin).
        start = actuator.index("bool jcEvalNudge")
        body = actuator[start:actuator.index("void jcApplyDetectedSensor", start)]
        assert body.index("SENSOR_HALL") < body.index("SENSOR_POT"), \
            "nudge must check the signed Hall count before the pot (avgPot carries HallB)"


class TestHallDriftCheck:
    """M17 Task 2 §2c: Hall joints sweep retract→extend→retract; the two retract
    counts must agree within JC_HALL_DRIFT_TOL or cal fails with hall_drift."""

    def test_repeat_retract_state_exists(self, actuator):
        assert "JC_RETRACT_AGAIN" in actuator, "a second retract state is needed for the drift check"
        # JC_EXTEND must route Hall joints to the repeat retract, pots straight to save.
        ext = actuator[actuator.index("case JC_EXTEND"):actuator.index("case JC_RETRACT_AGAIN")]
        assert "SENSOR_HALL" in ext and "JC_RETRACT_AGAIN" in ext, \
            "JC_EXTEND must send Hall joints to JC_RETRACT_AGAIN"

    def test_pot_skips_repeat_sweep(self, actuator):
        # A pot's absolute reading needs no repeat: the non-Hall branch goes straight to save.
        ext = actuator[actuator.index("case JC_EXTEND"):actuator.index("case JC_RETRACT_AGAIN")]
        assert re.search(r"else\s*{\s*\n\s*jcState\s*=\s*JC_SAVE", ext), \
            "pot joints must skip the repeat retract and save directly"

    def test_second_min_recorded_and_compared(self, actuator):
        again = actuator[actuator.index("case JC_RETRACT_AGAIN"):actuator.index("case JC_SAVE")]
        assert "jcHallMin2" in again, "the repeat retract must record hallMin_2"
        save = actuator[actuator.index("case JC_SAVE"):]
        assert "JC_HALL_DRIFT_TOL" in save, "JC_SAVE must compare against the drift tolerance"
        assert "hall_drift" in save, "a too-large drift must emit hall_drift"

    def test_tolerance_constant_defined(self, actuator):
        assert re.search(r"JC_HALL_DRIFT_TOL\s*=\s*\d+", actuator)


class TestDirectionalCalFirmware:
    """M17 Task 2: `calibrateJoint(idx, dir)` sweeps one end (RETRACT/EXTEND) and parks
    there; the firmware enforces the linear↔extend/retract, yaw↔left/right pairing."""

    def test_direction_enum_and_param(self, actuator):
        assert "enum CalDirection" in actuator
        assert re.search(r"CAL_DIR_NONE|CAL_DIR_RETRACT|CAL_DIR_EXTEND", actuator)
        assert re.search(r"calibrateJoint\s*\(\s*uint8_t\s+idx,\s*CalDirection\s+dir", actuator)

    def test_parse_enforces_joint_type_pairing(self, actuator):
        body = actuator[actuator.index("bool parseCalDirection"):
                        actuator.index("bool parseCalDirection") + 700]
        # yaw detection (4th char 'Y'), and each token gated on yaw vs linear
        assert "charAt(3) == 'Y'" in body
        assert '"retract"' in body and '"extend"' in body
        assert '"left"' in body and '"right"' in body

    def test_byname_passes_direction(self, actuator):
        assert re.search(r"calibrateJointByName\s*\(\s*const String&\s+name,\s*const String&\s+dir", actuator)
        body = actuator[actuator.index("void calibrateJointByName"):
                        actuator.index("void calibrateJointByName") + 600]
        assert "parseCalDirection" in body and "return" in body  # bad pairing → silent drop

    def test_retract_only_skips_extend(self, actuator):
        ret = actuator[actuator.index("case JC_RETRACT:"):actuator.index("case JC_EXTEND:")]
        assert "CAL_DIR_RETRACT" in ret and "JC_SAVE" in ret, \
            "a retract-only cal must save after the single stroke, not continue to extend"

    def test_extend_only_short_circuits_to_save(self, actuator):
        ext = actuator[actuator.index("case JC_EXTEND:"):actuator.index("case JC_RETRACT_AGAIN")]
        assert "CAL_DIR_NONE" in ext and "JC_SAVE" in ext, \
            "a directional extend must save immediately, not do the drift repeat"

    def test_extend_only_starts_at_extend(self, actuator):
        body = actuator[actuator.index("void jcBeginSweep"):
                        actuator.index("void jcBeginSweep") + 400]
        assert "CAL_DIR_EXTEND" in body and "JC_EXTEND" in body

    def test_save_merges_one_end_and_gates_on_both(self, actuator):
        save = actuator[actuator.index("case JC_SAVE"):]
        assert "endsRecorded" in save, "JC_SAVE must record which ends are present"
        assert "JOINTCAL_END_MIN" in save and "JOINTCAL_END_MAX" in save
        assert "bothEnds" in save, "calibrated must require both ends recorded"

    def test_endsrecorded_field_exists(self):
        layout = (ARDUINO / "eeprom_layout.h").read_text()
        assert "endsRecorded" in layout
        assert "JOINTCAL_END_MIN" in layout and "JOINTCAL_END_MAX" in layout

    def test_k_command_parses_direction_token(self, ino):
        body = ino[ino.index("cmdType == 'K'"):ino.index("cmdType == 'K'") + 700]
        assert "indexOf(' ')" in body, "K must split name from the optional direction token"
        assert "calibrateJointByName" in body

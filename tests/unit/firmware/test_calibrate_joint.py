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


class TestSensorTypeMapping:
    """The fixed per-joint sensor map in the actuator table: every KL (knee) is a
    potentiometer, every HL (hip-lift) and HY (hip-yaw) a Hall. One leg therefore has
    exactly one pot and two Halls; two pots or two Halls on HL+KL is a wiring error,
    never something calibration should discover."""

    def _typed_decls(self, ino):
        return re.findall(
            r'LinearActuator\s+\w+\("(\w+)"[^;]*?,\s*(SENSOR_HALL|SENSOR_POT)\)', ino)

    def test_every_joint_has_an_explicit_type(self, ino):
        decls = self._typed_decls(ino)
        assert len(decls) == 18, f"expected 18 explicitly-typed actuators, found {len(decls)}"

    def test_knees_are_pot_hips_are_hall(self, ino):
        for name, stype in self._typed_decls(ino):
            expected = "SENSOR_POT" if name.endswith("KL") else "SENSOR_HALL"
            assert stype == expected, f"{name} should be {expected}, declared {stype}"

    def test_each_leg_has_one_pot_two_halls(self, ino):
        types = dict(self._typed_decls(ino))
        # group by leg prefix (first 2 chars, e.g. FL, FR, RL, ...)
        legs = {}
        for name, stype in types.items():
            legs.setdefault(name[:2], []).append(stype)
        for leg, stypes in legs.items():
            assert stypes.count("SENSOR_POT") == 1, f"{leg} must have exactly one pot (the knee)"
            assert stypes.count("SENSOR_HALL") == 2, f"{leg} must have exactly two Halls (hip-yaw, hip-lift)"


class TestSensorTypeMismatchDetection:
    """Calibration must flag a slot whose physical sensor disagrees with its fixed type
    (e.g. a Hall actuator in the pot knee slot — how "two Halls on a leg" happens). The
    HallA signed count is the discriminator: it only moves when a Hall is present."""

    def test_mismatch_code_emitted(self, actuator):
        assert "sensor_type_mismatch" in actuator, \
            "the cal nudge must emit sensor_type_mismatch on a wrong-sensor slot"

    def test_nudge_cross_checks_both_sensors(self, actuator):
        body = actuator[actuator.index("bool jcEvalNudge"):actuator.index("void jcAbortCal")]
        # reads BOTH the Hall count and the pot pin, and uses the Hall count as the tell
        assert "hallSignedCount()" in body and "analogRead(a->pinPot)" in body
        assert "jcMismatch = true" in body, "a wrong sensor must raise jcMismatch"

    def test_pot_joint_flags_unexpected_hall(self, actuator):
        # In the SENSOR_POT branch, Hall movement (a Hall actuator present) → mismatch, but
        # gated on the higher present-threshold so a floating HallA pin's EMI doesn't false-trip.
        body = actuator[actuator.index("bool jcEvalNudge"):actuator.index("void jcAbortCal")]
        pot_branch = body[body.index("else {"):]  # the SENSOR_POT arm
        assert "JC_HALL_PRESENT_THRESHOLD" in pot_branch and "jcMismatch = true" in pot_branch

    def test_present_threshold_above_motion_threshold(self, actuator):
        # The "Hall present on a pot joint" bar must be well above the wired-Hall motion bar,
        # else floating-pin EMI false-trips sensor_type_mismatch on a real pot.
        present = int(re.search(r"JC_HALL_PRESENT_THRESHOLD\s*=\s*(\d+)", actuator).group(1))
        motion = int(re.search(r"JC_HALL_NUDGE_THRESHOLD\s*=\s*(\d+)", actuator).group(1))
        assert present > motion, "present-threshold must clear the EMI floor above the motion bar"

    def test_baseline_reads_pot_directly(self, actuator):
        # jcPotBefore must be a fresh analogRead (Hall joints skip avgPot in normal op),
        # else the cross-check has no valid pot baseline.
        start = actuator.index("void calibrateJoint(uint8_t idx")
        body = actuator[start:start + 700]
        assert re.search(r"jcPotBefore\s*=\s*analogRead", body), \
            "cal must baseline the pot pin with a direct read for the cross-check"

    def test_both_nudge_evals_abort_on_mismatch(self, actuator):
        # the nudge evals live between updateJointCal and the first sweep case (JC_RETRACT)
        start = actuator.index("void updateJointCal")
        body = actuator[start:actuator.index("case JC_RETRACT:", start)]
        # both the forward and reverse nudge evals route a mismatch to the abort
        assert body.count('jcAbortCal(a, "sensor_type_mismatch")') >= 2


class TestPotSweepLiveSensors:
    """The pot sweep reads avgPot, which the normal update() loop maintains — but that loop
    is bypassed during cal. updateJointCal must refresh sensors itself or the pot stays
    frozen and the sweep records min==max (span 0)."""

    def test_cal_refreshes_sensors(self, actuator):
        body = actuator[actuator.index("void updateJointCal"):
                        actuator.index("void updateJointCal") + 600]
        assert "updateSensors()" in body, \
            "updateJointCal must refresh the joint's sensors (else avgPot is frozen all sweep)"


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

    def test_reversed_pot_shows_absolute_span(self):
        # A reversed pot reads high→low as it extends, so max < min and the raw diff is
        # negative; span must display the magnitude, not "-822".
        s = cli_mod._format_cal("FLKL", {"type": "POT", "rev": "1", "min": "971", "max": "149", "cal": "1"})
        assert "span=822" in s and "span=-822" not in s

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


class TestPositionTargetScale:
    """M17 Task 3 foundation: a normalized [0,1] position target must map onto the
    CALIBRATED travel (the same flip-corrected frame getPos()/getRawPos() report), not the
    raw 0-1023 minStop/maxStop scale — else closed-loop position ignores calibration."""

    def test_settarget_uses_calibrated_endpoints(self, actuator):
        body = actuator[actuator.index("void setTarget"):actuator.index("void setTarget") + 700]
        assert "calValid" in body, "setTarget must branch on whether the joint is calibrated"
        # calibrated branch maps via the same applyFlip(calMin/calMax) lo/hi as getPos()
        assert "applyFlip" in body and "calHallMin" in body and "calPotMin" in body, \
            "calibrated target must use the flip-corrected cal endpoints"
        assert "minStop" in body, "uncalibrated joints keep the legacy raw-ADC fallback"

    def test_target_and_error_are_32bit(self, actuator):
        # Hall counts are signed and can exceed int16 — target/error must be int32_t.
        assert re.search(r"int32_t\s+currentTarget", actuator)
        assert re.search(r"int32_t\s+error\s*=\s*currentTarget", actuator)

    def test_attarget_settle_helper(self, actuator):
        body = actuator[actuator.index("bool atTarget"):actuator.index("bool atTarget") + 200]
        assert "getPos()" in body and "lastSetVal" in body, \
            "atTarget compares normalized getPos() against the last setTarget value"


class TestRangeCheckFirmware:
    def test_span_check_gates_calibrated_flag(self, actuator):
        body = actuator[actuator.index("case JC_SAVE"):]
        assert "JC_POT_MIN_SPAN" in body, "JC_SAVE must check the swept span"
        assert "pot_value_invalid" in body, "a too-small pot span must emit pot_value_invalid"
        assert re.search(r"calibrated\s*=\s*ok\s*\?\s*1\s*:\s*0", body)

    def test_query_command_dispatched(self, actuator, ino):
        assert "queryCalByName" in actuator and "printJointCal" in actuator
        assert re.search(r"cmdType\s*==\s*'Q'", ino), "loop() must dispatch the Q (read cal) command"

    def test_cal_verifies_fixed_sensor_type_not_autodetect(self, actuator):
        # Sensor type is a FIXED per-joint property. Calibration seeds jcSensorType from the
        # actuator and the nudge only VERIFIES the expected sensor moved — it never guesses
        # POT vs HALL. (Guessing on a bench with two Hall actuators is what mis-typed the
        # knee as Hall and produced the "two Halls on a leg" calibration.)
        assert "JC_HALL_DETECT" not in actuator, "runtime sensor-type auto-detect must be removed"
        assert re.search(r"jcSensorType\s*=\s*\(SensorType\)\s*actuators\[idx\]->sensorType", actuator), \
            "calibrateJoint must seed jcSensorType from the joint's fixed type, not default to POT"
        # jcEvalNudge reads only the joint's expected sensor (branch on jcSensorType).
        start = actuator.index("bool jcEvalNudge")
        body = actuator[start:actuator.index("void jcApplyNudgeResult", start)]
        assert "jcSensorType == SENSOR_HALL" in body, \
            "nudge must read the joint's fixed sensor, not probe Hall-then-pot"

    def test_applyjointcal_rejects_mismatched_sensor_type(self, actuator):
        # A stored cal whose sensorType disagrees with the compiled-in type is stale
        # (e.g. an old two-Hall cal on a knee) and must be rejected, not trusted.
        body = actuator[actuator.index("void applyJointCal"):
                        actuator.index("void applyJointCal") + 900]
        assert "jc.sensorType != sensorType" in body, \
            "applyJointCal must reject a cal recorded against a different sensor type"


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


class TestRuntimeHealth2f:
    """M17 Task 2 §2f: continuous runtime sensor-health monitoring — a joint driven
    during NORMAL operation whose sensor stops following emits one throttled ERR."""

    def test_monitor_method_exists(self, actuator):
        assert "checkRuntimeHealth" in actuator
        body = actuator[actuator.index("const char* checkRuntimeHealth"):]
        body = body[:body.index("\n    }")]
        assert "motor_did_not_move" in body and "motor_jammed" in body

    def test_wired_into_normal_update_loop_only(self, actuator):
        loop = actuator[actuator.index("void updateAll"):actuator.index("void updateAll") + 800]
        # runs in the else (normal) branch, NOT during jcActive cal
        assert "checkRuntimeHealth" in loop and "emitJointErr" in loop
        assert loop.index("jcActive") < loop.index("checkRuntimeHealth")

    def test_throttled_one_per_event(self, actuator):
        body = actuator[actuator.index("const char* checkRuntimeHealth"):]
        body = body[:body.index("\n    }")]
        assert "healthErrSent" in body, "must throttle to one ERR per stall event"
        # re-arms when not driven or moving again
        assert "currentPwm == 0" in body and "isStalled" in body

    def test_current_split_jam_vs_no_move(self, actuator):
        body = actuator[actuator.index("const char* checkRuntimeHealth"):]
        body = body[:body.index("\n    }")]
        assert re.search(r"avgIS\s*>=\s*JAM_CURRENT_THRESHOLD", body), \
            "high current while pinned = motor_jammed, else motor_did_not_move"

    def test_end_stops_suppressed(self, actuator):
        body = actuator[actuator.index("const char* checkRuntimeHealth"):]
        body = body[:body.index("\n    }")]
        # PARTIAL self-heals; FULL-at-a-known-limit is expected travel, not a fault
        assert "CAL_STATE_PARTIAL" in body and "CAL_STATE_FULL" in body
        assert "HEALTH_AT_LIMIT_EPS" in body

    def test_health_window_after_self_heal(self, actuator):
        # the health stall window must be longer than self-heal's so a PARTIAL joint
        # anchors (→FULL) before the monitor would consider it a fault
        assert "HEALTH_STALL_MS" in actuator and "SELF_HEAL_STALL_MS" in actuator

"""Host-side calibration support in KrabbyMCUSDK: command building, CAL line capture."""

import logging
import math
import re
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("serial")

from firmware import krabby_mcu
from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.krabby_mcu import (
    CAL_DEFAULT_PWM,
    CAL_PWM_MAX,
    CAL_PWM_MIN,
    KrabbyMCUSDK,
    build_calibrate_command,
    cal_line_state,
    cal_robot_finished,
    cal_status_line,
    curl_button,
    curl_pwm_sign,
    is_hip_or_knee,
    parse_calibration_pwm,
    validate_calibration_pwm,
)

CONSTANTS_HEADER = (
    Path(__file__).resolve().parents[3]
    / "firmware"
    / "arduino"
    / "src"
    / "actuator"
    / "calibration_constants.h"
)

TELEMETRY_LINE = "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"


def _sdk(open_port: bool = True) -> KrabbyMCUSDK:
    sdk = KrabbyMCUSDK(target=Mock())
    sdk.ser = Mock()
    sdk.ser.is_open = open_port
    return sdk


def _firmware_constant(name: str) -> int:
    text = CONSTANTS_HEADER.read_text(encoding="utf-8")
    match = re.search(
        rf"static constexpr\s+\w+\s+{name}\s*=\s*([+-]?\d+)", text
    )
    assert match, f"{name} not found in {CONSTANTS_HEADER}"
    return int(match.group(1))


class TestConstantsMatchFirmware:
    """The Python mirrors must not drift from calibration_constants.h."""

    @pytest.mark.skipif(not CONSTANTS_HEADER.exists(), reason="firmware tree not present")
    @pytest.mark.parametrize(
        "name,value",
        [
            ("CAL_PWM_MIN", CAL_PWM_MIN),
            ("CAL_PWM_MAX", CAL_PWM_MAX),
            ("CAL_DEFAULT_PWM", CAL_DEFAULT_PWM),
            ("CAL_HIP_CURL_PWM_SIGN", krabby_mcu.CAL_HIP_CURL_PWM_SIGN),
            ("CAL_KNEE_CURL_PWM_SIGN", krabby_mcu.CAL_KNEE_CURL_PWM_SIGN),
            ("CAL_HIP_SWEEP_EDGES", krabby_mcu.CAL_HIP_SWEEP_EDGES),
            ("CAL_HIP_STROKE_EDGES", krabby_mcu.CAL_HIP_STROKE_EDGES),
            ("CAL_KNEE_PARK_EDGES", krabby_mcu.CAL_KNEE_PARK_EDGES),
            ("CAL_KNEE_PARK_BAND_MIN_EDGES", krabby_mcu.CAL_KNEE_PARK_BAND_MIN_EDGES),
            ("CAL_KNEE_PARK_BAND_MAX_EDGES", krabby_mcu.CAL_KNEE_PARK_BAND_MAX_EDGES),
            ("CAL_KNEE_STROKE_EDGES", krabby_mcu.CAL_KNEE_STROKE_EDGES),
            ("CAL_KNEE_STROKE_SHORT_MAX_EDGES", krabby_mcu.CAL_KNEE_STROKE_SHORT_MAX_EDGES),
            ("CAL_HIP_NUDGE_EDGES", krabby_mcu.CAL_HIP_NUDGE_EDGES),
            ("CAL_KNEE_NUDGE_EDGES", krabby_mcu.CAL_KNEE_NUDGE_EDGES),
            ("CAL_HALL_MIN_STROKE_EDGES", krabby_mcu.CAL_HALL_MIN_STROKE_EDGES),
            ("CAL_HALL_MAX_STROKE_EDGES", krabby_mcu.CAL_HALL_MAX_STROKE_EDGES),
        ],
    )
    def test_value_matches_header(self, name, value):
        assert _firmware_constant(name) == value

    def test_known_values(self):
        # PWM 80 stalls mid-stroke with no current signature (owner decision F4).
        assert (CAL_PWM_MIN, CAL_PWM_MAX, CAL_DEFAULT_PWM) == (110, 150, 128)

    def test_table_safe_sweep_values(self):
        # FL's values, used for all six legs: park 264 (band 244-280), sweep 840,
        # derived stroke 1069. The sweep must stay short of the stroke.
        assert krabby_mcu.CAL_KNEE_PARK_EDGES == 264
        assert (
            krabby_mcu.CAL_KNEE_PARK_BAND_MIN_EDGES,
            krabby_mcu.CAL_KNEE_PARK_BAND_MAX_EDGES,
        ) == (244, 280)
        assert krabby_mcu.CAL_HIP_SWEEP_EDGES == 840
        assert krabby_mcu.CAL_HIP_STROKE_EDGES == 1069
        assert (
            krabby_mcu.CAL_KNEE_PARK_BAND_MIN_EDGES
            < krabby_mcu.CAL_KNEE_PARK_EDGES
            < krabby_mcu.CAL_KNEE_PARK_BAND_MAX_EDGES
        )
        assert krabby_mcu.CAL_HIP_SWEEP_EDGES < krabby_mcu.CAL_HIP_STROKE_EDGES
        # F1: a knee stroke under 761 (785 less 24) fails before its park.
        assert (krabby_mcu.CAL_KNEE_STROKE_EDGES, krabby_mcu.CAL_KNEE_STROKE_MIN_EDGES) == (785, 761)


class TestCurlMapping:
    def test_hip_curls_on_retract_knee_on_extend(self):
        assert curl_pwm_sign("FLHL") == -1
        assert curl_button("FLHL") == "retract"
        assert curl_pwm_sign("FLKL") == +1
        assert curl_button("FLKL") == "extend"

    def test_same_on_every_leg(self):
        for _, names in krabby_mcu.JOINT_GROUP_NAMES:
            for name in names:
                if name.endswith("HL"):
                    assert curl_button(name) == "retract", name
                elif name.endswith("KL"):
                    assert curl_button(name) == "extend", name

    def test_yaw_has_no_curl_direction(self):
        for _, names in krabby_mcu.JOINT_GROUP_NAMES:
            for name in names:
                if name.endswith("HY"):
                    assert curl_pwm_sign(name) is None
                    assert curl_button(name) is None

    def test_unknown_name(self):
        assert curl_button("") is None
        assert curl_button("X") is None


class TestValidateCalibrationPwm:
    @pytest.mark.parametrize("pwm", [110, 128, 150])
    def test_accepts_in_range(self, pwm):
        assert validate_calibration_pwm(pwm) == pwm

    @pytest.mark.parametrize("pwm", [109, 80, 40, 39, 151, 0, -80, 255, 10**9])
    def test_rejects_out_of_range(self, pwm):
        with pytest.raises(ValueError):
            validate_calibration_pwm(pwm)

    @pytest.mark.parametrize("pwm", ["120", 120.0, 120.5, None, True, False, [120]])
    def test_rejects_non_integers(self, pwm):
        with pytest.raises(ValueError):
            validate_calibration_pwm(pwm)


class TestParseCalibrationPwm:
    @pytest.mark.parametrize("text,expected", [("110", 110), (" 128 ", 128), ("150", 150)])
    def test_ok(self, text, expected):
        assert parse_calibration_pwm(text) == expected

    @pytest.mark.parametrize(
        "text", ["", " ", "abc", "8 0", "-80", "+80", "80.0", "80", "109", "151", "0x50", "٨٠"]
    )
    def test_rejects(self, text):
        with pytest.raises(ValueError):
            parse_calibration_pwm(text)


class TestBuildCalibrateCommand:
    @pytest.mark.parametrize(
        "pwm,passes,expected",
        [
            (None, None, "C"),
            (110, None, "C 110"),
            (None, "SEEK", "C SEEK"),
            (None, "SWEEP", "C SWEEP"),
            (120, "SEEK", "C SEEK 120"),
            (150, "SWEEP", "C SWEEP 150"),
            (110, "seek", "C SEEK 110"),
            (140, ("SEEK", "SWEEP"), "C 140"),
            (140, ["SWEEP", "SEEK"], "C 140"),
            (None, ("SEEK",), "C SEEK"),
        ],
    )
    def test_forms(self, pwm, passes, expected):
        assert build_calibrate_command(pwm, passes) == expected

    @pytest.mark.parametrize("passes", ["ALL", "", "SEEK,SWEEP", (), [], 1, ("SEEK", 2), "YAW"])
    def test_bad_passes(self, passes):
        with pytest.raises(ValueError):
            build_calibrate_command(128, passes)

    def test_docstring_examples_use_the_default_speed(self):
        doc = build_calibrate_command.__doc__
        assert "``C 128``" in doc and "``C SEEK 128``" in doc and "``C SWEEP 128``" in doc
        assert " 80``" not in doc
        assert "110-150" in doc and "40-150" not in doc

    def test_bad_pwm(self):
        with pytest.raises(ValueError):
            build_calibrate_command(109)
        with pytest.raises(ValueError):
            build_calibrate_command(80)
        with pytest.raises(ValueError):
            build_calibrate_command(151, "SEEK")


class TestSendCommandCalibrate:
    def test_bare_sends_c(self):
        sdk = _sdk()
        assert sdk.send_command_calibrate() == "C"
        sdk.ser.write.assert_called_once_with(b"C\n")
        sdk.ser.flush.assert_called()

    def test_with_pwm(self):
        sdk = _sdk()
        assert sdk.send_command_calibrate(120) == "C 120"
        sdk.ser.write.assert_called_once_with(b"C 120\n")

    def test_with_pass_and_pwm(self):
        sdk = _sdk()
        sdk.send_command_calibrate(pwm=140, passes="SWEEP")
        sdk.ser.write.assert_called_once_with(b"C SWEEP 140\n")

    @pytest.mark.parametrize("pwm", [0, 80, 109, 151, 120.0, "120", True])
    def test_invalid_pwm_raises_and_writes_nothing(self, pwm):
        sdk = _sdk()
        with pytest.raises(ValueError):
            sdk.send_command_calibrate(pwm)
        sdk.ser.write.assert_not_called()

    def test_invalid_passes_raises_and_writes_nothing(self):
        sdk = _sdk()
        with pytest.raises(ValueError):
            sdk.send_command_calibrate(128, "BOGUS")
        sdk.ser.write.assert_not_called()

    def test_validation_happens_even_with_port_closed(self):
        sdk = _sdk(open_port=False)
        with pytest.raises(ValueError):
            sdk.send_command_calibrate(1000)

    def test_closed_port_sends_nothing(self):
        sdk = _sdk(open_port=False)
        assert sdk.send_command_calibrate(128) is None
        sdk.ser.write.assert_not_called()

    def test_no_port_sends_nothing(self):
        sdk = _sdk()
        sdk.ser = None
        assert sdk.send_command_calibrate() is None

    def test_clears_previous_run_lines(self):
        sdk = _sdk()
        sdk._record_cal_line("CAL ROBOT DONE")
        sdk.send_command_calibrate(128)
        assert sdk.latest_cal_line is None
        assert sdk.recent_cal_lines() == []

    def test_send_does_not_clear_when_nothing_was_sent(self):
        sdk = _sdk(open_port=False)
        sdk._record_cal_line("CAL ROBOT DONE")
        sdk.send_command_calibrate(128)
        assert sdk.latest_cal_line == "CAL ROBOT DONE"


class _ScriptedSerial:
    """readline() returns the scripted lines, then stops the SDK reader loop."""

    def __init__(self, sdk, lines):
        self._sdk = sdk
        self._lines = [line.encode("utf-8") + b"\n" for line in lines]
        self.is_open = True

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._sdk.running = False
        return b"\n"


def _run_reader(sdk, lines):
    sdk.ser = _ScriptedSerial(sdk, lines)
    sdk.running = True
    sdk._reader_loop()


class TestCalLineCapture:
    def test_cal_lines_are_captured_in_order(self):
        sdk = KrabbyMCUSDK(target=Mock())
        lines = [
            "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
            "CAL FRONT PASS SEEK pwm=128",
            "CAL FRONT MOVE FLKL curl",
            "CAL FRONT END FLKL curl pot=1015 travel=0",
        ]
        _run_reader(sdk, lines)
        assert sdk.recent_cal_lines() == lines
        assert sdk.latest_cal_line == lines[-1]
        assert sdk.cal_line_count == 4

    def test_cal_lines_never_reach_telemetry_parser(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        parsed = []
        monkeypatch.setattr(sdk, "_parse_telemetry_line", parsed.append)
        _run_reader(
            sdk,
            [
                "CAL FRONT MOVE FLKL curl",
                # A WARN line contains ';' (like telemetry) and a joint name.
                "CAL RIGHT WARN MRKL pot reads lower when extended; position control "
                "will drive it the wrong way (swap the pot's end wires)",
                "CAL FRONT; FLHY 0.500 500 12 0 0 0 0 0 0",
                "CAL ROBOT DONE",
            ],
        )
        assert parsed == []
        assert sdk.joints == {}
        assert sdk.last_feedback_ts is None
        assert sdk.role_last_seen == {}
        assert sdk.cal_line_count == 4

    def test_telemetry_still_parses_between_cal_lines(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(
            sdk,
            ["CAL FRONT MOVE FLKL curl", TELEMETRY_LINE, "CAL FRONT PASS_DONE SEEK"],
        )
        assert "FLHY" in sdk.joints
        assert "FRONT" in sdk.role_last_seen
        assert sdk.recent_cal_lines() == [
            "CAL FRONT MOVE FLKL curl",
            "CAL FRONT PASS_DONE SEEK",
        ]

    def test_non_cal_lines_are_not_captured(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(
            sdk,
            [
                TELEMETRY_LINE,
                "VER 1.0 main abc",
                "ERR FWD LEFT: line overflow",
                "Saved calibration",
                "POWER CAL ok",
                "CALIBRATE",
            ],
        )
        assert sdk.recent_cal_lines() == []
        assert sdk.latest_cal_line is None
        assert sdk._last_ver_line == "VER 1.0 main abc"

    def test_buffer_is_bounded_but_count_keeps_rising(self):
        sdk = KrabbyMCUSDK(target=Mock())
        total = krabby_mcu.CAL_RECENT_LINES + 10
        for i in range(total):
            sdk._record_cal_line(f"CAL FRONT MOVE J{i} curl")
        kept = sdk.recent_cal_lines()
        assert len(kept) == krabby_mcu.CAL_RECENT_LINES
        assert kept[-1] == f"CAL FRONT MOVE J{total - 1} curl"
        assert sdk.cal_line_count == total

    def test_recent_lines_limit_and_copy(self):
        sdk = KrabbyMCUSDK(target=Mock())
        for i in range(5):
            sdk._record_cal_line(f"CAL FRONT MOVE J{i} curl")
        assert sdk.recent_cal_lines(2) == [
            "CAL FRONT MOVE J3 curl",
            "CAL FRONT MOVE J4 curl",
        ]
        assert sdk.recent_cal_lines(0) == []
        copy = sdk.recent_cal_lines()
        copy.append("junk")
        assert "junk" not in sdk.recent_cal_lines()

    def test_clear(self):
        sdk = KrabbyMCUSDK(target=Mock())
        sdk._record_cal_line("CAL ROBOT DONE")
        sdk.clear_cal_lines()
        assert sdk.latest_cal_line is None
        assert sdk.cal_line_count == 1  # the lifetime counter is not reset

    def test_thread_safe_under_concurrent_writer_and_readers(self):
        sdk = KrabbyMCUSDK(target=Mock())
        writes = 5000
        errors = []
        stop = threading.Event()

        def writer():
            for i in range(writes):
                sdk._record_cal_line(f"CAL FRONT MOVE J{i} curl")
            stop.set()

        def reader():
            try:
                while not stop.is_set():
                    lines = sdk.recent_cal_lines()
                    latest = sdk.latest_cal_line
                    assert len(lines) <= krabby_mcu.CAL_RECENT_LINES
                    assert latest is None or latest.startswith("CAL FRONT MOVE J")
                    sdk.cal_line_count
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                stop.set()

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        assert sdk.cal_line_count == writes


class TestCalLineClassification:
    @pytest.mark.parametrize(
        "line,state",
        [
            (None, ""),
            ("", ""),
            ("CAL ROBOT START pwm=128 passes=SEEK,SWEEP", "running"),
            ("CAL FRONT MOVE FLKL curl", "running"),
            ("CAL FRONT PASS_DONE SEEK", "running"),
            ("CAL ROBOT DONE", "done"),
            ("CAL ROBOT FAIL LEFT: no CAL report for 135 s", "fail"),
            ("CAL FRONT FAIL FLKL pot not connected", "fail"),
            ("CAL LEFT ABORTED H", "fail"),
            ("CAL ROBOT ABORTED H", "fail"),
            ("CAL ROBOT BUSY", "fail"),
            ("CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150", "fail"),
            ("CAL", "running"),
            # New in the hall-encoder protocol (design 8.1): all read as "running".
            ("CAL FRONT MOVE FLKL nudge", "running"),
            ("CAL FRONT SENSOR FLHL hall", "running"),
            ("CAL FRONT SENSOR FLHL pot (was hall)", "running"),
            ("CAL FRONT HOME FLKL count=-3", "running"),
            ("CAL FRONT WARN FLHL home moved 12 edges", "running"),
            ("CAL FRONT WARN FLHL stroke 800 -> 760 edges", "running"),
            ("CAL FRONT END FLKL curl hall=0 travel=312", "running"),
            ("CAL FRONT END FLKL curl hall=-3 travel=0 pot=512", "running"),
            ("CAL FRONT JOINT FLKL min=-776 max=0 edges", "running"),
            ("CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)", "running"),
            ("CAL FRONT JOINT FLHL min=0 max=1069 edges (extend switch-verified)", "running"),
            ("CAL FRONT PARK FLKL hall=-262 target=-264 (FL values)", "running"),
            ("CAL FRONT MOVE FLKL curl park", "running"),
            ("CAL FRONT MOVE FLHL uncurl sweep", "running"),
            ("CAL FRONT MOVE FLHL curl retract", "running"),
            ("CAL FRONT WARN FLHL stopped at 512 edges: retracting", "running"),
            ("CAL FRONT WARN FLKL left out at -264 edges: jog it to its curl end", "running"),
            ("CAL FRONT END FLHL curl hall=0 travel=-840 by=switch cur=75/110 floor=75", "running"),
            ("CAL FRONT END FLHL curl hall=0 travel=-840 open=75/75", "running"),
        ],
    )
    def test_state(self, line, state):
        assert cal_line_state(line) == state

    @pytest.mark.parametrize(
        "line,finished",
        [
            ("CAL ROBOT DONE", True),
            ("CAL ROBOT FAIL FRONT: x", True),
            ("CAL ROBOT ABORTED J", True),
            ("CAL ROBOT ERROR invalid request", True),
            ("CAL ROBOT BUSY", False),
            ("CAL ROBOT START pwm=128 passes=SEEK", False),
            ("CAL ROBOT SKIP LEFT no link", False),
            ("CAL FRONT FAIL FLKL pot not connected", False),
            ("CAL FRONT PASS_DONE SWEEP saved", False),
            ("ROBOT DONE", False),
            ("CAL ROBOT", False),
            ("", False),
            (None, False),
        ],
    )
    def test_robot_finished(self, line, finished):
        assert cal_robot_finished(line) is finished

    @pytest.mark.parametrize(
        "lines,shown",
        [
            ([], None),
            (["CAL ROBOT START pwm=128 passes=SEEK", "CAL FRONT MOVE FLKL curl"], "CAL FRONT MOVE FLKL curl"),
            # A follower's ABORTED H after the robot end line: the run's result stays.
            (
                ["CAL LEFT MOVE RLHL curl", "CAL ROBOT FAIL LEFT: no CAL report for 135 s", "CAL LEFT ABORTED H"],
                "CAL ROBOT FAIL LEFT: no CAL report for 135 s",
            ),
            (["CAL ROBOT ABORTED H", "CAL RIGHT ABORTED H"], "CAL ROBOT ABORTED H"),
            (["CAL ROBOT BUSY"], "CAL ROBOT BUSY"),
        ],
    )
    def test_status_line(self, lines, shown):
        assert cal_status_line(lines) == shown


class TestCalLinesAreNotTelemetry:
    """Every documented CAL line shape is rejected by the telemetry classifier."""

    @pytest.mark.parametrize(
        "line",
        [
            "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
            "CAL ROBOT SKIP LEFT no link",
            "CAL FRONT PASS SEEK pwm=128",
            "CAL FRONT MOVE FLKL curl",
            "CAL FRONT END FLKL uncurl pot=210 travel=-805",
            "CAL FRONT JOINT FLKL min=210 max=1015",
            "CAL RIGHT WARN MRKL pot reads lower when extended; position control will "
            "drive it the wrong way (swap the pot's end wires)",
            "CAL LEFT PASS_DONE SWEEP saved",
            "CAL LEFT FAIL RLHL barely moved (12 counts) at PWM 128: check its pot wiring",
            "CAL UNKWN ABORTED H",
            "CAL ROBOT ERROR usage: C [SEEK|SWEEP] [pwm], pwm 110-150",
        ],
    )
    def test_not_telemetry(self, line):
        assert TelemetryFrame.role_from_line(line) is None
        assert not TelemetryFrame.is_telemetry_line(line)


# One line per failure reason of design 8.1, as a board prints it: CAL <ROLE> FAIL <reason>.
HALL_FAIL_LINES = [
    "CAL FRONT FAIL FLKL pot not connected",
    "CAL FRONT FAIL FLKL curl: no end of travel in 120 s",
    "CAL FRONT FAIL FLHL barely moved (7 edges) at PWM 150: check hall wiring or raise speed",
    "CAL FRONT FAIL FLKL did not reach its curl end (hall=-212)",
    "CAL FRONT FAIL FLKL did not reach its curl end (pot=512)",
    "CAL FRONT FAIL FLKL hall counts the wrong way: check motor or hall A/B wiring",
    "CAL FRONT FAIL FLKL pot moves the wrong way: check motor or pot end wires",
    "CAL FRONT FAIL FLKL no feedback: no hall edges, pot not moving (check sensor cable)",
    "CAL FRONT FAIL FLKL no motor current at PWM 150: check motor wiring",
    "CAL FRONT FAIL FLKL no motion seen: jog it off its curl end, then retry",
    "CAL FRONT FAIL FLHL stopped 41 edges short of its curl end",
    "CAL FRONT FAIL FLHL stopped 41 counts short of its curl end",
    "CAL FRONT FAIL FLKL stroke 1600 edges, outside 450-1500",
    "CAL FRONT FAIL FLKL stroke 600 edges, saved 776: obstacle?",
    "CAL FRONT FAIL FLHL stopped 259 edges short of its 840-edge sweep: obstacle? check its home",
    "CAL FRONT FAIL FLHL lost counts: switch at 1010 edges, expected 1069",
    "CAL FRONT FAIL FLHL lost counts: switch at 14 edges, expected 0",
    "CAL FRONT FAIL FLKL at -300 edges, outside its park band -280..-244",
    "CAL FRONT FAIL FLKL stopped 40 edges short of its park at -264: obstacle?",
    "CAL FRONT FAIL FLKL stroke 685 edges < 761 (FL): obstacle or bind at its curl end?",
    "CAL FRONT FAIL FLHL not back at its curl switch (hall=212): jog it home",
    "CAL FRONT FAIL FLHL and its knee must both be hall encoders",
    "CAL FRONT FAIL FLHL stuck mid-stroke after 612 edges at PWM 150: raise PWM or check load",
    "CAL FRONT FAIL FLHL no edges and no switch signature at PWM 128: check hall cable or load",
    "CAL FRONT FAIL FLHL out at 700 edges: jog it to its curl end first",
    "CAL FRONT FAIL FLHL partner not proven curled",
]


class TestHallCalLines:
    """Hall-encoder wording (design 8.1) next to the pot fixtures above."""

    @pytest.mark.parametrize("line", HALL_FAIL_LINES)
    def test_every_failure_reason_is_a_fail(self, line):
        assert cal_line_state(line) == "fail"
        assert not cal_robot_finished(line)  # a board line, not the run's end

    @pytest.mark.parametrize("line", HALL_FAIL_LINES)
    def test_robot_end_line_echoing_the_reason_is_the_status(self, line):
        reason = line.split(" FAIL ", 1)[1]
        robot = f"CAL ROBOT FAIL FRONT: {reason}"
        assert cal_robot_finished(robot)
        assert cal_line_state(robot) == "fail"
        assert cal_status_line([line, robot, "CAL LEFT ABORTED H"]) == robot

    @pytest.mark.parametrize(
        "line",
        [
            "CAL FRONT MOVE FLKL nudge",
            "CAL FRONT SENSOR FLHL hall",
            "CAL FRONT SENSOR FLHL pot (was hall)",
            "CAL FRONT HOME FLHL count=-3",
            "CAL FRONT WARN FLHL home moved 12 edges",
            "CAL FRONT WARN FLHL stroke 800 -> 760 edges",
            "CAL FRONT END FLKL curl hall=0 travel=312",
            "CAL FRONT END FLHL uncurl hall=-812 travel=-812",
            "CAL FRONT END FLKL curl hall=-3 travel=0 pot=512",
            "CAL FRONT JOINT FLKL min=-776 max=0 edges",
            "CAL FRONT JOINT FLHL min=0 max=1069 edges (extend derived from FL, not switch-verified)",
        ],
    )
    def test_progress_words_are_running_and_not_an_end(self, line):
        assert cal_line_state(line) == "running"
        assert not cal_robot_finished(line)
        assert TelemetryFrame.role_from_line(line) is None

    @pytest.mark.parametrize("line", HALL_FAIL_LINES)
    def test_failure_lines_are_not_telemetry(self, line):
        assert TelemetryFrame.role_from_line(line) is None
        assert not TelemetryFrame.is_telemetry_line(line)

    def test_hall_lines_are_captured_as_cal_lines(self):
        sdk = KrabbyMCUSDK(target=Mock())
        lines = [
            "CAL FRONT SENSOR FLHL hall",
            "CAL FRONT MOVE FLKL nudge",
            "CAL FRONT HOME FLKL count=-3",
            "CAL FRONT WARN FLHL home moved 12 edges",
        ]
        _run_reader(sdk, lines)
        assert sdk.recent_cal_lines() == lines
        assert sdk.joint_sensor_state == {}  # a CAL SENSOR line is not a SENS line


# Mixed-firmware gate (review finding R1-8): a quadrature leader refuses to start
# while a linked follower has printed no SENS line since the leader booted.
NO_SENS_LINES = [
    "CAL ROBOT FAIL LEFT: no SENS: flash all three boards with the same firmware, "
    "or wait a few seconds after connecting",
    "CAL ROBOT FAIL RIGHT: no SENS: flash all three boards with the same firmware, "
    "or wait a few seconds after connecting",
]


class TestNoSensRefusal:
    @pytest.mark.parametrize("line", NO_SENS_LINES)
    def test_is_the_runs_failed_end(self, line):
        assert len(line) > 100  # longer than the usual robot FAIL line
        assert cal_robot_finished(line)
        assert cal_line_state(line) == "fail"
        assert cal_status_line(["CAL ROBOT BUSY", line]) == line

    @pytest.mark.parametrize("line", NO_SENS_LINES)
    def test_is_kept_whole(self, line):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, [line])
        assert sdk.recent_cal_lines() == [line]
        assert sdk.latest_cal_line == line
        assert not TelemetryFrame.is_telemetry_line(line)


def _telemetry(name, pos):
    return JointTelemetry(
        name=name, pos=pos, pot=500, current=0, en=(0, 0), pwm=(0, 0), saf=0
    )


class TestSensorLine:
    def test_parses_each_token(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHY:? FLHL:e FLKL:x FRHY:P FRHL:s FRKL:E"])
        assert sdk.sensor_states() == {
            "FLHY": "?",
            "FLHL": "e",
            "FLKL": "x",
            "FRHY": "P",
            "FRHL": "s",
            "FRKL": "E",
        }
        assert sdk.sensor_state("FLHL") == "e"
        assert sdk.sensor_state("flkl") == "x"  # names are matched case-insensitively

    def test_unseen_joint_is_none(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHL:E"])
        assert sdk.sensor_state("FLKL") is None
        assert sdk.sensor_state("RLHL") is None

    def test_later_line_replaces_the_state(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHL:e", "SENS FRONT FLHL:E FLKL:s"])
        assert sdk.sensor_states() == {"FLHL": "E", "FLKL": "s"}

    def test_boards_do_not_overwrite_each_others_joints(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHL:E", "SENS LEFT RLHL:e"])
        assert sdk.sensor_states() == {"FLHL": "E", "RLHL": "e"}

    @pytest.mark.parametrize(
        "line,expected",
        [
            # A bad token is skipped; the good ones on the line are kept.
            ("SENS FRONT FLHL:e BAD FLKL:Z FRHL:E", {"FLHL": "e", "FRHL": "E"}),
            ("SENS FRONT FLHL:ee FLKL:E", {"FLKL": "E"}),  # code is one character
            ("SENS FRONT FLHL: FLKL:E", {"FLKL": "E"}),  # empty code
            ("SENS FRONT :E FLKL:s", {"FLKL": "s"}),  # empty name
            ("SENS FRONT FL-HL:E FLKL:s", {"FLKL": "s"}),  # not a joint name
            ("SENS FRONT FLHL:e:E FLKL:s", {"FLKL": "s"}),  # two colons
            ("SENS FRONT FLHL:p FLKL:s", {"FLKL": "s"}),  # codes are case-sensitive
            ("SENS FRONT FLHL=e FLKL:s", {"FLKL": "s"}),
            # No usable token at all.
            ("SENS", {}),
            ("SENS FRONT", {}),
            ("SENS FRONT garbage", {}),
            ("SENS FRONT FLHL:Z", {}),
            ("SENSOR FRONT FLHL:E", {}),  # not a SENS line
        ],
    )
    def test_malformed_input(self, line, expected):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, [line])
        assert sdk.sensor_states() == expected

    def test_bad_line_keeps_the_previous_states(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHL:E", "SENS FRONT garbage", "SENS"])
        assert sdk.sensor_states() == {"FLHL": "E"}

    def test_sens_lines_stay_out_of_the_cal_deque_and_telemetry_parser(self, monkeypatch):
        sdk = KrabbyMCUSDK(target=Mock())
        parsed = []
        monkeypatch.setattr(sdk, "_parse_telemetry_line", parsed.append)
        _run_reader(sdk, ["SENS FRONT FLHL:E", "CAL FRONT MOVE FLKL curl"])
        assert parsed == []
        assert sdk.recent_cal_lines() == ["CAL FRONT MOVE FLKL curl"]
        assert sdk.cal_line_count == 1
        assert sdk.joints == {}
        assert sdk.last_feedback_ts is None
        assert sdk.role_last_seen == {}

    def test_telemetry_still_parses_between_sens_lines(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHY:?", TELEMETRY_LINE, "SENS FRONT FLHY:?"])
        assert "FLHY" in sdk.joints
        assert "FRONT" in sdk.role_last_seen
        assert sdk.sensor_state("FLHY") == "?"

    def test_states_are_a_copy(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, ["SENS FRONT FLHL:E"])
        copy = sdk.sensor_states()
        copy["FLHL"] = "x"
        assert sdk.sensor_state("FLHL") == "E"

    def test_no_sens_line_means_legacy_empty_state(self):
        sdk = KrabbyMCUSDK(target=Mock())
        _run_reader(sdk, [TELEMETRY_LINE, "VER 1.0 main abc"])
        assert sdk.sensor_states() == {}
        assert sdk.sensor_state("FLHY") is None

    def test_state_change_is_logged_once(self, caplog):
        sdk = KrabbyMCUSDK(target=Mock())
        with caplog.at_level(logging.INFO, logger="KrabbySDK"):
            _run_reader(
                sdk,
                ["SENS FRONT FLHL:e", "SENS FRONT FLHL:e", "SENS FRONT FLHL:E"],
            )
        changes = [r.getMessage() for r in caplog.records if "sensor FLHL" in r.getMessage()]
        assert changes == ["[MCU] sensor FLHL: - -> e", "[MCU] sensor FLHL: e -> E"]

    def test_concurrent_writer_and_readers(self):
        sdk = KrabbyMCUSDK(target=Mock())
        stop = threading.Event()
        errors = []

        def writer():
            for i in range(3000):
                sdk._record_sensor_line(f"SENS FRONT FLHL:{'eE'[i % 2]} FLKL:s")
            stop.set()

        def reader():
            try:
                while not stop.is_set():
                    states = sdk.sensor_states()
                    assert states == {} or states.get("FLHL") in ("e", "E")
                    sdk.sensor_state("FLHL")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)
                stop.set()

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors


class TestErrLines:
    def test_err_lines_are_logged_as_errors_and_not_captured(self, caplog):
        sdk = KrabbyMCUSDK(target=Mock())
        lines = [
            "ERR T refused FLHL FLKL (no position)",
            "ERR STALL FLHL",
            "ERR HOME FLKL lost (x=-412)",
        ]
        with caplog.at_level(logging.INFO, logger="KrabbySDK"):
            _run_reader(sdk, lines)
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert errors == [f"[MCU] {line}" for line in lines]
        assert sdk.recent_cal_lines() == []
        assert sdk.sensor_states() == {}


class TestEepromLines:
    def test_eeprom_load_and_save_lines_are_logged(self, caplog):
        sdk = KrabbyMCUSDK(target=Mock())
        lines = [
            "Calibration v2 loaded",
            "Calibration v1 ignored: run C",
            "Calibration from a build without halls ignored: run C",
            "Calibration record corrupt: defaults",
            "Calibration joint FLHL invalid: defaults",
            "No calibration record: defaults",
            "Limits saved",
        ]
        with caplog.at_level(logging.INFO, logger="KrabbySDK"):
            _run_reader(sdk, lines + ["some other boot noise"])
        infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        assert infos == [f"[MCU] {line}" for line in lines]
        assert sdk.recent_cal_lines() == []


class TestIsHipOrKnee:
    @pytest.mark.parametrize("name", ["FLHL", "FLKL", "mrkl", " RRHL "])
    def test_hips_and_knees(self, name):
        assert is_hip_or_knee(name)

    @pytest.mark.parametrize("name", ["FLHY", "MRHY", "", "L", "FLHX"])
    def test_everything_else(self, name):
        assert not is_hip_or_knee(name)


def _sdk_with(joints=None, states=None):
    """SDK with telemetry and SENS states set directly (name -> pos, name -> code)."""
    sdk = KrabbyMCUSDK(target=Mock())
    for name, pos in (joints or {}).items():
        sdk.joints[name] = _telemetry(name, pos)
    for name, code in (states or {}).items():
        sdk.joint_sensor_state[name] = code
    return sdk


ALL_NAMES = [n for _, names in krabby_mcu.JOINT_GROUP_NAMES for n in names]
HIPS_AND_KNEES = [n for n in ALL_NAMES if is_hip_or_knee(n)]
YAWS = [n for n in ALL_NAMES if n.endswith("HY")]


class TestPositionReady:
    @pytest.mark.parametrize("state", ["P", "E"])
    def test_ready_states_with_a_finite_position(self, state):
        sdk = _sdk_with({"FLHL": 0.4}, {"FLHL": state})
        assert sdk.position_ready("FLHL")

    @pytest.mark.parametrize("state", ["?", "e", "x", "s"])
    def test_other_states_are_not_ready(self, state):
        sdk = _sdk_with({"FLHL": 0.4}, {"FLHL": state})
        assert not sdk.position_ready("FLHL")

    def test_legacy_no_sens_uses_the_finite_position(self):
        sdk = _sdk_with({"FLHL": 0.4, "FLKL": math.nan})
        assert sdk.sensor_state("FLHL") is None
        assert sdk.position_ready("FLHL")
        assert not sdk.position_ready("FLKL")

    @pytest.mark.parametrize("pos", [math.nan, math.inf, -math.inf])
    def test_non_finite_position_is_never_ready(self, pos):
        for state in (None, "P", "E"):
            states = {} if state is None else {"FLHL": state}
            assert not _sdk_with({"FLHL": pos}, states).position_ready("FLHL")

    def test_no_telemetry_yet_is_not_ready(self):
        assert not _sdk_with(states={"FLHL": "E"}).position_ready("FLHL")
        assert not _sdk_with().position_ready("FLHL")

    @pytest.mark.parametrize("name", YAWS)
    def test_yaws_are_never_ready(self, name):
        for state in (None, "P", "E"):
            states = {} if state is None else {name: state}
            assert not _sdk_with({name: 0.5}, states).position_ready(name)

    def test_name_is_matched_case_insensitively(self):
        sdk = _sdk_with({"FLHL": 0.4}, {"FLHL": "E"})
        assert sdk.position_ready("flhl")

    def test_state_of_another_joint_does_not_count(self):
        sdk = _sdk_with({"FLHL": 0.4}, {"FLKL": "x"})
        assert sdk.position_ready("FLHL")  # no SENS for FLHL: legacy rule


class TestNeutralTargets:
    def test_all_ready_sends_every_hip_and_knee_and_no_yaw(self):
        sdk = _sdk_with(
            {n: 0.3 for n in ALL_NAMES}, {n: "E" for n in HIPS_AND_KNEES}
        )
        targets, skipped = sdk.neutral_targets()
        assert set(targets) == set(HIPS_AND_KNEES)
        assert len(targets) == 12
        assert set(targets.values()) == {0.5}
        assert skipped == []

    def test_yaws_are_excluded_even_when_they_have_a_position(self):
        sdk = _sdk_with({n: 0.3 for n in ALL_NAMES})  # legacy firmware, all finite
        targets, skipped = sdk.neutral_targets()
        assert not set(targets) & set(YAWS)
        assert not set(skipped) & set(YAWS)
        assert set(targets) == set(HIPS_AND_KNEES)

    def test_unready_joints_are_skipped_and_named(self):
        states = {n: "E" for n in HIPS_AND_KNEES}
        states.update({"FLHL": "e", "RLKL": "x", "MRHL": "s", "FRKL": "?"})
        sdk = _sdk_with({n: 0.3 for n in ALL_NAMES}, states)
        targets, skipped = sdk.neutral_targets()
        assert set(skipped) == {"FLHL", "RLKL", "MRHL", "FRKL"}
        assert not set(targets) & set(skipped)
        assert len(targets) + len(skipped) == 12

    def test_skipped_follow_the_joint_table_order(self):
        sdk = _sdk_with({}, {})
        targets, skipped = sdk.neutral_targets()
        assert targets == {}
        assert skipped == HIPS_AND_KNEES

    def test_nan_position_is_skipped(self):
        joints = {n: 0.3 for n in HIPS_AND_KNEES}
        joints["FLKL"] = math.nan
        sdk = _sdk_with(joints)
        targets, skipped = sdk.neutral_targets()
        assert skipped == ["FLKL"]
        assert "FLKL" not in targets

    def test_custom_value(self):
        sdk = _sdk_with({"FLHL": 0.1}, {"FLHL": "P"})
        targets, _ = sdk.neutral_targets(0.25)
        assert targets == {"FLHL": 0.25}

    def test_a_reset_board_stops_receiving_neutral(self):
        sdk = _sdk_with({"FLHL": 0.4}, {"FLHL": "E"})
        assert "FLHL" in sdk.neutral_targets()[0]
        sdk._record_sensor_line("SENS FRONT FLHL:e")  # the board rebooted: unhomed
        targets, skipped = sdk.neutral_targets()
        assert "FLHL" not in targets and "FLHL" in skipped

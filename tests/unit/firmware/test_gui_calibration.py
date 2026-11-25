"""Headless GUI tests: jog direction (B3), curl labels, Calibrate dialog, CAL status.

CI has no $DISPLAY so real Tk cannot start. These tests swap ``ttk``/``tk.StringVar``
in ``firmware.gui.app`` for fakes that record calls and keep the bound callbacks, so the
real ``JointRow`` / ``KrabbyTestGUI`` code runs unmodified.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("serial")

from firmware.gui import app
from firmware.gui.app import (
    CALIBRATE_EXPLANATION,
    EXTEND_DIRECTION,
    JOG_PWM,
    RETRACT_DIRECTION,
    CalibrateDialog,
    CalibrationChoice,
    JointRow,
    KrabbyTestGUI,
    cal_status_text,
    joint_button_labels,
)
from firmware.krabby_mcu import (
    CAL_DEFAULT_PWM,
    CAL_PWM_MAX,
    CAL_PWM_MIN,
    JOINT_GROUP_NAMES,
    build_calibrate_command,
)


class _FakeStringVar:
    def __init__(self, value=""):
        self._value = value

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeWidget:
    def __init__(self, parent=None, **kwargs):
        self.kwargs = kwargs
        self.text = kwargs.get("text")
        self.bindings = {}

    def grid(self, **kwargs):
        return self

    def bind(self, sequence, callback):
        self.bindings[sequence] = callback

    def pack(self, **kwargs):
        return self

    def focus_set(self):
        pass

    def selection_range(self, start, end):
        pass

    def configure(self, **kwargs):
        self.kwargs.update(kwargs)


class _FakeTtk:
    Label = _FakeWidget
    Button = _FakeWidget


@pytest.fixture
def fake_tk(monkeypatch):
    monkeypatch.setattr(app, "ttk", _FakeTtk)
    monkeypatch.setattr(app.tk, "StringVar", _FakeStringVar)
    monkeypatch.setattr(app.tk, "BooleanVar", _FakeStringVar)


def _row(name, fake_tk_fixture=None):
    jogs = []
    row = JointRow(object(), name, 1, lambda joint, pwm: jogs.append((joint, pwm)))
    return row, jogs


def _press(widget):
    widget.bindings["<ButtonPress-1>"](None)


def _release(widget):
    widget.bindings["<ButtonRelease-1>"](None)


class TestJogDirection:
    """B3: the firmware's +PWM extends the rod, so Retract must send -PWM."""

    def test_constants(self):
        assert RETRACT_DIRECTION == -1
        assert EXTEND_DIRECTION == +1

    @pytest.mark.parametrize(
        "name", [n for _, names in JOINT_GROUP_NAMES for n in names]
    )
    def test_retract_sends_negative_pwm(self, fake_tk, name):
        row, jogs = _row(name)
        _press(row.btn_retract)
        assert jogs == [(name, -JOG_PWM)]

    @pytest.mark.parametrize(
        "name", [n for _, names in JOINT_GROUP_NAMES for n in names]
    )
    def test_extend_sends_positive_pwm(self, fake_tk, name):
        row, jogs = _row(name)
        _press(row.btn_extend)
        assert jogs == [(name, +JOG_PWM)]

    def test_release_stops_either_button(self, fake_tk):
        row, jogs = _row("FLHL")
        _press(row.btn_retract)
        _release(row.btn_retract)
        _press(row.btn_extend)
        _release(row.btn_extend)
        assert jogs == [
            ("FLHL", -JOG_PWM),
            ("FLHL", 0),
            ("FLHL", +JOG_PWM),
            ("FLHL", 0),
        ]

    def test_button_labels_match_actions(self, fake_tk):
        row, _ = _row("FLHL")
        assert "Retract" in row.btn_retract.text
        assert "Extend" in row.btn_extend.text


class TestCurlLabels:
    def test_hip_curl_tag_on_retract_only(self):
        retract, extend = joint_button_labels("FLHL")
        assert retract.endswith("Retract (curl)")
        assert "(curl)" not in extend
        assert "Extend" in extend

    def test_knee_curl_tag_on_extend_only(self):
        retract, extend = joint_button_labels("FLKL")
        assert "(curl)" not in retract
        assert "Retract" in retract
        assert extend.startswith("Extend (curl)")

    def test_yaw_has_no_curl_tag(self):
        retract, extend = joint_button_labels("FLHY")
        assert "curl" not in retract
        assert "curl" not in extend

    def test_every_leg_follows_the_same_rule(self):
        for _, names in JOINT_GROUP_NAMES:
            for name in names:
                retract, extend = joint_button_labels(name)
                if name.endswith("HL"):
                    assert "(curl)" in retract and "(curl)" not in extend, name
                elif name.endswith("KL"):
                    assert "(curl)" in extend and "(curl)" not in retract, name
                else:
                    assert "curl" not in retract + extend, name

    def test_row_buttons_carry_the_labels(self, fake_tk):
        hip, _ = _row("RLHL")
        knee, _ = _row("RLKL")
        assert "(curl)" in hip.btn_retract.text and "(curl)" not in hip.btn_extend.text
        assert "(curl)" in knee.btn_extend.text and "(curl)" not in knee.btn_retract.text


class TestCalibrateExplanation:
    def test_describes_the_new_sequence(self):
        text = CALIBRATE_EXPLANATION.lower()
        assert "one joint at a time" in text
        assert "curl" in text
        assert "yaw" in text and "never" in text
        assert "hold all" in text
        assert "hip curls on retract" in text
        assert "knee on extend" in text

    def test_no_stale_wording(self):
        assert "ALL limbs" not in CALIBRATE_EXPLANATION

    def test_mentions_the_curl_only_option(self):
        text = CALIBRATE_EXPLANATION.lower()
        assert "curl only" in text
        assert "saved" in text


class TestCalibrateExplanationEncoders:
    """Dialog text for the shipped default: hip and knee nudge (design 8.3, 13.1)."""

    def test_sensor_kind_is_detected_automatically(self):
        text = CALIBRATE_EXPLANATION.lower()
        assert "pot or encoder" in text
        assert "detected automatically" in text

    def test_encoder_joints_need_curl_only_after_every_power_up(self):
        text = CALIBRATE_EXPLANATION
        assert "homed with Curl only after every power-up before Neutral works" in text
        assert "Curl only" in text and "saved" in text  # Curl only still saves nothing

    def test_no_instruction_to_jog_before_curl_only_or_the_nudge(self):
        # The knee nudge is the shipped default: nothing is jogged before Curl only.
        text = " ".join(CALIBRATE_EXPLANATION.lower().split())
        assert "off its curl end" not in text
        assert "half a second" not in text
        assert "on its retract button" not in text

    def test_before_you_start_jog_hips_inside_560_and_knees_to_curl(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "jog every hip inside 0-560 counts and every knee to its curl end" in text
        assert "HOME line" in text and "limit switch" in text

    def test_parked_hips_and_knees_are_nudged_and_checked(self):
        text = CALIBRATE_EXPLANATION
        assert "A hip or knee resting fully curled is nudged out a little" in text
        assert "straight back so its sensor can be checked" in text

    def test_states_the_nudge_bounds_in_hall_counts(self):
        text = CALIBRATE_EXPLANATION
        assert "at most 24 hall counts for a hip, 12 for a knee" in text
        # The counts per millimetre is not measured yet: no millimetres.
        assert "mm" not in text
        assert "millimet" not in text.lower()

    def test_seek_order_is_knee_hip_knee_again(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "one leg at a time: knee, then hip, then the knee again" in text
        assert "The second knee curl is the proven one" in text

    def test_sweep_order_is_table_safe(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert (
            "knee curl, hip curl (home 0), knee curl again, knee uncurl to its switch, knee park "
            f"{app.CAL_KNEE_PARK_EDGES} counts out of its curl stop, hip sweep to "
            f"{app.CAL_HIP_SWEEP_EDGES} hall counts with the knee held there, hip back to "
            "its curl switch" in text
        )
        assert (app.CAL_KNEE_PARK_EDGES, app.CAL_HIP_SWEEP_EDGES) == (264, 840)
        assert "within 10 counts of 0, else counts were lost" in text
        assert text.endswith("nothing for the board in progress.")
        assert "knee curl. The hip moves past 560 counts only with its knee parked" in text
        assert "no automatic retry" in text
        assert (
            "a knee never moves while its hip is known to be out past 560 counts" in text
        )
        assert "jog the hip to its curl end first" in text

    def test_neutral_note_keeps_hips_at_or_below_560_on_a_stand(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "Neutral sends every hall hip to about 534 counts" in text
        assert "every knee to mid-stroke" in text
        assert "keep hip targets at or below 560 counts" in text
        assert "stand envelope" in text

    def test_old_sweep_claims_are_gone(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "70%" not in text and "700" not in text
        assert "Each hip goes out" not in text

    def test_hip_extend_limit_is_derived_not_switch_verified(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "never sought on a solid stand" in text
        assert f"derived ({app.CAL_HIP_STROKE_EDGES} hall counts" in text
        assert app.CAL_HIP_STROKE_EDGES == 1069
        assert "not switch-verified" in text
        assert "Every leg uses the front-left leg's values" in text

    def test_knee_spring_after_its_nudge_is_expected(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "A curled knee springs a little further after its nudge stops" in text
        assert "ends about 20 counts out" in text

    def test_flash_all_three_boards_and_wait_after_connecting(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "Flash all three boards together with the same firmware" in text
        assert "wait a few seconds after connecting" in text

    def test_advises_the_default_speed_and_a_curl_only_retry(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert f"Use the default speed (PWM {app.CAL_DEFAULT_PWM})" in text
        assert app.CAL_DEFAULT_PWM == 128
        assert f"the range is {app.CAL_PWM_MIN}-{app.CAL_PWM_MAX}" in text
        assert f"{app.CAL_PWM_MIN}-{app.CAL_DEFAULT_PWM - 1} is not validated on the robot" in text
        assert "PWM 80 stalled mid-stroke with no current signature" in text
        assert f"nothing below {app.CAL_PWM_MIN} is accepted" in text
        assert 'If a knee stops with "curled back only", run Curl only again.' in text
        # The old low-speed advice is gone.
        assert "PWM 40-50" not in text
        assert "its ends are found by stall alone" not in text

    def test_end_detection_claims_are_corrected(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "(every hip curl, every knee uncurl, the hip's extended end)" in text
        assert "ends only when the motor current is flat at its switch-open level" in text
        assert "any stall without that signature stops the run as stuck mid-stroke" in text
        assert "A knee curl ends at its compliant leg stop" in text
        assert "an obstacle on a knee curl looks the same" in text
        assert "A hip sweep treats any stall as an obstacle" in text
        assert "never taken as an end" not in text

    def test_recovery_after_a_stuck_hip(self):
        text = " ".join(CALIBRATE_EXPLANATION.split())
        assert "never rerun at 128: jog the hip to its curl end, or rerun at 150" in text
        assert "more than 40 counts short of it fails instead of homing there" in text

    def test_run_stops_and_names_a_joint_that_shows_no_movement(self):
        text = CALIBRATE_EXPLANATION
        assert "A joint that still shows no movement stops the run" in text
        assert "the run names it" in text
        assert "cannot be verified" not in text

    def test_stand_and_hold_all(self):
        text = CALIBRATE_EXPLANATION
        assert "robot on its stand" in text
        assert "Hold All stops everything" in text


class _DialogHarness(CalibrateDialog):
    """CalibrateDialog without Tk: only the button logic is exercised."""

    def __init__(  # noqa: super().__init__ would need a display
        self, speed_text, curl_only=False
    ):
        self._speed_var = _FakeStringVar(speed_text)
        self._curl_only_var = _FakeStringVar(curl_only)
        self._error_var = _FakeStringVar("")
        self.result = None
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class TestCalibrateDialogLogic:
    def test_default_speed_constant(self):
        assert (CAL_DEFAULT_PWM, CAL_PWM_MIN, CAL_PWM_MAX) == (128, 110, 150)

    @pytest.mark.parametrize("text,expected", [("110", 110), (" 128", 128), ("150", 150)])
    def test_valid_speed_closes_with_result(self, text, expected):
        dialog = _DialogHarness(text)
        dialog._on_start()
        assert dialog.result == CalibrationChoice(expected, None)
        assert dialog.destroyed
        assert dialog._error_var.get() == ""

    @pytest.mark.parametrize("text", ["", "abc", "109", "80", "40", "151", "-80", "120.5", "0"])
    def test_invalid_speed_stays_open_with_error(self, text):
        dialog = _DialogHarness(text)
        dialog._on_start()
        assert dialog.result is None
        assert not dialog.destroyed
        assert str(CAL_PWM_MIN) in dialog._error_var.get()
        assert str(CAL_PWM_MAX) in dialog._error_var.get()

    def test_cancel(self):
        dialog = _DialogHarness("128")
        dialog.result = CalibrationChoice(99, None)
        dialog._on_cancel()
        assert dialog.result is None
        assert dialog.destroyed

    def test_curl_only_ticked_yields_seek(self):
        dialog = _DialogHarness("130", curl_only=True)
        dialog._on_start()
        assert dialog.result == CalibrationChoice(130, "SEEK")
        assert dialog.destroyed

    def test_curl_only_unticked_yields_the_full_run(self):
        dialog = _DialogHarness("130", curl_only=False)
        dialog._on_start()
        assert dialog.result == CalibrationChoice(130, None)

    def test_cancel_with_curl_only_ticked_returns_nothing(self):
        dialog = _DialogHarness("130", curl_only=True)
        dialog._on_cancel()
        assert dialog.result is None
        assert dialog.destroyed

    def test_invalid_speed_with_curl_only_ticked_stays_open(self):
        dialog = _DialogHarness("80", curl_only=True)
        dialog._on_start()
        assert dialog.result is None
        assert not dialog.destroyed


class _RecordingTtk:
    """ttk stand-in that keeps every widget so the built dialog can be inspected."""

    def __init__(self):
        self.widgets = []

    def _make(self, kind):
        def build(parent=None, **kwargs):
            widget = _FakeWidget(parent, **kwargs)
            widget.kind = kind
            self.widgets.append(widget)
            return widget

        return build

    def __getattr__(self, name):
        if name in ("Frame", "Label", "Entry", "Button", "Checkbutton"):
            return self._make(name)
        raise AttributeError(name)


@pytest.fixture
def build_dialog(monkeypatch, fake_tk):
    """Construct the real CalibrateDialog without a display; returns a builder."""
    ttk = _RecordingTtk()
    monkeypatch.setattr(app, "ttk", ttk)
    monkeypatch.setattr(app.tk.Toplevel, "__init__", lambda self, parent=None: None)
    for name in ("title", "resizable", "transient", "protocol", "grab_set", "bind"):
        monkeypatch.setattr(CalibrateDialog, name, lambda self, *a, **k: None)
    monkeypatch.setattr(CalibrateDialog, "destroy", lambda self: None)

    def build():
        return CalibrateDialog(object()), ttk

    return build


class TestCalibrateDialogCurlOnlyBox:
    def test_checkbox_exists_unticked_and_bound_to_the_choice(self, build_dialog):
        dialog, ttk = build_dialog()
        boxes = [w for w in ttk.widgets if w.kind == "Checkbutton"]
        assert len(boxes) == 1
        assert "curl only" in boxes[0].kwargs["text"].lower()
        assert boxes[0].kwargs["variable"] is dialog._curl_only_var
        assert dialog._curl_only_var.get() is False

    def test_ticked_start_yields_seek(self, build_dialog):
        dialog, _ = build_dialog()
        dialog._curl_only_var.set(True)
        dialog._on_start()
        assert dialog.result == CalibrationChoice(CAL_DEFAULT_PWM, "SEEK")

    def test_unticked_start_yields_the_full_run(self, build_dialog):
        dialog, _ = build_dialog()
        dialog._on_start()
        assert dialog.result == CalibrationChoice(CAL_DEFAULT_PWM, None)

    def test_cancel_with_box_ticked_returns_nothing(self, build_dialog):
        dialog, _ = build_dialog()
        dialog._curl_only_var.set(True)
        dialog._on_cancel()
        assert dialog.result is None

    def test_reopened_dialog_starts_unticked(self, build_dialog):
        first, _ = build_dialog()
        first._curl_only_var.set(True)
        first._on_start()
        second, _ = build_dialog()
        assert second._curl_only_var.get() is False
        second._on_start()
        assert second.result == CalibrationChoice(CAL_DEFAULT_PWM, None)


def _gui(stm=False, connected=True):
    gui = KrabbyTestGUI.__new__(KrabbyTestGUI)
    gui._stm = stm
    gui._connected = connected
    gui._mcu = Mock()
    return gui


class TestCalibrateButton:
    def test_sends_chosen_speed(self, monkeypatch):
        gui = _gui()
        monkeypatch.setattr(
            app, "ask_calibration_settings", lambda parent: CalibrationChoice(130, None)
        )
        gui._calibrate()
        gui._mcu.send_command_calibrate.assert_called_once_with(130, None)
        assert build_calibrate_command(130, None) == "C 130"

    def test_curl_only_sends_seek(self, monkeypatch):
        gui = _gui()
        monkeypatch.setattr(
            app, "ask_calibration_settings", lambda parent: CalibrationChoice(130, "SEEK")
        )
        gui._calibrate()
        gui._mcu.send_command_calibrate.assert_called_once_with(130, "SEEK")
        assert build_calibrate_command(130, "SEEK") == "C SEEK 130"

    def test_default_speed_is_offered_as_128(self, monkeypatch):
        gui = _gui()
        seen = []

        def ask(parent):
            seen.append(parent)
            return CalibrationChoice(CAL_DEFAULT_PWM, None)

        monkeypatch.setattr(app, "ask_calibration_settings", ask)
        gui._calibrate()
        assert seen == [gui]
        gui._mcu.send_command_calibrate.assert_called_once_with(128, None)

    def test_cancel_sends_nothing(self, monkeypatch):
        gui = _gui()
        monkeypatch.setattr(app, "ask_calibration_settings", lambda parent: None)
        gui._calibrate()
        gui._mcu.send_command_calibrate.assert_not_called()

    def test_not_connected_does_not_even_ask(self, monkeypatch):
        gui = _gui(connected=False)
        ask = Mock(return_value=CalibrationChoice(128, None))
        monkeypatch.setattr(app, "ask_calibration_settings", ask)
        gui._calibrate()
        ask.assert_not_called()
        gui._mcu.send_command_calibrate.assert_not_called()

    def test_rejected_speed_is_reported_not_raised(self, monkeypatch):
        gui = _gui()
        gui._mcu.send_command_calibrate.side_effect = ValueError("bad speed")
        monkeypatch.setattr(
            app, "ask_calibration_settings", lambda parent: CalibrationChoice(128, None)
        )
        shown = Mock()
        monkeypatch.setattr(app.messagebox, "showerror", shown)
        gui._calibrate()
        shown.assert_called_once()
        assert "bad speed" in shown.call_args.args[1]

    def test_stm_path_unchanged(self, monkeypatch):
        gui = _gui(stm=True)
        gui._joint_rows = {"FLHL": object()}
        ask_settings = Mock()
        monkeypatch.setattr(app, "ask_calibration_settings", ask_settings)
        monkeypatch.setattr(app.simpledialog, "askstring", lambda *a, **k: "flhl")
        info = Mock()
        monkeypatch.setattr(app.messagebox, "showinfo", info)
        gui._calibrate()
        ask_settings.assert_not_called()
        gui._mcu.send_calibration_for_joint.assert_called_once_with("FLHL")
        gui._mcu.send_command_calibrate.assert_not_called()
        info.assert_called_once()


class TestCalibrateWhileARunIsInProgress:
    """R1-22: the modal dialog must not open (and block Hold All) during a run."""

    def _gui(self, lines):
        gui = _gui()
        gui._neutral_text = ""
        gui._neutral_until = 0.0
        gui._status_var = _FakeStringVar("Connected")
        gui._mcu.recent_cal_lines.return_value = lines
        return gui

    def _start(self, gui, monkeypatch, choice=CalibrationChoice(128, "SEEK")):
        ask = Mock(return_value=choice)
        monkeypatch.setattr(app, "ask_calibration_settings", ask)
        gui._calibrate()
        return ask

    def test_second_press_during_a_run_does_not_open_the_dialog(self, monkeypatch):
        gui = self._gui([])
        self._start(gui, monkeypatch)
        gui._mcu.recent_cal_lines.return_value = [
            "CAL ROBOT START pwm=128 passes=SEEK",
            "CAL FRONT MOVE FLKL curl",
        ]
        ask = self._start(gui, monkeypatch)
        ask.assert_not_called()
        gui._mcu.send_command_calibrate.assert_called_once_with(128, "SEEK")
        assert gui._status_var.get() == app.CALIBRATION_RUNNING_NOTE
        assert "Hold All" in app.CALIBRATION_RUNNING_NOTE
        assert gui._neutral_until > time.time()

    def test_no_reply_yet_still_counts_as_running(self, monkeypatch):
        gui = self._gui([])
        self._start(gui, monkeypatch)
        self._start(gui, monkeypatch).assert_not_called()

    @pytest.mark.parametrize(
        "end",
        ["CAL ROBOT DONE", "CAL ROBOT FAIL FRONT: FLKL no motion", "CAL ROBOT ABORTED H"],
    )
    def test_robot_end_line_allows_a_new_run(self, monkeypatch, end):
        gui = self._gui([])
        self._start(gui, monkeypatch)
        gui._mcu.recent_cal_lines.return_value = [end, "CAL LEFT ABORTED H"]
        self._start(gui, monkeypatch).assert_called_once_with(gui)
        assert gui._mcu.send_command_calibrate.call_count == 2

    def test_hold_all_ends_the_wait(self, monkeypatch):
        gui = self._gui([])
        self._start(gui, monkeypatch)
        gui._hold_all()
        gui._mcu.send_command_joints_hold.assert_called_once_with()
        self._start(gui, monkeypatch).assert_called_once_with(gui)

    def test_unsent_command_does_not_arm_the_guard(self, monkeypatch):
        gui = self._gui([])
        gui._mcu.send_command_calibrate.return_value = None  # no open port
        self._start(gui, monkeypatch)
        self._start(gui, monkeypatch).assert_called_once_with(gui)

    def test_cancelled_dialog_does_not_arm_the_guard(self, monkeypatch):
        gui = self._gui([])
        self._start(gui, monkeypatch, choice=None)
        self._start(gui, monkeypatch).assert_called_once_with(gui)

    def test_rejected_speed_does_not_arm_the_guard(self, monkeypatch):
        gui = self._gui([])
        gui._mcu.send_command_calibrate.side_effect = ValueError("bad speed")
        monkeypatch.setattr(app.messagebox, "showerror", Mock())
        self._start(gui, monkeypatch)
        self._start(gui, monkeypatch).assert_called_once_with(gui)

    def test_sdk_without_cal_lines_waits_for_hold_all(self, monkeypatch):
        gui = self._gui([])
        gui._mcu = Mock(spec=["send_command_calibrate", "send_command_joints_hold"])
        self._start(gui, monkeypatch)
        self._start(gui, monkeypatch).assert_not_called()
        gui._hold_all()
        self._start(gui, monkeypatch).assert_called_once_with(gui)

    def test_non_list_cal_lines_count_as_running(self, monkeypatch):
        gui = self._gui(Mock())
        self._start(gui, monkeypatch)
        self._start(gui, monkeypatch).assert_not_called()

    def test_hold_all_when_disconnected_sends_nothing(self):
        gui = _gui(connected=False)
        gui._cal_awaiting_end = True
        gui._hold_all()
        gui._mcu.send_command_joints_hold.assert_not_called()


def _neutral_gui(targets, skipped, connected=True):
    gui = _gui(connected=connected)
    gui._neutral_text = ""
    gui._neutral_until = 0.0
    gui._status_var = _FakeStringVar("Connected")
    gui._mcu.neutral_targets.return_value = (targets, skipped)
    return gui


ALL_HIPS_AND_KNEES = [
    n for _, names in JOINT_GROUP_NAMES for n in names if n.endswith(("HL", "KL"))
]


class TestNeutralButton:
    def test_sends_only_what_the_sdk_says_is_ready(self):
        targets = {"FLHL": 0.5, "FLKL": 0.5}
        gui = _neutral_gui(targets, ["FRHL"])
        gui._neutral()
        gui._mcu.neutral_targets.assert_called_once_with(0.5)
        gui._mcu.send_command_joints.assert_called_once_with(targets)

    def test_never_sends_a_yaw(self):
        from firmware.krabby_mcu import KrabbyMCUSDK
        from firmware.interfaces.joint_telemetry import JointTelemetry

        sdk = KrabbyMCUSDK(target=Mock())
        for _, names in JOINT_GROUP_NAMES:
            for n in names:
                sdk.joints[n] = JointTelemetry(
                    name=n, pos=0.3, pot=500, current=0, en=(0, 0), pwm=(0, 0), saf=0
                )
        gui = _gui()
        gui._neutral_text = ""
        gui._neutral_until = 0.0
        gui._status_var = _FakeStringVar()
        gui._mcu = sdk
        sdk.ser = Mock(is_open=True)
        gui._neutral()
        line = sdk.ser.write.call_args.args[0].decode()
        sent = line.split()[1::2]
        assert sorted(sent) == sorted(ALL_HIPS_AND_KNEES)
        assert not any(name.endswith("HY") for name in sent)

    def test_skipped_joints_are_named_on_the_status_bar(self):
        gui = _neutral_gui({"FLHL": 0.5}, ["FLKL", "RLHL"])
        gui._neutral()
        assert gui._status_var.get() == (
            "Neutral sent to 1 joints; no position, skipped: FLKL, RLHL"
        )
        assert gui._neutral_until > time.time()

    def test_nothing_ready_sends_nothing_and_says_so(self):
        gui = _neutral_gui({}, ALL_HIPS_AND_KNEES)
        gui._neutral()
        gui._mcu.send_command_joints.assert_not_called()
        assert gui._status_var.get() == (
            "Neutral: no hip or knee has a position yet - run Calibrate (Curl only)"
        )

    def test_nothing_ready_with_no_saved_stroke_asks_for_the_full_run(self):
        # R1-20: after a fresh flash and a Curl only every hip and knee is "s";
        # repeating Curl only could never give them a position.
        gui = _neutral_gui({}, ALL_HIPS_AND_KNEES)
        gui._mcu.sensor_states.return_value = {n: "s" for n in ALL_HIPS_AND_KNEES}
        gui._neutral()
        assert gui._status_var.get() == (
            "Neutral: no hip or knee has a position yet - run Calibrate (full calibration)"
        )

    def test_nothing_ready_and_only_unhomed_asks_for_curl_only(self):
        gui = _neutral_gui({}, ALL_HIPS_AND_KNEES)
        gui._mcu.sensor_states.return_value = {n: "e" for n in ALL_HIPS_AND_KNEES}
        gui._neutral()
        assert gui._status_var.get().endswith("run Calibrate (Curl only)")

    def test_all_ready_is_silent_and_clears_an_old_note(self):
        gui = _neutral_gui({n: 0.5 for n in ALL_HIPS_AND_KNEES}, [])
        gui._neutral_text = "old"
        gui._status_var.set("Connected")
        gui._neutral()
        gui._mcu.send_command_joints.assert_called_once()
        assert gui._neutral_text == ""
        assert gui._status_var.get() == "Connected"

    def test_not_connected_does_nothing(self):
        gui = _neutral_gui({"FLHL": 0.5}, [], connected=False)
        gui._neutral()
        gui._mcu.neutral_targets.assert_not_called()
        gui._mcu.send_command_joints.assert_not_called()


class TestNeutralDuringCalibration:
    """Host check of 78e18136 (e): Neutral is refused while a C sent from this
    window has no robot end line and no Hold All: its targets would move joints
    the run is driving or has parked."""

    def _running(self, monkeypatch, lines=()):
        gui = _neutral_gui({"FLHL": 0.5, "FLKL": 0.5}, [])
        gui._mcu.recent_cal_lines.return_value = list(lines)
        monkeypatch.setattr(app, "ask_calibration_settings", Mock(return_value=CalibrationChoice(128, None)))
        gui._calibrate()
        gui._mcu.send_command_calibrate.assert_called_once_with(128, None)
        return gui

    def test_refused_while_a_run_is_in_progress(self, monkeypatch):
        gui = self._running(monkeypatch)
        gui._mcu.recent_cal_lines.return_value = ["CAL ROBOT START pwm=128 passes=SEEK,SWEEP"]
        gui._neutral()
        gui._mcu.neutral_targets.assert_not_called()
        gui._mcu.send_command_joints.assert_not_called()
        assert gui._status_var.get() == app.NEUTRAL_DURING_CALIBRATION_NOTE
        assert "Hold All" in app.NEUTRAL_DURING_CALIBRATION_NOTE
        assert gui._neutral_until > time.time()

    @pytest.mark.parametrize(
        "end",
        ["CAL ROBOT DONE", "CAL ROBOT FAIL FRONT: FLKL no motion", "CAL ROBOT ABORTED H"],
    )
    def test_sent_once_the_run_has_ended(self, monkeypatch, end):
        gui = self._running(monkeypatch)
        gui._mcu.recent_cal_lines.return_value = [end]
        gui._neutral()
        gui._mcu.send_command_joints.assert_called_once_with({"FLHL": 0.5, "FLKL": 0.5})

    def test_sent_after_hold_all(self, monkeypatch):
        gui = self._running(monkeypatch)
        gui._hold_all()
        gui._neutral()
        gui._mcu.send_command_joints.assert_called_once_with({"FLHL": 0.5, "FLKL": 0.5})

    def test_no_run_sent_from_this_window_does_not_block(self):
        gui = _neutral_gui({"FLHL": 0.5}, [])
        gui._neutral()
        gui._mcu.send_command_joints.assert_called_once_with({"FLHL": 0.5})


class TestCalStatus:
    def test_empty_when_no_line(self):
        assert cal_status_text(None) == ("", "")
        assert cal_status_text("") == ("", "")

    def test_running_line_has_default_colour(self):
        text, colour = cal_status_text("CAL FRONT MOVE FLKL curl")
        assert text == "Calibration: CAL FRONT MOVE FLKL curl"
        assert colour == ""

    def test_failure_is_red(self):
        for line in (
            "CAL ROBOT FAIL LEFT: no CAL report for 135 s",
            "CAL FRONT ABORTED H",
            "CAL ROBOT BUSY",
        ):
            assert cal_status_text(line)[1] == app.STATE_COLOR_STALE

    def test_done_is_green(self):
        assert cal_status_text("CAL ROBOT DONE")[1] == app.STATE_COLOR_OK

    def test_gui_shows_latest_line(self):
        gui = _gui()
        gui._cal_var = _FakeStringVar()
        gui._cal_label = _FakeWidget()
        gui._mcu.recent_cal_lines = lambda: [
            "CAL FRONT MOVE FLHL curl",
            "CAL FRONT FAIL FLHL pot not connected",
            "CAL ROBOT FAIL FRONT: FLHL pot not connected",
        ]
        gui._update_cal_status()
        assert gui._cal_var.get() == (
            "Calibration: CAL ROBOT FAIL FRONT: FLHL pot not connected"
        )
        assert gui._cal_label.kwargs["foreground"] == app.STATE_COLOR_STALE

    def test_gui_keeps_the_robot_result_when_a_follower_line_follows(self):
        # The leader sends H before printing its FAIL; a still-running follower
        # answers "CAL LEFT ABORTED H" just after it.
        gui = _gui()
        gui._cal_var = _FakeStringVar()
        gui._cal_label = _FakeWidget()
        gui._mcu.recent_cal_lines = lambda: [
            "CAL LEFT MOVE RLHL uncurl",
            "CAL ROBOT FAIL LEFT: no CAL report for 135 s",
            "CAL LEFT ABORTED H",
        ]
        gui._update_cal_status()
        assert gui._cal_var.get() == (
            "Calibration: CAL ROBOT FAIL LEFT: no CAL report for 135 s"
        )

    def test_no_sens_refusal_is_shown_whole_in_red(self):
        line = (
            "CAL ROBOT FAIL LEFT: no SENS: flash all three boards with the same "
            "firmware, or wait a few seconds after connecting"
        )
        gui = _gui()
        gui._cal_var = _FakeStringVar()
        gui._cal_label = _FakeWidget()
        gui._mcu.recent_cal_lines = lambda: ["CAL ROBOT BUSY", line]
        gui._update_cal_status()
        assert gui._cal_var.get() == f"Calibration: {line}"
        assert gui._cal_label.kwargs["foreground"] == app.STATE_COLOR_STALE

    def test_status_label_wraps_to_the_window_width(self):
        gui = _gui()
        gui._cal_label = _FakeWidget()
        gui._wrap_cal_label(SimpleNamespace(width=900))
        assert gui._cal_label.kwargs["wraplength"] == 900 - app.CAL_LABEL_PADDING_PX
        gui._wrap_cal_label(SimpleNamespace(width=50))
        assert gui._cal_label.kwargs["wraplength"] == app.CAL_LABEL_MIN_WRAP_PX

    def test_gui_shows_progress_until_the_run_ends(self):
        gui = _gui()
        gui._cal_var = _FakeStringVar()
        gui._cal_label = _FakeWidget()
        gui._mcu.recent_cal_lines = lambda: [
            "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
            "CAL FRONT MOVE FLKL curl",
        ]
        gui._update_cal_status()
        assert gui._cal_var.get() == "Calibration: CAL FRONT MOVE FLKL curl"
        assert gui._cal_label.kwargs["foreground"] == ""

    def test_gui_clears_when_no_line(self):
        gui = _gui()
        gui._cal_var = _FakeStringVar("stale")
        gui._cal_label = _FakeWidget()
        gui._mcu.recent_cal_lines = lambda: []
        gui._update_cal_status()
        assert gui._cal_var.get() == ""

    def test_adapter_with_only_the_latest_line_is_shown(self):
        gui = _gui()
        gui._mcu = SimpleNamespace(latest_cal_line="CAL ROBOT DONE")
        gui._cal_var = _FakeStringVar()
        gui._cal_label = _FakeWidget()
        gui._update_cal_status()
        assert gui._cal_var.get() == "Calibration: CAL ROBOT DONE"

    def test_sdk_without_cal_support_is_tolerated(self):
        gui = _gui()
        gui._mcu = SimpleNamespace()  # e.g. an adapter with no latest_cal_line
        gui._cal_var = _FakeStringVar("old")
        gui._cal_label = _FakeWidget()
        gui._update_cal_status()
        assert gui._cal_var.get() == ""

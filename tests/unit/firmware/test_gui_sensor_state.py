"""Headless GUI telemetry display tests."""

import math
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("tkinter")
pytest.importorskip("serial")

from firmware.gui import app
from firmware.gui.app import (
    DASH,
    JOINT_GRID_HEADERS,
    STATE_COLOR_MUTED,
    STATE_COLOR_OK,
    STATE_COLOR_PENDING,
    STATE_COLOR_STALE,
    ImuRow,
    JointRow,
    KrabbyTestGUI,
    encoder_banner,
    joint_cells,
    neutral_note,
)
from firmware.interfaces.imu_telemetry import ImuTelemetry
from firmware.interfaces.joint_telemetry import JointTelemetry


def _sample(valid=True):
    return ImuTelemetry(
        accel=(0.0, 0.0, 9.80665), gyro=(0.0, 0.0, 0.0), temp_c=24.0, valid=valid
    )


class TestResolveImuState:
    def test_absent_when_no_sample(self):
        text, col = ImuRow.resolve_state(None)
        assert text == "—"
        assert col == ""

    def test_stale_when_sensor_reading_is_invalid(self):
        text, col = ImuRow.resolve_state(_sample(valid=False))
        assert text == "STALE"
        assert col == STATE_COLOR_STALE

    def test_fresh_when_sensor_reading_is_valid(self):
        text, col = ImuRow.resolve_state(_sample(valid=True))
        assert text == "fresh"
        assert col == STATE_COLOR_OK


class _FakeStringVar:
    """Stand-in for tk.StringVar — CI has no $DISPLAY so real Tk cannot start."""

    def __init__(self, value="---"):
        self._value = value

    def set(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeLabel:
    """Stand-in for a ttk.Label that keeps its grid placement and configure() calls."""

    def __init__(self, parent=None, **kwargs):
        self.kwargs = dict(kwargs)
        self.grid_kwargs = {}

    def grid(self, **kwargs):
        self.grid_kwargs = kwargs

    def configure(self, **kwargs):
        self.kwargs.update(kwargs)

    def pack(self, **kwargs):
        pass

    def bind(self, *args):
        pass


def _bare_row():
    """A JointRow without Tk: only the variables and labels update_from_telemetry uses."""
    row = JointRow.__new__(JointRow)
    row.var_sensor = _FakeStringVar("")
    row.var_pos = _FakeStringVar()
    row.var_pot = _FakeStringVar()
    row.var_cur = _FakeStringVar()
    row.var_pwm = _FakeStringVar()
    row.var_hall = _FakeStringVar()
    row.lbl_sensor = _FakeLabel()
    row.lbl_pot = _FakeLabel()
    return row


def _joint(name="FLHL", pos=0.4321, pot=512, saf=0):
    return JointTelemetry(
        name=name, pos=pos, pot=pot, current=30, en=(1, 1), pwm=(0, 200), saf=saf
    )


class TestJointRowCurrentColumn:
    def test_cur_shows_telemetry_current_field(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=0.5))
        assert row.var_cur.get() == "30"

    def test_pwm_and_hall_cells(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(saf=-212))
        assert row.var_pwm.get() == "L0 R200"
        assert row.var_hall.get() == "-212"  # signed hall count


# (state, Sensor text, Sensor colour, Pos text, Pot text, Pot colour) for a joint at
# pos 0.4321 with pot reading 512 (design 8.3).
SENSOR_CASES = [
    ("P", "POT", "", "0.432", "512", ""),
    ("E", "ENC", "", "0.432", DASH, STATE_COLOR_MUTED),
    ("e", "ENC unhomed", STATE_COLOR_PENDING, DASH, DASH, STATE_COLOR_MUTED),
    ("s", "ENC no stroke", STATE_COLOR_PENDING, DASH, DASH, STATE_COLOR_MUTED),
    ("x", "ENC home lost", STATE_COLOR_STALE, DASH, DASH, STATE_COLOR_MUTED),
    ("?", "?", STATE_COLOR_MUTED, DASH, "512", ""),
    (None, "", "", "0.432", "512", ""),
]


class TestSensorAndPosColumns:
    @pytest.mark.parametrize("state,sensor,colour,pos,pot,pot_colour", SENSOR_CASES)
    def test_cells_for_every_state(self, state, sensor, colour, pos, pot, pot_colour):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(), state)
        assert row.var_sensor.get() == sensor
        assert row.lbl_sensor.kwargs["foreground"] == colour
        assert row.var_pos.get() == pos
        assert row.var_pot.get() == pot
        assert row.lbl_pot.kwargs["foreground"] == pot_colour

    @pytest.mark.parametrize("state,sensor,colour,pos,pot,pot_colour", SENSOR_CASES)
    def test_joint_cells_matches_the_row(self, state, sensor, colour, pos, pot, pot_colour):
        cells = joint_cells(state, _joint())
        assert tuple(cells) == (sensor, colour, pos, pot, pot_colour)

    def test_no_sens_is_exactly_the_legacy_display(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=0.25, pot=700))
        assert row.var_sensor.get() == ""
        assert row.var_pot.get() == "700"
        assert row.var_pos.get() == "0.250"
        assert row.lbl_pot.kwargs["foreground"] == ""

    @pytest.mark.parametrize("state", [None, "P", "E"])
    @pytest.mark.parametrize("pos", [math.nan, math.inf])
    def test_non_finite_position_shows_a_dash(self, state, pos):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=pos), state)
        assert row.var_pos.get() == DASH

    @pytest.mark.parametrize("state", ["e", "s", "x", "?"])
    def test_finite_position_is_hidden_when_the_joint_has_no_position(self, state):
        # An unhomed encoder joint reports nan today, but a stale finite value must
        # never be shown as a position.
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=0.9), state)
        assert row.var_pos.get() == DASH

    def test_positions_use_three_decimals(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=0.5), "E")
        assert row.var_pos.get() == "0.500"
        JointRow.update_from_telemetry(row, _joint(pos=1.0), "P")
        assert row.var_pos.get() == "1.000"

    def test_state_change_updates_the_same_row(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, _joint(pos=math.nan), "e")
        assert row.var_sensor.get() == "ENC unhomed"
        JointRow.update_from_telemetry(row, _joint(pos=0.5), "E")
        assert row.var_sensor.get() == "ENC"
        assert row.lbl_sensor.kwargs["foreground"] == ""
        assert row.var_pos.get() == "0.500"

    def test_sensor_shows_before_the_first_telemetry(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, None, "e")
        assert row.var_sensor.get() == "ENC unhomed"
        assert row.lbl_sensor.kwargs["foreground"] == STATE_COLOR_PENDING
        assert row.var_pos.get() == "---"  # untouched: no telemetry yet

    def test_no_telemetry_and_no_state_changes_nothing(self):
        row = _bare_row()
        JointRow.update_from_telemetry(row, None)
        assert row.var_sensor.get() == ""
        assert row.var_pos.get() == "---"
        assert row.var_pot.get() == "---"


class TestJointGridColumns:
    def test_header_order(self):
        assert JOINT_GRID_HEADERS == (
            "Joint", "Retract", "Extend", "Sensor", "Pos", "Pot", "Cur", "PWM", "Hall",
        )

    def test_row_widgets_sit_under_their_headers(self, monkeypatch):
        placed = {}

        class _Label(_FakeLabel):
            def grid(self, **kwargs):
                super().grid(**kwargs)
                variable = self.kwargs.get("textvariable")
                if variable is not None:
                    placed[variable] = kwargs["column"]

        class _Button(_FakeLabel):
            pass

        monkeypatch.setattr(app, "ttk", SimpleNamespace(Label=_Label, Button=_Button))
        monkeypatch.setattr(app.tk, "StringVar", _FakeStringVar)
        row = JointRow(object(), "FLHL", 1, lambda joint, pwm: None)
        by_header = {
            "Sensor": row.var_sensor,
            "Pos": row.var_pos,
            "Pot": row.var_pot,
            "Cur": row.var_cur,
            "PWM": row.var_pwm,
            "Hall": row.var_hall,
        }
        for header, variable in by_header.items():
            assert placed[variable] == JOINT_GRID_HEADERS.index(header), header
        assert row.lbl_sensor.grid_kwargs["column"] == 3
        assert row.lbl_pot.grid_kwargs["column"] == 5


class TestEncoderBanner:
    def test_no_states_no_banner(self):
        assert encoder_banner({}) == ("", "")

    def test_ready_and_pot_states_do_not_warn(self):
        states = {"FLHL": "E", "FLKL": "P", "RLHL": "E"}
        assert encoder_banner(states) == ("", "")

    def test_unknown_hips_and_knees_ask_for_the_full_calibration(self):
        # R1-19: after a fresh flash (or an ignored, corrupt or invalid record) every
        # hip and knee is "?" and needs a full calibration; the banner must say so.
        states = {n: "?" for _, names in app.JOINT_GROUP_NAMES for n in names}
        text, colour = encoder_banner(states)
        hips_and_knees = [
            n for _, names in app.JOINT_GROUP_NAMES for n in names if n[-2:] in ("HL", "KL")
        ]
        assert text == (
            f"Sensor not yet detected: {', '.join(hips_and_knees)}. "
            "Run Calibrate > full calibration"
        )
        assert colour == STATE_COLOR_PENDING

    def test_unknown_takes_precedence_over_curl_only(self):
        text, colour = encoder_banner({"FLHL": "?", "FLKL": "e", "FRHL": "E"})
        assert text == (
            "Sensor not yet detected: FLHL. Encoder joints not ready: FLKL. "
            "Run Calibrate > full calibration"
        )
        assert colour == STATE_COLOR_PENDING

    def test_unknown_with_home_lost_is_red(self):
        text, colour = encoder_banner({"FLHL": "?", "RLKL": "x"})
        assert text.startswith(
            "Sensor not yet detected: FLHL. Encoder joints not ready: RLKL."
        )
        assert colour == STATE_COLOR_STALE

    def test_unknown_yaws_never_count(self):
        assert encoder_banner({"FLHY": "?", "FRHY": "?"}) == ("", "")

    def test_unhomed_names_the_joints_and_asks_for_curl_only(self):
        text, colour = encoder_banner({"FLKL": "e", "FLHL": "e", "FRHL": "E"})
        assert text == (
            "Encoder joints not ready: FLHL, FLKL. Run Calibrate > Curl only"
        )
        assert colour == STATE_COLOR_PENDING

    def test_no_stroke_asks_for_the_full_calibration(self):
        text, colour = encoder_banner({"FLHL": "s", "FLKL": "e"})
        assert text == (
            "Encoder joints not ready: FLHL, FLKL. Run Calibrate > full calibration"
        )
        assert colour == STATE_COLOR_PENDING

    def test_home_lost_is_red(self):
        text, colour = encoder_banner({"FLHL": "x"})
        assert text == "Encoder joints not ready: FLHL. Run Calibrate > Curl only"
        assert colour == STATE_COLOR_STALE

    def test_any_home_lost_makes_it_red_even_with_no_stroke(self):
        text, colour = encoder_banner({"FLHL": "s", "RLKL": "x"})
        assert colour == STATE_COLOR_STALE
        assert text.endswith("full calibration")

    def test_yaws_never_count(self):
        assert encoder_banner({"FLHY": "e", "MRHY": "x", "RRHY": "s"}) == ("", "")

    def test_names_follow_the_grid_order(self):
        states = {name: "e" for _, names in app.JOINT_GROUP_NAMES for name in names}
        text, _ = encoder_banner(states)
        listed = text.split(": ")[1].split(". ")[0].split(", ")
        assert listed == [
            n for _, names in app.JOINT_GROUP_NAMES for n in names if n[-2:] in ("HL", "KL")
        ]


class TestNeutralNote:
    def test_nothing_sent(self):
        assert neutral_note(0, ["FLHL", "FLKL"]) == (
            "Neutral: no hip or knee has a position yet - run Calibrate (Curl only)"
        )

    def test_nothing_sent_and_only_unhomed_asks_for_curl_only(self):
        assert neutral_note(0, ["FLHL", "FLKL"], {"FLHL": "e", "FLKL": "x"}) == (
            "Neutral: no hip or knee has a position yet - run Calibrate (Curl only)"
        )

    @pytest.mark.parametrize("state", ["s", "?"])
    def test_nothing_sent_with_no_stroke_or_unknown_asks_for_the_full_run(self, state):
        # R1-20: Curl only can never give an "s" or "?" joint a position.
        note = neutral_note(0, ["FLHL", "FLKL"], {"FLHL": "e", "FLKL": state})
        assert note == (
            "Neutral: no hip or knee has a position yet - run Calibrate (full calibration)"
        )

    def test_advice_looks_only_at_the_skipped_joints(self):
        note = neutral_note(0, ["FLHL"], {"FLHL": "e", "RRKL": "s"})
        assert note.endswith("(Curl only)")

    def test_partly_sent_lists_the_skipped(self):
        assert neutral_note(10, ["FLHL", "RLKL"]) == (
            "Neutral sent to 10 joints; no position, skipped: FLHL, RLKL"
        )

    def test_all_sent_is_silent(self):
        assert neutral_note(12, []) == ""


def _polling_gui(states=None, joints=None, *, sens_support=True, stm=False):
    """A KrabbyTestGUI with just enough state to run _poll_telemetry once."""
    gui = KrabbyTestGUI.__new__(KrabbyTestGUI)
    gui._stm = stm
    gui._connected = True
    gui._neutral_text = ""
    gui._neutral_until = 0.0
    names = {n for _, ns in app.JOINT_GROUP_NAMES for n in ns}
    gui._joint_rows = {n: Mock() for n in names}
    mcu = SimpleNamespace(
        joints=joints or {},
        imu=None,
        battery=None,
        role_last_seen={},
        recent_cal_lines=lambda: [],
        last_error=None,
        last_feedback_ts=time.time(),
        target="COM-test",
    )
    if sens_support:
        mcu.sensor_states = lambda: dict(states or {})
    gui._mcu = mcu
    gui._imu_row = Mock()
    gui._batt_row = Mock()
    gui._role_var = _FakeStringVar()
    gui._status_var = _FakeStringVar()
    gui._cal_var = _FakeStringVar()
    gui._cal_label = _FakeLabel()
    gui._banner_var = _FakeStringVar()
    gui._banner_label = _FakeLabel()
    gui.after = Mock()
    return gui


class TestPollShowsSensorState:
    def test_each_row_gets_its_own_state(self):
        gui = _polling_gui(
            states={"FLHL": "e", "FLKL": "E"}, joints={"FLHL": _joint("FLHL")}
        )
        gui._poll_telemetry()
        gui._joint_rows["FLHL"].update_from_telemetry.assert_called_once_with(
            gui._mcu.joints["FLHL"], "e"
        )
        gui._joint_rows["FLKL"].update_from_telemetry.assert_called_once_with(None, "E")
        gui._joint_rows["FRHL"].update_from_telemetry.assert_called_once_with(None, None)

    def test_banner_follows_the_states(self):
        gui = _polling_gui(states={"FLHL": "e", "FLKL": "s"})
        gui._poll_telemetry()
        assert gui._banner_var.get() == (
            "Encoder joints not ready: FLHL, FLKL. Run Calibrate > full calibration"
        )
        assert gui._banner_label.kwargs["foreground"] == STATE_COLOR_PENDING
        gui._mcu.sensor_states = lambda: {"FLHL": "E", "FLKL": "E"}
        gui._poll_telemetry()
        assert gui._banner_var.get() == ""
        assert gui._banner_label.kwargs["foreground"] == ""

    def test_fresh_flash_unknown_states_show_the_banner(self):
        gui = _polling_gui(states={"FLHL": "?", "FLKL": "?", "FLHY": "?"})
        gui._poll_telemetry()
        assert gui._banner_var.get() == (
            "Sensor not yet detected: FLHL, FLKL. Run Calibrate > full calibration"
        )
        assert gui._banner_label.kwargs["foreground"] == STATE_COLOR_PENDING

    def test_firmware_without_sens_shows_no_banner_and_no_states(self):
        gui = _polling_gui(sens_support=False)
        gui._poll_telemetry()
        assert gui._banner_var.get() == ""
        gui._joint_rows["FLHL"].update_from_telemetry.assert_called_once_with(None, None)

    def test_adapter_returning_a_non_dict_is_treated_as_no_states(self):
        gui = _polling_gui()
        gui._mcu.sensor_states = lambda: Mock()
        gui._poll_telemetry()
        assert gui._banner_var.get() == ""

    def test_stm_path_has_no_banner_and_still_polls(self):
        gui = _polling_gui(stm=True, sens_support=False)
        del gui._banner_var, gui._banner_label, gui._cal_var, gui._cal_label
        gui._poll_telemetry()  # must not touch the banner or cal widgets
        gui._joint_rows["FLHL"].update_from_telemetry.assert_called_once_with(None, None)

    def test_neutral_note_rides_on_the_status_line_then_expires(self):
        gui = _polling_gui()
        gui._neutral_text = "Neutral sent to 10 joints; no position, skipped: FLHL, FLKL"
        gui._neutral_until = time.time() + 60
        gui._poll_telemetry()
        assert gui._status_var.get() == (
            "Connected: COM-test | Neutral sent to 10 joints; no position, skipped: FLHL, FLKL"
        )
        gui._neutral_until = time.time() - 1
        gui._poll_telemetry()
        assert gui._status_var.get() == "Connected: COM-test"

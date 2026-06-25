"""Unit tests for the headless live-telemetry table rendered by the interactive
key-control menu (firmware/__main__.py). The render is a pure function so the
formatting that the bench operator reads (AC 1b: pot / current / Hall per joint)
is testable without a serial port or a terminal."""
from firmware.__main__ import (
    _err_line,
    _fmt_row,
    _render_telemetry,
    _selected_board,
)
from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.krabby_mcu import ErrorEvent


def _jt(name, pos=0.5, pot=512, current=0, saf=0, cal_state=0):
    return JointTelemetry(
        name=name, pos=pos, pot=pot, current=current,
        en=(0, 0), pwm=(0, 0), saf=saf, cal_state=cal_state,
    )


class TestCalStateColumn:
    def test_header_has_state(self):
        from firmware.__main__ import _TELEM_HEADER
        assert "CAL" in _TELEM_HEADER

    def test_partial_shown(self):
        row = _fmt_row("FRONT", ">", "FLHL", _jt("FLHL", cal_state=1))
        assert "PARTIAL" in row

    def test_full_shown(self):
        row = _fmt_row("FRONT", ">", "FLHL", _jt("FLHL", cal_state=2))
        assert "FULL" in row

    def test_uncal_default(self):
        row = _fmt_row("FRONT", ">", "FLHL", _jt("FLHL"))  # cal_state defaults to 0
        assert "UNCAL" in row


class TestParseCalStateToken:
    def test_ten_tokens_parses_cal_state(self):
        # name pos pot cur enL enR pwmL pwmR saf cal_state
        jts = JointTelemetry.parse_line("FRONT; FLHL 0.500 512 0 1 1 0 120 64 1;")
        assert len(jts) == 1 and jts[0].cal_state == 1 and jts[0].cal_state_name == "PARTIAL"

    def test_nine_tokens_still_parses_legacy(self):
        jts = JointTelemetry.parse_line("FRONT; FLHL 0.500 512 0 1 1 0 120 64;")
        assert len(jts) == 1 and jts[0].cal_state == 0  # default for old firmware

    def test_eight_tokens_rejected(self):
        assert JointTelemetry.parse_line("FRONT; FLHL 0.5 512 0 1 1 0 120;") == []


class TestSelectedBoard:
    def test_none_is_front(self):
        assert _selected_board(False, False) == "FRONT"

    def test_key1_is_left(self):
        assert _selected_board(True, False) == "LEFT"

    def test_key2_is_right(self):
        assert _selected_board(False, True) == "RIGHT"

    def test_both_is_all(self):
        assert _selected_board(True, True) == "ALL"


class TestFmtRow:
    def test_missing_joint_shows_dashes(self):
        row = _fmt_row("FRONT", ">", "FLHY", None)
        assert "FLHY" in row
        assert "----" in row  # no telemetry yet

    def test_live_joint_shows_values(self):
        row = _fmt_row("FRONT", ">", "FLHL", _jt("FLHL", pos=0.498, pot=511, current=38, saf=14))
        assert "FLHL" in row
        assert "0.498" in row
        assert "511" in row
        assert "38" in row
        assert "14" in row  # Hall edge count


class TestRenderTelemetry:
    def test_all_18_joints_present(self):
        frame = _render_telemetry({}, "FRONT")
        text = "\n".join(frame)
        # every joint from every board appears even with no telemetry yet
        for name in ("FLHY", "FRKL", "RLKL", "MLHY", "RRHL", "MRKL"):
            assert name in text

    def test_selected_board_is_marked(self):
        frame = _render_telemetry({}, "LEFT")
        # the LEFT board's header row carries the '>' selection marker
        left_row = next(ln for ln in frame if ln.startswith("LEFT"))
        assert ">" in left_row
        front_row = next(ln for ln in frame if ln.startswith("FRONT") and "JOINT" not in ln)
        assert ">" not in front_row

    def test_all_marks_every_board(self):
        frame = _render_telemetry({}, "ALL")
        for board in ("FRONT", "LEFT", "RIGHT"):
            row = next(ln for ln in frame if ln.startswith(board) and "JOINT" not in ln)
            assert ">" in row

    def test_live_value_rendered_in_frame(self):
        joints = {"FLHL": _jt("FLHL", pos=0.498, pot=511, current=38, saf=14)}
        text = "\n".join(_render_telemetry(joints, "FRONT"))
        assert "0.498" in text and "511" in text and "14" in text

    def test_status_and_err_lines_included(self):
        text = "\n".join(_render_telemetry({}, "FRONT", status="neutral sent", err_line="last ERR: FLKL motor_did_not_move"))
        assert "neutral sent" in text
        assert "motor_did_not_move" in text


class TestErrLine:
    def test_no_errors(self):
        mcu = type("M", (), {"get_errors": lambda self: []})()
        assert _err_line(mcu) == "last ERR: (none)"

    def test_latest_error_shown(self):
        events = [ErrorEvent("FLKL", "motor_did_not_move", 1.0), ErrorEvent("RRHL", "hall_no_edges", 2.0)]
        mcu = type("M", (), {"get_errors": lambda self: events})()
        assert _err_line(mcu) == "last ERR: RRHL hall_no_edges"

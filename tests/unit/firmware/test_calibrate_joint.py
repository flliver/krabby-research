"""Unit tests for the single-joint calibration command path (M17 Task 2)."""
import threading
import time

import pytest

from firmware.krabby_mcu import ALL_JOINT_NAMES, parse_cal_reply


class TestParseCalReply:
    def test_saved_reply(self):
        assert parse_cal_reply("CAL FLHL 212 987 saved") == {
            "joint": "FLHL", "min": 212, "max": 987, "ok": True}

    def test_fail_reply(self):
        assert parse_cal_reply("CAL FLHY FAIL no_stop") == {
            "joint": "FLHY", "why": "no_stop", "ok": False}

    def test_fail_range_reply(self):
        assert parse_cal_reply("CAL RRKL FAIL range_too_small") == {
            "joint": "RRKL", "why": "range_too_small", "ok": False}

    def test_non_cal_lines_return_none(self):
        assert parse_cal_reply("VER 1.0 main abc") is None
        assert parse_cal_reply("FRONT; FLHY 0.5 ...") is None
        assert parse_cal_reply("CALIBRATION COMPLETE") is None
        assert parse_cal_reply("") is None

    def test_malformed_cal_line_returns_none(self):
        assert parse_cal_reply("CAL FLHL notanumber 987 saved") is None
        assert parse_cal_reply("CAL FLHL") is None


class TestCalibrateJoint:
    def test_all_18_joints_valid(self):
        assert len(ALL_JOINT_NAMES) == 18

    def test_unknown_joint_raises_before_write(self, bare_sdk):
        with pytest.raises(ValueError, match="unknown joint"):
            bare_sdk.calibrate_joint("ELBOW")
        bare_sdk.ser.write.assert_not_called()

    def test_writes_wire_line(self, bare_sdk):
        bare_sdk.calibrate_joint("flhl", timeout=0.05)  # lowercase accepted
        bare_sdk.ser.write.assert_called_once_with(b"C FLHL\n")

    def test_returns_parsed_reply(self, bare_sdk):
        def deliver():
            time.sleep(0.05)
            bare_sdk._last_cal_line = "CAL FLHL 212 987 saved"

        t = threading.Thread(target=deliver)
        t.start()
        result = bare_sdk.calibrate_joint("FLHL", timeout=1.0)
        t.join()
        assert result == {"joint": "FLHL", "min": 212, "max": 987, "ok": True}

    def test_returns_fail_reply(self, bare_sdk):
        def deliver():
            time.sleep(0.05)
            bare_sdk._last_cal_line = "CAL FLHL FAIL no_stop"

        t = threading.Thread(target=deliver)
        t.start()
        result = bare_sdk.calibrate_joint("FLHL", timeout=1.0)
        t.join()
        assert result == {"joint": "FLHL", "why": "no_stop", "ok": False}

    def test_ignores_reply_for_other_joint(self, bare_sdk):
        def deliver():
            time.sleep(0.05)
            bare_sdk._last_cal_line = "CAL FRKL 100 900 saved"  # not the joint we asked for

        t = threading.Thread(target=deliver)
        t.start()
        result = bare_sdk.calibrate_joint("FLHL", timeout=0.3)
        t.join()
        assert result is None

    def test_times_out_returns_none(self, bare_sdk):
        assert bare_sdk.calibrate_joint("FLHL", timeout=0.05) is None

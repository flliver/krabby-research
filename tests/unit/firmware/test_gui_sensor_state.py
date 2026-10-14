"""Headless GUI telemetry display tests."""

import pytest

pytest.importorskip("tkinter")

from firmware.gui.app import (
    STATE_COLOR_OK,
    STATE_COLOR_STALE,
    ImuRow,
    JointRow,
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


class TestJointRowCurrentColumn:
    def test_cur_shows_telemetry_current_field(self):
        import tkinter as tk

        # StringVar needs a Tk root; CI/headless runners have none by default.
        root = tk.Tk()
        root.withdraw()
        try:
            row = JointRow.__new__(JointRow)
            row.var_pot = tk.StringVar(master=root, value="---")
            row.var_cur = tk.StringVar(master=root, value="---")
            row.var_pwm = tk.StringVar(master=root, value="---")
            row.var_hall = tk.StringVar(master=root, value="---")
            jt = JointTelemetry(
                name="FLHL",
                pos=0.5,
                pot=512,
                current=30,
                en=(1, 1),
                pwm=(0, 200),
                saf=0,
            )
            JointRow.update_from_telemetry(row, jt)
            assert row.var_cur.get() == "30"
        finally:
            root.destroy()

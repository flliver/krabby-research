"""Headless GUI IMU-state tests."""

from firmware.gui.app import (
    STATE_COLOR_OK,
    STATE_COLOR_STALE,
    ImuRow,
)
from firmware.interfaces.imu_telemetry import ImuTelemetry


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

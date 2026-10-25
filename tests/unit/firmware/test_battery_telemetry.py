"""BATT segment parsing and M16 nominal value ranges."""

from firmware.interfaces.battery_telemetry import BatteryTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.sim.battery_sim_defaults import (
    DEFAULT_BATT_A_V,
    DEFAULT_BATT_B_V,
    DEFAULT_PACK_A,
    DEFAULT_PACK_CHARGE_C,
    DEFAULT_PACK_V,
    DEFAULT_PACK_W,
    M16_CELL_V_MAX,
    M16_CELL_V_MIN,
    M16_PACK_V_MAX,
    M16_PACK_V_MIN,
)

GOLDEN_BATT_SEGMENT = (
    f";BATT {DEFAULT_PACK_V:.2f} {DEFAULT_PACK_A:.3f} {DEFAULT_PACK_W:.2f} "
    f"{DEFAULT_PACK_CHARGE_C:.1f} {DEFAULT_BATT_A_V:.2f} {DEFAULT_BATT_B_V:.2f} 0 0"
)


def test_golden_batt_segment_parses():
    tokens = GOLDEN_BATT_SEGMENT.lstrip(";").split()
    batt = BatteryTelemetry.from_tokens(tokens)
    assert batt is not None
    assert batt.pack_v == DEFAULT_PACK_V
    assert batt.pack_i == DEFAULT_PACK_A
    assert batt.divergence_flag == 0
    assert batt.power_state == 0


def test_golden_values_in_m16_ranges():
    batt = BatteryTelemetry.from_tokens(GOLDEN_BATT_SEGMENT.lstrip(";").split())
    assert batt is not None
    assert M16_PACK_V_MIN <= batt.pack_v <= M16_PACK_V_MAX
    assert M16_CELL_V_MIN <= batt.batt_a_v <= M16_CELL_V_MAX
    assert M16_CELL_V_MIN <= batt.batt_b_v <= M16_CELL_V_MAX
    assert abs((batt.batt_a_v + batt.batt_b_v) - batt.pack_v) < 0.05


def test_telemetry_frame_parses_imu_and_batt_together():
    line = (
        "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"
        ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
        + GOLDEN_BATT_SEGMENT
    )
    frame = TelemetryFrame.parse_line(line)
    assert frame.imu is not None
    assert frame.battery is not None
    assert frame.battery.pack_v == DEFAULT_PACK_V

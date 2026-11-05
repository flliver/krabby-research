"""Host I2C E2E: sim Wire → leader firmware → ;IMU / ;BATT or OLED GDDRAM."""

from __future__ import annotations

import pytest

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
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.i2c_shipping_scope import SHIPPING_I2C_DEVICES
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from tests.unit.firmware.i2c_sensor_e2e_ci import I2C_SENSOR_E2E_CI, NATIVE_CTEST_TARGETS

HOST_STATIONARY_IMU_SUFFIX = (
    ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
)

TELEMETRY_INTERVAL_MS = 50
OLED_DRAW_INTERVAL_MS = 250

REPO_CMAKE_NATIVE = (
    __import__("pathlib").Path(__file__).resolve().parents[3]
    / "tests/native/firmware/CMakeLists.txt"
)

IMU_DEVICE = next(d for d in SHIPPING_I2C_DEVICES if "LSM6DSO" in d.name)


@pytest.fixture(scope="module")
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return bridge


def _drain_usb_tx(host) -> None:
    while host.serial_usb_tx_available():
        host.serial_usb_tx_read()


def _read_usb_line(host) -> str:
    chars: list[str] = []
    while host.serial_usb_tx_available():
        byte = host.serial_usb_tx_read()
        if byte == ord("\n"):
            break
        if byte != ord("\r"):
            chars.append(chr(byte))
    return "".join(chars)


def _stage_front_board_sensors(host) -> None:
    config = SimulatedRobotConfig.front_board_rev3()
    for slot in config.slots:
        host.set_actuator_stops(slot.firmware_actuator_id, 100, 900)
        host.set_analog_read(slot.pin_pot, 500)
        host.set_analog_read(slot.pin_is, 12)


def _emit_front_telemetry_line(host) -> str:
    host.firmware_setup()
    _drain_usb_tx(host)
    host.advance_millis(TELEMETRY_INTERVAL_MS)
    host.firmware_loop_once()
    return _read_usb_line(host)


def test_i2c_sensor_e2e_ci_matrix_covers_all_sensors():
    keys = {row.key for row in I2C_SENSOR_E2E_CI}
    assert keys == {
        "wire",
        "imu",
        "oled",
        "battery",
        "nominal_defaults",
        "nonfatal_init",
    }


def test_i2c_sensor_e2e_native_targets_registered_in_cmake():
    cmake = REPO_CMAKE_NATIVE.read_text(encoding="utf-8")
    for target in NATIVE_CTEST_TARGETS:
        assert f"arduino/{target}" in cmake


def test_i2c_wire_reachable_via_firmware_imu_segment(host):
    """Firmware setup + IMU segment implies sim Wire served LSM6DSO reads."""
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_front_telemetry_line(host)
    assert ";IMU " in line
    assert line.endswith(" 1") or " 1" in line.split(";IMU")[-1]


def test_imu_telemetry_end_to_end(host):
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_front_telemetry_line(host)
    frame = TelemetryFrame.parse_line(line)
    assert frame.imu is not None
    assert frame.imu.valid == 1
    assert frame.imu.accel[2] == pytest.approx(9.806, abs=0.01)
    assert HOST_STATIONARY_IMU_SUFFIX in line


def test_oled_framebuffer_end_to_end(host):
    """Firmware loop drives SSD1306 GDDRAM over sim Wire (jog→glyph in native test)."""
    host.reset()
    _stage_front_board_sensors(host)
    host.firmware_setup()
    host.update_all()
    host.advance_millis(OLED_DRAW_INTERVAL_MS)
    host.firmware_loop_once()
    assert host.oled_wire_data_bytes_received() > 0
    assert host.oled_framebuffer_crc32() != 0


def test_battery_telemetry_end_to_end(host):
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_front_telemetry_line(host)
    frame = TelemetryFrame.parse_line(line)
    assert frame.battery is not None
    assert frame.battery.pack_v == pytest.approx(DEFAULT_PACK_V, abs=0.01)
    assert frame.battery.pack_i == pytest.approx(DEFAULT_PACK_A, abs=0.001)


def test_battery_nominal_defaults_on_wire(host):
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_front_telemetry_line(host)
    frame = TelemetryFrame.parse_line(line)
    assert frame.battery is not None
    batt = frame.battery
    assert M16_PACK_V_MIN <= batt.pack_v <= M16_PACK_V_MAX
    assert M16_CELL_V_MIN <= batt.batt_a_v <= M16_CELL_V_MAX
    assert M16_CELL_V_MIN <= batt.batt_b_v <= M16_CELL_V_MAX
    assert batt.pack_w == pytest.approx(DEFAULT_PACK_W, abs=0.05)
    assert batt.pack_charge == pytest.approx(DEFAULT_PACK_CHARGE_C, rel=0, abs=0.1)
    assert batt.batt_a_v == pytest.approx(DEFAULT_BATT_A_V, abs=0.01)
    assert batt.batt_b_v == pytest.approx(DEFAULT_BATT_B_V, abs=0.01)


def test_missing_imu_nonfatal(host):
    host.reset()
    host.i2c_set_device_present(IMU_DEVICE.primary_address, False)
    if IMU_DEVICE.alternate_address is not None:
        host.i2c_set_device_present(IMU_DEVICE.alternate_address, False)
    _stage_front_board_sensors(host)
    line = _emit_front_telemetry_line(host)
    assert line.startswith("FRONT;")
    assert ";IMU 0.000 0.000 0.000 0.0000 0.0000 0.0000 0.0 0" in line
    host.handle_jog("FLHY", 80)
    host.firmware_loop_once()
    assert host.pin_digital(22) == 1


@pytest.mark.parametrize("mode", ["front", "three_board", "single_leader"])
def test_runtime_battery_inputs_use_driver_in_all_robot_modes(host, mode):
    host.reset()
    if mode == "three_board":
        host.enable_three_board_firmware_loop()
    elif mode == "single_leader":
        host.enable_single_leader_eighteen_joint()
    # Initialize the selected topology before setting charge: the real driver
    # resets the accumulator at boot.
    host.firmware_loop_once()
    _drain_usb_tx(host)
    host.set_battery_inputs(
        pack_volts=26.25, pack_amps=-3.125,
        battery_a_volts=12.75, charge_coulombs=-1000,
    )
    host.advance_millis(50)
    host.firmware_loop_once()
    frames = []
    while host.serial_usb_tx_available():
        line = _read_usb_line(host)
        frames.append((TelemetryFrame.role_from_line(line), TelemetryFrame.parse_line(line)))
    battery_frames = [(role, frame.battery) for role, frame in frames if frame.battery]
    assert len(battery_frames) == 1
    role, battery = battery_frames[0]
    assert role == "FRONT"
    assert battery.pack_volts == pytest.approx(26.25, abs=0.01)
    assert battery.pack_current_amperes == pytest.approx(-3.125, abs=0.01)
    # INA228 power is an unsigned magnitude, unlike the deleted V*I shortcut.
    assert battery.pack_power_watts == pytest.approx(82.03125, abs=0.1)
    assert battery.pack_charge_coulombs == pytest.approx(-1000, abs=0.1)
    assert battery.battery_b_volts == pytest.approx(13.5, abs=0.01)
    assert battery.divergence and battery.pack_valid and battery.midpoint_valid


def test_battery_disconnect_recovery_and_oled_share_measurements(host):
    host.reset()
    host.firmware_setup()
    host.advance_millis(250)
    host.firmware_loop_once()
    initial_screen = host.oled_framebuffer_crc32()
    _drain_usb_tx(host)
    host.i2c_set_device_present(0x41, False)
    host.advance_millis(250)
    host.firmware_loop_once()
    battery = TelemetryFrame.parse_line(_read_usb_line(host)).battery
    assert battery is not None
    assert battery.pack_valid and not battery.midpoint_valid
    assert battery.divergence
    assert host.oled_framebuffer_crc32() != initial_screen
    host.i2c_set_device_present(0x41, True)
    host.advance_millis(250)
    host.firmware_loop_once()
    battery = TelemetryFrame.parse_line(_read_usb_line(host)).battery
    assert battery.pack_valid and battery.midpoint_valid
    assert host.oled_framebuffer_crc32() == initial_screen


def test_battery_inputs_reject_nonfinite_values_without_changing_device(host):
    host.reset()
    host.firmware_setup()
    for invalid in [float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite"):
            host.set_battery_inputs(pack_volts=invalid, pack_amps=1, battery_a_volts=13)
    battery = TelemetryFrame.parse_line(_emit_front_telemetry_line(host)).battery
    assert battery.pack_volts == pytest.approx(DEFAULT_PACK_V, abs=0.01)

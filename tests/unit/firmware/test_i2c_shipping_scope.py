"""Shipping I2C scope stays aligned with firmware headers."""

from pathlib import Path

from firmware.sim.i2c_shipping_scope import (
    I2cSimStatus,
    IMU_TELEMETRY_TAG,
    SHIPPING_I2C_DEVICES,
)

REPO = Path(__file__).resolve().parents[3]
IMU_CONSTANTS = REPO / "firmware/arduino/src/imu/imu_constants.h"
DISPLAY_CONSTANTS = REPO / "firmware/arduino/src/display/display_constants.h"
INA228_CONSTANTS = REPO / "firmware/arduino/src/power/ina228_constants.h"


def test_imu_addresses_match_imu_constants_h():
    text = IMU_CONSTANTS.read_text(encoding="utf-8")
    imu = next(d for d in SHIPPING_I2C_DEVICES if "LSM6DSO" in d.name)
    assert f"LSM6DSO_PRIMARY_ADDRESS = 0x{imu.primary_address:02X}" in text
    assert f"LSM6DSO_ALTERNATE_ADDRESS = 0x{imu.alternate_address:02X}" in text


def test_oled_address_matches_display_constants_h():
    text = DISPLAY_CONSTANTS.read_text(encoding="utf-8")
    oled = next(d for d in SHIPPING_I2C_DEVICES if "OLED" in d.name)
    assert f"SSD1306_I2C_ADDRESS = 0x{oled.primary_address:02X}" in text


def test_ina228_devices_marked_shipping_on_firmware():
    ina = [d for d in SHIPPING_I2C_DEVICES if "INA228" in d.name]
    assert len(ina) == 2
    assert all(d.status is I2cSimStatus.SHIPPING for d in ina)
    assert all(d.primary_address in (0x40, 0x41) for d in ina)


def test_ina228_addresses_match_ina228_constants_h():
    text = INA228_CONSTANTS.read_text(encoding="utf-8")
    pack = next(d for d in SHIPPING_I2C_DEVICES if d.primary_address == 0x40)
    mid = next(d for d in SHIPPING_I2C_DEVICES if d.primary_address == 0x41)
    assert f"INA228_PACK_ADDRESS = 0x{pack.primary_address:02X}" in text
    assert f"INA228_MIDPOINT_ADDRESS = 0x{mid.primary_address:02X}" in text


def test_imu_telemetry_tag_matches_firmware():
    text = (REPO / "firmware/arduino/src/telemetry.h").read_text(encoding="utf-8")
    assert f'IMU_TELEMETRY_TAG[] = "{IMU_TELEMETRY_TAG}"' in text


def test_operator_i2c_scope_doc_lists_lsm6dso_addresses():
    doc = (REPO / "firmware/sim/I2C_SIM_SCOPE.md").read_text(encoding="utf-8")
    assert "LSM6DSO" in doc and "0x6B" in doc and "0x6A" in doc
    assert "i2c_shipping_scope.py" in doc

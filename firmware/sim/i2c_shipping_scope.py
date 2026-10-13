"""Shipping-firmware I2C inventory for host simulation (follow shipping code, not older IMU docs).

Keep addresses in sync with:
- ``firmware/arduino/src/imu/imu_constants.h``
- ``firmware/arduino/src/display/display_constants.h``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class I2cSimStatus(str, Enum):
    SHIPPING = "in shipping firmware"
    NOT_WIRED = "not in firmware yet"
    SIM_ONLY = "sim / tooling only"


@dataclass(frozen=True, slots=True)
class I2cDeviceScope:
    name: str
    primary_address: int
    alternate_address: int | None
    leader_only: bool
    status: I2cSimStatus
    firmware_sources: tuple[str, ...]
    notes: str
    telemetry_segment: str | None = None


I2C_BUS_CLOCK_HZ = 100_000
I2C_SDA_PIN = 20
I2C_SCL_PIN = 21

SHIPPING_I2C_DEVICES: tuple[I2cDeviceScope, ...] = (
    I2cDeviceScope(
        name="LSM6DSO IMU (SparkFun 6DoF Qwiic)",
        primary_address=0x6B,
        alternate_address=0x6A,
        leader_only=True,
        status=I2cSimStatus.SHIPPING,
        firmware_sources=(
            "firmware/arduino/src/imu/lsm6dso_adapter.h",
            "firmware/arduino/arduino.ino (imuSetup, appendImuMeasurement)",
        ),
        notes="Some older docs describe BMI270 at 0x68/0x69; shipping uses SparkFun LSM6DSO.",
        telemetry_segment=";IMU <ax> <ay> <az> <gx> <gy> <gz> <temp_c> <valid>",
    ),
    I2cDeviceScope(
        name="Qwiic OLED 1.3\" (SSD1306 via SparkFun_Qwiic_OLED)",
        primary_address=0x3D,
        alternate_address=0x3C,
        leader_only=True,
        status=I2cSimStatus.SHIPPING,
        firmware_sources=(
            "firmware/arduino/src/display/ssd1306_adapter.h",
            "firmware/arduino/src/display/display_constants.h",
            "firmware/arduino/arduino.ino (oledRenderer)",
        ),
        notes="In-tree SparkFun_Qwiic_OLED driver; alternate jumper address 0x3C.",
        telemetry_segment=None,
    ),
    I2cDeviceScope(
        name="Pack INA228",
        primary_address=0x40,
        alternate_address=None,
        leader_only=True,
        status=I2cSimStatus.NOT_WIRED,
        firmware_sources=(),
        notes="Host sim probes 0x40; shipping Mega firmware does not include INA228 driver yet.",
        telemetry_segment=";BATT <pack_v> <pack_i> <pack_w> <pack_charge> <batt_a_v> <batt_b_v> <div> <pwr>",
    ),
    I2cDeviceScope(
        name="Midpoint INA228",
        primary_address=0x41,
        alternate_address=None,
        leader_only=True,
        status=I2cSimStatus.NOT_WIRED,
        firmware_sources=(),
        notes="Host sim probes 0x41 midpoint; OLED bars still use placeholders until firmware reads it.",
        telemetry_segment=None,
    ),
)

IMU_TELEMETRY_TAG = "IMU"
# Parsed by firmware/interfaces/imu_telemetry.py + TelemetryFrame.parse_line

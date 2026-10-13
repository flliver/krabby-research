"""Host I2C sensor paths → CI mapping (Wire → firmware → telemetry or OLED).

Native Unity targets: ``make -C firmware test-native``.
Python E2E: ``test_host_i2c_sensor_e2e.py`` via ``host_bridge``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SensorE2eKey = Literal[
    "wire",
    "imu",
    "oled",
    "battery",
    "nominal_defaults",
    "nonfatal_init",
]


@dataclass(frozen=True, slots=True)
class I2cSensorE2eCiRow:
    key: SensorE2eKey
    summary: str
    native_ctest_target: str
    python_e2e: str


I2C_SENSOR_E2E_CI: tuple[I2cSensorE2eCiRow, ...] = (
    I2cSensorE2eCiRow(
        "wire",
        "Sim Wire / addressed I2C with graceful absent-device handling",
        "test_host_i2c",
        "test_i2c_wire_reachable_via_firmware_imu_segment",
    ),
    I2cSensorE2eCiRow(
        "imu",
        "IMU register model → leader ;IMU telemetry segment",
        "test_host_firmware_loop",
        "test_imu_telemetry_end_to_end",
    ),
    I2cSensorE2eCiRow(
        "oled",
        "OLED health screen over sim Wire from live firmware state",
        "test_host_oled_wire",
        "test_oled_framebuffer_end_to_end",
    ),
    I2cSensorE2eCiRow(
        "battery",
        "INA228 ×2 → ;BATT telemetry frame",
        "test_host_firmware_loop",
        "test_battery_telemetry_end_to_end",
    ),
    I2cSensorE2eCiRow(
        "nominal_defaults",
        "Pack/cell values match host sim defaults and nominal ranges",
        "test_host_leader_battery",
        "test_battery_nominal_defaults_on_wire",
    ),
    I2cSensorE2eCiRow(
        "nonfatal_init",
        "Missing sensor: invalid/offline signaling; control loop continues",
        "test_host_nonfatal_sensor_init",
        "test_missing_imu_nonfatal",
    ),
)

NATIVE_CTEST_TARGETS = tuple({row.native_ctest_target for row in I2C_SENSOR_E2E_CI})

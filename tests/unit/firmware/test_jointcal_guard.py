"""Guard test for the per-joint calibration schema + sensor abstraction (M17 Task 2,
the JointCal/sensor-abstraction slice of 2a/2b/2d).

These lock in the firmware-side invariants that can't be bench-checked without a
re-cal on real hardware:
  - JointCalBlock is a SEPARATE EEPROM region (own magic/CRC at its own base addr),
    not a field of EepromLayout — so adding cal doesn't invalidate Task-1 role/serial;
  - the read site (getRawPos/getPos) applies the sensor-type select + direction flip;
  - an uncalibrated joint falls back to the legacy pot path (zero behaviour change).
"""
import re
from pathlib import Path

import pytest

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


@pytest.fixture(scope="module")
def eeprom() -> str:
    return (ARDUINO / "eeprom_layout.h").read_text()


@pytest.fixture(scope="module")
def actuator() -> str:
    return (ARDUINO / "actuator_manager.h").read_text()


class TestJointCalSchema:
    def test_magic_is_ca17(self, eeprom):
        assert re.search(r"JOINTCAL_MAGIC\s*=\s*0xCA17", eeprom)

    def test_separate_region_not_addr_zero(self, eeprom):
        m = re.search(r"JOINTCAL_BASE_ADDR\s*=\s*(\d+)", eeprom)
        assert m and int(m.group(1)) > 0, "joint cal must live at its own base addr, not 0"

    def test_no_overlap_guard(self, eeprom):
        assert "static_assert(sizeof(EepromLayout)" in eeprom, \
            "must static_assert the config struct can't overrun the cal region"

    def test_structs_and_io_defined(self, eeprom):
        assert re.search(r"struct\s+JointCal\b", eeprom)
        assert re.search(r"struct\s+JointCalBlock\b", eeprom)
        assert re.search(r"void\s+jointCalSave\s*\(", eeprom)
        assert re.search(r"bool\s+jointCalLoad\s*\(", eeprom)

    def test_sensor_type_enum(self, eeprom):
        assert re.search(r"enum\s+SensorType", eeprom)
        assert "SENSOR_POT" in eeprom and "SENSOR_HALL" in eeprom

    def test_load_validates_magic_schema_crc(self, eeprom):
        body = eeprom[eeprom.index("jointCalLoad"):]
        assert "JOINTCAL_MAGIC" in body
        assert "JOINTCAL_SCHEMA_VER" in body
        assert "eepromCrc32" in body


class TestSensorAbstraction:
    def test_getrawpos_selects_sensor_and_flips(self, actuator):
        body = actuator[actuator.index("int32_t getRawPos"):]
        head = body[:450]
        assert "SENSOR_HALL" in head and "avgPot" in head, "getRawPos must pick pot vs Hall"
        assert "applyFlip" in head, "getRawPos must apply the direction flip"

    def test_flip_uses_sensor_reversed(self, actuator):
        assert re.search(r"sensorReversed\s*\?", actuator), "applyFlip must branch on sensorReversed"

    def test_getpos_falls_back_when_uncalibrated(self, actuator):
        assert re.search(r"if\s*\(\s*!calValid\s*\)", actuator), \
            "getPos must keep the legacy minStop/maxStop path until a JointCal is loaded"

    def test_apply_and_load_wired_up(self, actuator):
        assert re.search(r"void\s+applyJointCal\s*\(", actuator)
        # initAll must actually call loadJointCals(), or EEPROM cal never reaches actuators.
        init = actuator[actuator.index("void initAll"):]
        assert re.search(r"loadJointCals\s*\(\s*\)", init[:200])

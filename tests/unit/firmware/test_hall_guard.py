"""Guard test for the HallA edge-counting implementation (M17 Task 1, AC 1b).

`hall_hw.cpp` implements per-slot HallA edge counting via pin-change interrupts, and
the telemetry line carries the count. The failure modes that would *silently* disable
it without breaking the build are:
  - `hallHwInit()` not called in setup() (interrupts never armed → counts stay 0);
  - the wrong PCINT mask/enable (wrong pins counted, or none);
  - the telemetry field dropped (count never reaches the host).

These assertions lock in the wired-up Rev-3 (Krabby-Uno v0.2/v0.3) path — the default
build. Bench verification on a real Hall sensor is hardware-gated (hip-yaw joints), so
this protects the firmware-side invariants that bench testing can't.
"""
import re
from pathlib import Path

import pytest

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


@pytest.fixture(scope="module")
def ino() -> str:
    return (ARDUINO / "arduino.ino").read_text()


@pytest.fixture(scope="module")
def hall() -> str:
    return (ARDUINO / "hall_hw.cpp").read_text()


@pytest.fixture(scope="module")
def actuator() -> str:
    return (ARDUINO / "actuator_manager.h").read_text()


class TestHallWiredUp:
    def test_init_called_in_setup(self, ino):
        # Without this call the PCINT counters are never armed and Hall reads 0.
        assert re.search(r"\bhallHwInit\s*\(\s*\)", ino), "hallHwInit() must be called in setup()"

    def test_edge_count_in_telemetry(self, actuator):
        assert re.search(r"hallHwGetEdgeCount\s*\(", actuator), \
            "telemetry must print hallHwGetEdgeCount(slot)"


class TestHallRev3Counter:
    def test_counter_storage_is_volatile(self, hall):
        assert re.search(r"volatile\s+uint32_t\s+g_hallEdgeCount\s*\[\s*6\s*\]", hall)

    def test_portb_pcint_armed_for_fl_halls(self, hall):
        # D50/D51/D52 = PB3/PB2/PB1 → mask 0x0E on PCINT0.
        assert re.search(r"PCMSK0\s*\|=\s*0x0E", hall), "FL hall mask (D50-D52) must be 0x0E"
        assert re.search(r"PCICR\s*\|=\s*_BV\(\s*PCIE0\s*\)", hall), "PCINT0 must be enabled"
        assert re.search(r"ISR\s*\(\s*PCINT0_vect\s*\)", hall), "PCINT0_vect ISR must exist"

    def test_portk_pcint_armed_for_fr_halls(self, hall):
        # A12/A13/A14 = PK4/PK5/PK6 → mask 0x70 on PCINT2.
        assert re.search(r"PCMSK2\s*\|=\s*0x70", hall), "FR hall mask (A12-A14) must be 0x70"
        assert re.search(r"PCICR\s*\|=\s*_BV\(\s*PCIE2\s*\)", hall), "PCINT2 must be enabled"
        assert re.search(r"ISR\s*\(\s*PCINT2_vect\s*\)", hall), "PCINT2_vect ISR must exist"

    def test_isrs_increment_the_counters(self, hall):
        # Both vectors must actually bump g_hallEdgeCount; 3 slots each = >=6 increments.
        bumps = re.findall(r"g_hallEdgeCount\s*\[[^\]]+\]\s*\+\+", hall)
        assert len(bumps) >= 6, f"expected >=6 counter increments across the ISRs, found {len(bumps)}"


class TestHallQuadrature:
    """M17 Task 2 §5 (2c): signed quadrature from HallA edge + HallB phase."""

    def test_signed_count_storage_and_accessor(self, hall):
        assert re.search(r"volatile\s+int32_t\s+g_hallSignedCount\s*\[\s*6\s*\]", hall)
        assert re.search(r"int32_t\s+hallHwGetSignedCount\s*\(", hall)

    def test_signed_accessor_declared(self):
        header = (ARDUINO / "hall_hw.h").read_text()
        assert "hallHwGetSignedCount" in header

    def test_rev3_isrs_sample_hallB_via_pinf(self, hall):
        # Quadrature direction needs the B channel; on Rev 3 HallB is A0-A5 on PORTF.
        rev3 = hall[hall.index("KRABBY_PIN_REV == 3"):hall.index("KRABBY_PIN_REV == 1")]
        assert "PINF" in rev3, "Rev 3 ISRs must read HallB from PINF for the quadrature phase"
        assert "quadStep" in rev3, "must use the A-vs-B quadrature step"

    def test_signed_count_updated_in_isrs(self, hall):
        bumps = re.findall(r"g_hallSignedCount\s*\[[^\]]+\]\s*\+=", hall)
        assert len(bumps) >= 6, f"expected >=6 signed-count updates, found {len(bumps)}"

    def test_actuator_uses_signed_count_not_edges(self, actuator):
        body = actuator[actuator.index("int32_t hallSignedCount"):]
        assert "hallHwGetSignedCount" in body[:200], "hallSignedCount() must return the signed quadrature count"

    def test_hall_pins_pulled_up(self, hall):
        # Disconnected hall lines idle high rather than floating into noise.
        assert re.search(r"pinMode\([^,]+,\s*INPUT_PULLUP\s*\)", hall)

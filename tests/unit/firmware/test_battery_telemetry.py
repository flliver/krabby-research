"""BATT parsing and display tests.

Adapter availability, numeric readings, divergence, and frame age remain
separate fields. Divergence includes unavailable and non-finite battery inputs.
"""

import math
from unittest import mock

from firmware.gui.app import (
    SENSOR_STALE_S,
    STATE_COLOR_OK,
    STATE_COLOR_STALE,
    BattRow,
)
from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.interfaces.battery_telemetry import (
    BatteryTelemetry,
    PackVoltageRegion,
)

# Every field distinctive, so a serializer or parser that transposed two of them
# could not produce a passing result by coincidence.
BATT_SEG = "BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0 1 1"


def _seg(divergence=0, region=0, pack_valid=1, midpoint_valid=1):
    """Build a BATT segment with the trailing bytes set explicitly."""
    return (
        "BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 "
        f"{divergence} {region} {pack_valid} {midpoint_valid}"
    )


def _parse(segment=BATT_SEG):
    return BatteryTelemetry.from_segment(segment)


class TestParsesAllTenFields:
    def test_named_unpacking_preserves_order_units_and_signs(self):
        b = _parse()

        assert b.pack_volts == 26.55
        assert b.pack_current_amperes == -12.34  # signed: discharge is negative
        assert b.pack_power_watts == 327.6
        assert b.pack_charge_coulombs == 1450.2
        assert b.battery_a_volts == 13.35
        assert b.battery_b_volts == 13.20
        assert b.divergence is False
        assert b.pack_region is PackVoltageRegion.NORMAL
        assert b.pack_valid is True
        assert b.midpoint_valid is True

    def test_divergence_flag_is_read_as_a_bool(self):
        assert _parse(_seg(divergence=1)).divergence is True


class TestRegionByte:
    def test_every_defined_region_parses(self):
        for region in PackVoltageRegion:
            b = _parse(_seg(region=int(region)))
            assert b.pack_region is region

    def test_an_unknown_byte_is_retained_without_disturbing_the_other(self):
        # A newer firmware stays parseable, and the unknown stays visible as an
        # int rather than being coerced onto a value we do know.
        b = _parse(_seg(region=200))
        assert b.pack_region == 200
        assert not isinstance(b.pack_region, PackVoltageRegion)
        assert b.pack_valid is True and b.midpoint_valid is True


class TestMalformedSegmentsAreDropped:
    def test_a_short_frame_is_rejected(self):
        # The pre-validity wire format, rejected on token count rather than
        # parsed with the trailing bytes silently defaulted.
        assert _parse("BATT 26.55 -12.34 327.6 1450.2 13.35 13.20 0 0") is None

    def test_extra_tokens_are_rejected(self):
        assert _parse(BATT_SEG + " 0") is None

    def test_nonnumeric_measurement_is_rejected(self):
        assert _parse(BATT_SEG.replace("327.6", "warm")) is None

    def test_nonfinite_measurement_is_preserved(self):
        assert str(_parse(BATT_SEG.replace("26.55", "nan")).pack_volts) == "nan"
        assert _parse(BATT_SEG.replace("26.55", "inf")).pack_volts == float("inf")

    def test_invalid_divergence_is_rejected(self):
        assert _parse(_seg(divergence=2)) is None

    def test_nonnumeric_state_byte_is_rejected(self):
        assert _parse(_seg(region="x")) is None

    def test_a_wrong_tag_is_not_claimed(self):
        assert _parse(BATT_SEG.replace("BATT", "IMU")) is None


class TestCompactFormatting:
    def test_shows_both_axes_without_recombining_them(self):
        text = _parse(_seg(region=1)).format_compact()

        assert "WARN" in text

    def test_an_unknown_byte_shows_its_number(self):
        text = _parse(_seg(region=200)).format_compact()

        assert " 200" in text

    def test_divergence_is_called_out(self):
        assert "DIVERGE" in _parse(_seg(divergence=1)).format_compact()


class TestResolveBattState:
    """state is freshness only: how long ago the sample arrived."""

    def test_absent_when_no_sample(self):
        text, col = BattRow.resolve_state(None, None)
        assert text == "—"
        assert col == ""

    def test_fresh_sample_reads_fresh(self):
        text, col = BattRow.resolve_state(_parse(), 0.1)
        assert text == "fresh"
        assert col == STATE_COLOR_OK

    def test_divergence_does_not_affect_freshness(self):
        # The two are independent: a diverged sample that just arrived is fresh.
        diverged = _parse(_seg(divergence=1))

        assert BattRow.resolve_state(diverged, 0.1)[0] == "fresh"
        assert BattRow.resolve_state(diverged, SENSOR_STALE_S + 0.1)[0] == "stale"

    def test_the_stale_boundary_is_exclusive(self):
        assert BattRow.resolve_state(_parse(), SENSOR_STALE_S)[0] == "fresh"
        assert BattRow.resolve_state(_parse(), SENSOR_STALE_S + 0.001)[0] == "stale"


class TestResolveDivergence:
    """diverge is the frame's own field, reported without regard to age."""

    def test_absent_when_no_sample(self):
        assert BattRow.resolve_divergence(None) == ("—", "")

    def test_balanced_sample_reads_ok(self):
        text, col = BattRow.resolve_divergence(_parse())
        assert text == "ok"
        assert col == STATE_COLOR_OK

    def test_diverged_sample_is_called_out(self):
        text, col = BattRow.resolve_divergence(_parse(_seg(divergence=1)))
        assert text == "DIVERGED"
        assert col == STATE_COLOR_STALE


class TestTheTwoDisplayAxesAreIndependent:
    """A diverged sample can be either fresh or stale."""

    def test_a_pack_that_was_diverging_when_it_went_quiet_reports_both(self):
        diverged = _parse(_seg(divergence=1))
        age = SENSOR_STALE_S + 5.0

        assert BattRow.resolve_divergence(diverged)[0] == "DIVERGED"
        assert BattRow.resolve_state(diverged, age)[0] == "stale"

    def test_all_four_combinations_are_distinguishable(self):
        balanced = _parse()
        diverged = _parse(_seg(divergence=1))
        fresh, stale = 0.1, SENSOR_STALE_S + 0.1

        seen = {
            (BattRow.resolve_divergence(b)[0], BattRow.resolve_state(b, age)[0])
            for b in (balanced, diverged)
            for age in (fresh, stale)
        }
        assert seen == {
            ("ok", "fresh"),
            ("ok", "stale"),
            ("DIVERGED", "fresh"),
            ("DIVERGED", "stale"),
        }


class TestMonitorValidity:
    """The wire flags preserve each adapter's availability independently."""

    def test_both_up(self):
        b = _parse()
        assert BattRow.resolve_monitor(b.pack_valid) == ("up", STATE_COLOR_OK)
        assert BattRow.resolve_monitor(b.midpoint_valid) == ("up", STATE_COLOR_OK)

    def test_each_monitor_is_reported_independently(self):
        pack_down = _parse(_seg(pack_valid=0, midpoint_valid=1))
        mid_down = _parse(_seg(pack_valid=1, midpoint_valid=0))
        both_down = _parse(_seg(pack_valid=0, midpoint_valid=0))

        assert pack_down.pack_valid is False and pack_down.midpoint_valid is True
        assert mid_down.pack_valid is True and mid_down.midpoint_valid is False

        assert BattRow.resolve_monitor(pack_down.pack_valid)[0] == "DOWN"
        assert BattRow.resolve_monitor(pack_down.midpoint_valid)[0] == "up"
        assert BattRow.resolve_monitor(mid_down.pack_valid)[0] == "up"
        assert BattRow.resolve_monitor(mid_down.midpoint_valid)[0] == "DOWN"
        assert BattRow.resolve_monitor(both_down.pack_valid)[0] == "DOWN"
        assert BattRow.resolve_monitor(both_down.midpoint_valid)[0] == "DOWN"

    def test_a_dead_midpoint_still_delivers_the_pack_measurements(self):
        # The four Pack fields 3g.1-3g.4 require, no longer suppressed by an
        # unrelated monitor's failure.
        b = _parse(_seg(pack_valid=1, midpoint_valid=0))

        assert b.pack_volts == 26.55
        assert b.pack_current_amperes == -12.34
        assert b.pack_power_watts == 327.6
        assert b.pack_charge_coulombs == 1450.2

    def test_absent_when_no_sample(self):
        assert BattRow.resolve_monitor(None) == ("—", "")


class TestSdkStorage:
    def _bare_sdk(self) -> KrabbyMCUSDK:
        """A real SDK with no serial port opened; __init__ is I/O-free when given
        an explicit port, so this exercises the real object's invariants."""
        return KrabbyMCUSDK(port="unused")

    def test_battery_starts_as_none(self):
        assert self._bare_sdk().battery is None

    def test_a_cached_battery_sample_does_not_survive_a_reconnect(self):
        # A sample held across a disconnect is of unknown age, so sdk.battery
        # would stop being able to mean "nothing read this session". connect()
        # already drops sdk.imu for the same reason; Task 3 did not follow it.
        sdk = self._bare_sdk()
        sdk.battery = _parse()
        sdk.imu = object()

        with mock.patch("firmware.krabby_mcu.serial.Serial"), \
             mock.patch("firmware.krabby_mcu.time.sleep"), \
             mock.patch("firmware.krabby_mcu.threading.Thread"), \
             mock.patch.object(KrabbyMCUSDK, "send_command_joints_hold"):
            assert sdk.connect() is True

        assert sdk.battery is None
        assert sdk.imu is None


def test_battery_segments_share_the_consolidated_telemetry_parser():
    from firmware.interfaces.telemetry_frame import TelemetryFrame

    frame = TelemetryFrame.parse_line(
        "FRONT; FLHY 0.50 0 100 500 101 0 0 0 1;"
        "IMU 0 0 9.807 0 0 0 25 1;" + BATT_SEG + ";BATT malformed"
    )
    assert len(frame.joints) == 1
    assert frame.joints[0].name == "FLHY"
    assert frame.imu.valid is True
    assert frame.battery == _parse()


class TestUnavailableBatteryVoltages:
    def test_missing_pack_keeps_numeric_battery_values(self):
        b = _parse("BATT 0.00 0.00 0.0 0.0 12.75 -12.75 0 0 0 1")
        assert b is not None
        assert not b.pack_valid and b.midpoint_valid
        assert b.battery_a_volts == 12.75 and b.battery_b_volts == -12.75
        assert not b.split_available
        assert "A:12.75V B:-12.75V" in b.format_compact()
        assert BattRow.resolve_divergence(b) == ("—", "")

    def test_missing_midpoint_does_not_discard_pack(self):
        b = _parse("BATT 25.50 -12.50 331.2 1200.0 0.00 25.50 0 0 1 0")
        assert b is not None and b.pack_valid and not b.midpoint_valid
        assert b.pack_volts == 25.5
        assert "A:0.00V B:25.50V" in b.format_compact()
        assert b.format_battery_voltage(b.battery_a_volts) == "0.00"

    def test_numeric_range_does_not_change_availability(self):
        b = _parse("BATT 10.00 -12.50 331.2 1200.0 12.00 -2.00 0 0 1 1")
        assert b.pack_valid and b.midpoint_valid
        assert b.split_available
        assert BattRow.resolve_divergence(b)[0] != "—"

    def test_zero_volts_is_available_and_recovery_restores_split(self):
        b = _parse("BATT 0.00 0.00 0.0 0.0 0.00 0.00 0 0 1 1")
        assert b.split_available
        assert b.format_battery_voltage(0.0) == "0.00"
        assert _parse().split_available

    def test_monitor_validity_is_required_even_with_numeric_battery_fields(self):
        b = _parse(_seg(pack_valid=0))
        assert not b.split_available
        assert BattRow.resolve_divergence(b) == ("—", "")


def test_nonfinite_midpoint_preserves_pack_and_disables_split():
    b = _parse("BATT 26.00 1.00 26.0 100.0 nan nan 0 0 1 0")
    assert b.pack_volts == 26.0 and b.pack_valid
    assert not b.split_available
    assert BattRow.resolve_divergence(b) == ("—", "")


def test_divergence_is_not_hidden_when_a_monitor_is_unavailable():
    for pack, midpoint in ((0, 1), (1, 0), (0, 0)):
        b = _parse(_seg(divergence=1, pack_valid=pack, midpoint_valid=midpoint))
        assert BattRow.resolve_divergence(b) == ("DIVERGED", STATE_COLOR_STALE)
        assert "DIVERGE" in b.format_compact()


def test_nonfinite_voltage_divergence_preserves_working_pack():
    b = _parse("BATT 26.00 1.00 26.0 100.0 nan nan 1 0 1 1")
    assert b.pack_volts == 26.0
    assert BattRow.resolve_divergence(b) == ("DIVERGED", STATE_COLOR_STALE)
    assert "DIVERGE" in b.format_compact()

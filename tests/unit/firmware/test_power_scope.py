from firmware.gui.power_scope import (
    LIFECYCLE_COLORS,
    LIFECYCLE_ORDER,
    REGION_COLORS,
    REGION_ORDER,
    THRESHOLDS,
    PowerTimelineRecorder,
    ordered_positions,
    state_spans,
)
from firmware.interfaces.battery_telemetry import BatteryTelemetry
from firmware.interfaces.battery_telemetry import (
    PackVoltageRegion,
    PowerSafetyState,
)
from firmware.interfaces.power_messages import (
    PowerMessage,
    PowerMessageType,
    PoweringDownReason,
)


def battery(region=0, state=0, volts=26.5):
    return BatteryTelemetry.from_segment(
        f"BATT {volts} 0 0 0 13.2 13.3 0 1 1 {region} {state}"
    )


def test_records_each_battery_frame_once_and_preserves_both_axes():
    recorder = PowerTimelineRecorder()
    first = battery()
    second = battery(region=1, state=1, volts=24.7)

    recorder.observe(first, None, 1.0)
    recorder.observe(first, None, 1.1)
    recorder.observe(second, None, 2.0)

    assert [sample.pack_volts for sample in recorder.samples()] == [26.5, 24.7]
    assert [event.text for event in recorder.events()] == [
        "start NORMAL/OPERATING",
        "region NORMAL → WARN",
        "lifecycle OPERATING → POWERING_DOWN",
    ]


def test_records_power_messages_once():
    recorder = PowerTimelineRecorder()
    message = PowerMessage(
        PowerMessageType.POWERING_DOWN,
        PoweringDownReason.UNDER_VOLTAGE_SOFT,
    )

    recorder.observe(None, message, 1.0)
    recorder.observe(None, message, 1.1)

    assert [event.text for event in recorder.events()] == [
        "PWR 1 POWERING_DOWN under_voltage_soft"
    ]


def test_clear_allows_the_same_objects_to_start_a_new_trace():
    recorder = PowerTimelineRecorder()
    sample = battery()
    recorder.observe(sample, None, 1.0)
    recorder.clear()
    recorder.observe(sample, None, 2.0)

    assert len(recorder.samples()) == 1
    assert recorder.events()[0].text == "start NORMAL/OPERATING"


def test_state_spans_split_only_when_the_value_changes():
    assert state_spans([0, 1, 2, 3], [0, 0, 1, 1], 5) == [
        (0, 2, 0),
        (2, 5, 1),
    ]


def test_every_known_region_and_lifecycle_has_a_color():
    assert set(REGION_COLORS) == set(PackVoltageRegion)
    assert set(LIFECYCLE_COLORS) == set(PowerSafetyState)
    assert all(region in REGION_COLORS for _, _, region in THRESHOLDS)


def test_region_axis_follows_increasing_pack_voltage():
    assert REGION_ORDER == (
        PackVoltageRegion.HARD_CUT,
        PackVoltageRegion.SOFT_CUT,
        PackVoltageRegion.WARN,
        PackVoltageRegion.NORMAL,
        PackVoltageRegion.ABOVE_RECOVERY,
        PackVoltageRegion.OVER_VOLT,
    )
    assert ordered_positions(REGION_ORDER, REGION_ORDER) == list(range(6))


def test_lifecycle_axis_follows_effective_power_condition():
    assert LIFECYCLE_ORDER == (
        PowerSafetyState.OVER_VOLTAGE_LOCKOUT,
        PowerSafetyState.SLEEP,
        PowerSafetyState.POWERING_DOWN,
        PowerSafetyState.RESUMING,
        PowerSafetyState.OPERATING,
    )
    assert ordered_positions(LIFECYCLE_ORDER, LIFECYCLE_ORDER) == list(range(5))

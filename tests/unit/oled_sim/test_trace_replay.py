"""Firmware draw-call trace replay tests."""
from __future__ import annotations

import shutil

import pytest

from krab import KrabState, render, render_sequence, trace

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None, reason="oled_trace needs cmake to build")


def lit(frame) -> set:
    rows = frame.to_rows()
    return {(x, y) for y, row in enumerate(rows)
            for x, cell in enumerate(row) if cell == "#"}


def test_a_frame_comes_back_with_the_header_rule_drawn():
    frame = render(KrabState())

    assert all(frame.get(x, 9) for x in range(128))


def test_the_two_gauges_differ_when_their_charge_differs():
    frame = render(KrabState(battery_volts=(13.4, 12.0)))

    band = range(11, 19)
    left = sum(frame.get(x, y) for x in range(0, 64) for y in band)
    right = sum(frame.get(x, y) for x in range(64, 128) for y in band)
    assert left > right


def test_a_full_gauge_lights_more_of_its_bar_than_an_empty_one():
    full = lit(render(KrabState(battery_volts=(13.4, 13.4))))
    empty = lit(render(KrabState(battery_volts=(12.0, 12.0))))

    assert len(full) > len(empty)


def test_a_second_frame_carries_only_what_changed():
    states = [
        KrabState(battery_volts=(13.4, 13.4)),
        KrabState(battery_volts=(13.4, 12.3)),
    ]
    first, second = trace(states)

    assert any(call == "erase" for call in first)
    assert not any(call == "erase" for call in second)
    assert len(second) < len(first) / 4


def test_an_incremental_frame_is_replayed_onto_the_one_before_it():
    states = [
        KrabState(battery_volts=(13.4, 13.4)),
        KrabState(battery_volts=(13.4, 12.3)),
    ]
    frames = render_sequence(states)

    assert lit(frames[0]) != lit(frames[1])
    body = {(x, y) for x, y in lit(frames[0]) if y >= 22}
    assert body and body <= lit(frames[1])


def test_a_rendered_state_is_reproducible():
    assert lit(render(KrabState(battery_volts=(12.7, 12.7)))) == \
        lit(render(KrabState(battery_volts=(12.7, 12.7))))


def test_battery_voltage_labels_and_pack_sum_come_from_native_model():
    calls, = trace([KrabState(battery_volts=(13.3, 12.7))])
    for voltage in ("13.3V", "12.7V", "26.0V"):
        assert any(call.endswith(voltage) for call in calls)
    assert not any("%" in call for call in calls)


def test_voltage_only_changes_and_missing_readings_replay_cleanly():
    states = [
        KrabState(battery_volts=(14.0, 14.0)),
        KrabState(battery_volts=(14.0, 14.1)),
        KrabState(battery_volts=(-1.0, 14.1)),
    ]
    frames = render_sequence(states)
    for state, frame in zip(states, frames):
        assert lit(frame) == lit(render(state))
    assert lit(frames[0]) != lit(frames[1])
    calls, = trace([states[-1]])
    assert sum(call.endswith("--.-V") for call in calls) == 2


def test_missing_left_board_overrides_its_actuator_inputs():
    hold = ("hold", "hold", "hold")
    extend = ("extend", "extend", "extend")
    baseline = KrabState(left=False, legs=[hold] * 6)
    impossible_input = KrabState(
        left=False,
        legs=[hold, hold, extend, hold, extend, hold],
    )

    assert lit(render(baseline)) == lit(render(impossible_input))
    assert lit(render(KrabState(left=True, legs=impossible_input.legs))) != \
        lit(render(impossible_input))


def test_missing_front_board_overrides_its_actuator_inputs():
    hold = ("hold", "hold", "hold")
    extend = ("extend", "extend", "extend")
    baseline = KrabState(front=False, legs=[hold] * 6)
    impossible_input = KrabState(
        front=False,
        legs=[extend, extend, hold, hold, hold, hold],
    )

    assert lit(render(baseline)) == lit(render(impossible_input))
    assert lit(render(KrabState(front=True, legs=impossible_input.legs))) != \
        lit(render(impossible_input))


def test_an_invalid_scene_is_rejected_before_it_reaches_the_binary():
    with pytest.raises(ValueError):
        KrabState(legs=[("hold", "hold", "sideways")] * 6).to_fields()
    with pytest.raises(ValueError):
        KrabState(role="MIDDLE").to_fields()
    with pytest.raises(ValueError):
        KrabState(legs=[("hold", "hold", "hold")] * 5).to_fields()
    with pytest.raises(ValueError):
        KrabState(battery_volts=(12.7,)).to_fields()

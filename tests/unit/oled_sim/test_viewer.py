from krab import KrabState
from viewer import build


def test_viewer_has_one_live_screen_and_state_controls():
    page = build()

    assert page.count("<canvas") == 1
    assert 'id="screen"' in page
    assert 'fetch("/render"' in page
    for control in (
        "role", "front", "left", "right", "imu_valid",
        "roll", "pitch", "battery_a", "battery_b",
    ):
        assert f'id="{control}"' in page


def test_browser_payload_maps_to_hardware_state():
    legs = [["hold", "extend", "retract"] for _ in range(6)]
    state = KrabState.from_payload({
        "role": "RIGHT",
        "roll": 8,
        "pitch": -3,
        "imu_valid": False,
        "battery_volts": [13.2, 12.7],
        "front": True,
        "left": False,
        "right": True,
        "legs": legs,
    })

    assert state.role == "RIGHT"
    assert (state.roll, state.pitch) == (8, -3)
    assert state.imu_valid is False
    assert state.battery_volts == (13.2, 12.7)
    assert (state.front, state.left, state.right) == (True, False, True)
    assert state.legs == legs

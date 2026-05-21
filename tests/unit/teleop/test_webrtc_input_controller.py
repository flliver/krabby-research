from __future__ import annotations

from teleop.edge.webrtc_input_controller import WebRTCInputController


def test_webrtc_input_controller_applies_payload_and_notifies_callback() -> None:
    ctl = WebRTCInputController()
    seen = []
    ctl.register_callback(lambda s: seen.append(s))
    ctl.start(update_rate_hz=50.0)
    out = ctl.update_from_payload(
        {
            "LT": True,
            "LB": False,
            "LS": True,
            "RS": False,
            "RT": True,
            "RB": True,
            "LX": 0.25,
            "LY": -0.5,
            "RX": 0.75,
            "RY": -1.2,
        }
    )
    assert out.LT is True
    assert out.LS is True
    assert out.RY == -1.0
    assert len(seen) == 1


def test_webrtc_input_controller_requires_start_before_update() -> None:
    ctl = WebRTCInputController()
    try:
        ctl.update_from_payload({})
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass

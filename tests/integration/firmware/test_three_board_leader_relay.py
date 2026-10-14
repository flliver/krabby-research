"""Leader forwards commands and aggregates FRONT/LEFT/RIGHT telemetry for the SDK."""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.three_board_mcu_host import ThreeBoardMcuHost


@pytest.fixture
def bridge():
    try:
        host = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return host


def _fast_connect_sleep(monkeypatch) -> None:
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_receives_three_role_telemetry_lines(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if {"FRONT", "LEFT", "RIGHT"}.issubset(mcu.role_last_seen):
                break
            time.sleep(0.02)

        assert "FRONT" in mcu.role_last_seen
        assert "LEFT" in mcu.role_last_seen
        assert "RIGHT" in mcu.role_last_seen
    finally:
        mcu.running = False
        host.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_follower_jog_updates_left_joint_only(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if mcu.joints.get("RLHY") is not None and mcu.joints.get("FLHY") is not None:
                break
            time.sleep(0.02)

        mcu.send_command_jog("RLHY", 200)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            left = mcu.joints.get("RLHY")
            front = mcu.joints.get("FLHY")
            if (
                left is not None
                and (left.pwm[0] >= 100 or left.pwm[1] >= 100)
                and front is not None
                and front.pwm[0] < 50
                and front.pwm[1] < 50
            ):
                break
            time.sleep(0.02)

        left = mcu.joints["RLHY"]
        front = mcu.joints["FLHY"]
        assert left.pwm[0] >= 100 or left.pwm[1] >= 100
        assert front.pwm[0] < 50 and front.pwm[1] < 50
    finally:
        mcu.running = False
        host.close()

"""Single-leader 18-joint mode: SDK sees three roles without role election."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.single_leader_mcu_host import SingleLeaderEighteenMcuHost


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
def test_sdk_receives_three_roles_in_single_leader_mode(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = SingleLeaderEighteenMcuHost(bridge)
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

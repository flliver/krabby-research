"""KrabbyMCUSDK over PTY against the host firmware loop (no Isaac)."""

from __future__ import annotations

import os
import threading
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.serial_pty_bridge import SimMcuPtyBridge


@pytest.fixture
def bridge():
    try:
        host = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    host.reset()
    return host


def _fast_connect_sleep(monkeypatch) -> None:
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


def _firmware_tick_loop(bridge, stop: threading.Event) -> None:
    bridge.firmware_setup()
    while not stop.is_set():
        bridge.advance_millis(10)
        bridge.firmware_loop_once()
        time.sleep(0.005)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_connect_jog_and_joint_telemetry(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    stop = threading.Event()
    tick = threading.Thread(
        target=_firmware_tick_loop,
        args=(bridge, stop),
        name="host-firmware-tick",
        daemon=True,
    )

    with SimMcuPtyBridge(bridge) as pty:
        tick.start()
        mcu = KrabbyMCUSDK(port=pty.slave_path)
        try:
            assert mcu.connect() is True

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if mcu.joints.get("FLHY") is not None:
                    break
                time.sleep(0.02)
            assert mcu.joints.get("FLHY") is not None

            mcu.send_command_jog("FLHY", 200)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                jt = mcu.joints.get("FLHY")
                if jt is not None and jt.pwm[1] >= 200:
                    break
                time.sleep(0.02)

            jt = mcu.joints["FLHY"]
            assert jt is not None
            assert jt.pwm[1] >= 200
        finally:
            mcu.running = False
            mcu.close()
            stop.set()
            tick.join(timeout=3.0)

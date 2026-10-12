"""KrabbyMCUSDK ``current`` field over Isaac PTY (same path as firmware GUI Cur column)."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.current_sense_mapping import (
    DEFAULT_GROUND_CONTACT_ADC,
    DEFAULT_MOVING_NO_CONTACT_ADC,
)
from firmware.sim.isaac_mcu_host import IsaacMcuHost
from tests.integration.firmware.conftest import park_at_default_pose, skip_unless_firmware_isaac

pytestmark = pytest.mark.isaacsim

JOINT = "FLHL"
JOG_PWM = 200
TELEMETRY_WAIT_S = 4.0


def _fast_connect_sleep(monkeypatch) -> None:
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


def _wait_joint_current(mcu: KrabbyMCUSDK, name: str, predicate, timeout: float = TELEMETRY_WAIT_S) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jt = mcu.joints.get(name)
        if jt is not None and predicate(jt.current):
            return int(jt.current)
        time.sleep(0.02)
    jt = mcu.joints.get(name)
    got = jt.current if jt is not None else None
    raise AssertionError(f"timeout waiting for {name} current predicate; last={got!r}")


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_cur_idle_and_moving_bands(front_board_rig, monkeypatch):
    skip_unless_firmware_isaac()
    _fast_connect_sleep(monkeypatch)
    core, robot, _config, _ = front_board_rig
    park_at_default_pose(core, robot)

    with IsaacMcuHost(core) as port:
        mcu = KrabbyMCUSDK(port=port)
        try:
            assert mcu.connect() is True
            idle = _wait_joint_current(mcu, JOINT, lambda c: c <= 1)
            assert idle == 0

            core.jog(JOINT, JOG_PWM)
            moving = _wait_joint_current(
                mcu,
                JOINT,
                lambda c: c >= DEFAULT_MOVING_NO_CONTACT_ADC - 1,
            )
            assert moving >= DEFAULT_MOVING_NO_CONTACT_ADC - 1
        finally:
            mcu.running = False


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_cur_ground_contact_band(gravity_front_board, monkeypatch):
    skip_unless_firmware_isaac()
    _fast_connect_sleep(monkeypatch)
    core, _robot, _config = gravity_front_board
    core.reset()

    with IsaacMcuHost(core) as port:
        mcu = KrabbyMCUSDK(port=port)
        try:
            assert mcu.connect() is True

            core.jog("FLKL", JOG_PWM)
            loaded = _wait_joint_current(
                mcu,
                "FLKL",
                lambda c: c >= DEFAULT_GROUND_CONTACT_ADC - 1,
                timeout=8.0,
            )
            assert loaded >= DEFAULT_GROUND_CONTACT_ADC - 1
        finally:
            mcu.running = False

"""KrabbyMCUSDK ``current`` field over Isaac PTY (same path as firmware GUI Cur column)."""

from __future__ import annotations

import os
import time

import pytest

from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.current_sense_mapping import (
    DEFAULT_GROUND_CONTACT_ADC,
    DEFAULT_LIFT_LOAD_ADC,
    DEFAULT_MOVING_NO_CONTACT_ADC,
)
from firmware.sim.isaac_mcu_host import IsaacMcuHost
from firmware.sim.isaac_scene import advance_sim, physics_step_ms
from tests.integration.firmware.conftest import park_at_default_pose, skip_unless_firmware_isaac

pytestmark = pytest.mark.isaacsim

JOINT = "FLHL"
JOG_PWM = 200
TELEMETRY_WAIT_S = 4.0
LIFT_SETTLE_STEPS = 120
LIFT_JOG_STEPS = 200
LIFT_JOG_PWM = 255
LIFT_ADC_MIN = DEFAULT_LIFT_LOAD_ADC - 5
FRONT_LIFT_ACTUATORS = ("FLHL", "FLKL", "FRHL", "FRKL")


def _gui_cur_text(jt: JointTelemetry | None) -> str:
    """Same value as ``JointRow.update_from_telemetry`` → ``var_cur`` (no Tk import)."""
    if jt is None:
        return "---"
    return str(jt.current)


def _fast_connect_sleep(monkeypatch) -> None:
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


def _wait_joint_current(
    mcu: KrabbyMCUSDK,
    name: str,
    predicate,
    timeout: float = TELEMETRY_WAIT_S,
    *,
    core=None,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jt = mcu.joints.get(name)
        if jt is not None and predicate(jt.current):
            return int(jt.current)
        if core is not None:
            core.step()
        time.sleep(0.02)
    jt = mcu.joints.get(name)
    got = jt.current if jt is not None else None
    raise AssertionError(f"timeout waiting for {name} current predicate; last={got!r}")


def _peak_current_after_lift(
    core,
    mcu: KrabbyMCUSDK,
    *,
    gui_cur: bool,
    host_driven: bool = False,
) -> int:
    """Same lift physics as ``test_pin_is_hits_lift_band``; read Cur via SDK/GUI telemetry path."""
    step_s = physics_step_ms(core.env) / 1000.0
    if host_driven:
        # IsaacMcuHost already calls core.step(); avoid advance_sim + double-step stalls.
        time.sleep(LIFT_SETTLE_STEPS * step_s)
    else:
        for _ in range(LIFT_SETTLE_STEPS):
            advance_sim(core.env)

    for actuator_id in FRONT_LIFT_ACTUATORS:
        core.jog(actuator_id, LIFT_JOG_PWM)

    peak = 0
    for _ in range(LIFT_JOG_STEPS):
        if host_driven:
            time.sleep(step_s)
        else:
            core.step()
        for actuator_id in FRONT_LIFT_ACTUATORS:
            jt = mcu.joints.get(actuator_id)
            if jt is None:
                continue
            value = int(_gui_cur_text(jt)) if gui_cur else int(jt.current)
            peak = max(peak, value)
    return peak


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_cur_idle_and_moving_bands(front_board_rig, monkeypatch):
    pytest.importorskip("serial")
    skip_unless_firmware_isaac()
    _fast_connect_sleep(monkeypatch)
    core, robot, _config, _ = front_board_rig
    park_at_default_pose(core, robot)

    with IsaacMcuHost(core) as port:
        mcu = KrabbyMCUSDK(port=port)
        try:
            assert mcu.connect() is True
            idle = _wait_joint_current(mcu, JOINT, lambda c: c <= 1, core=core)
            assert idle == 0

            core.jog(JOINT, JOG_PWM)
            moving = _wait_joint_current(
                mcu,
                JOINT,
                lambda c: c >= DEFAULT_MOVING_NO_CONTACT_ADC - 1,
                core=core,
            )
            assert moving >= DEFAULT_MOVING_NO_CONTACT_ADC - 1
        finally:
            mcu.running = False


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_cur_ground_contact_band(gravity_front_board, monkeypatch):
    pytest.importorskip("serial")
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
                core=core,
            )
            assert loaded >= DEFAULT_GROUND_CONTACT_ADC - 1
        finally:
            mcu.running = False


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_sdk_cur_lift_load_band(gravity_front_board, monkeypatch):
    """SDK ``current`` reaches ~30 under front lift load (OVERVIEW 3i / 2g band)."""
    pytest.importorskip("serial")
    skip_unless_firmware_isaac()
    _fast_connect_sleep(monkeypatch)
    core, _robot, _config = gravity_front_board
    core.reset()

    with IsaacMcuHost(core) as port:
        mcu = KrabbyMCUSDK(port=port)
        try:
            assert mcu.connect() is True
            _wait_joint_current(mcu, "FLHL", lambda _c: True, timeout=8.0, core=core)
            peak = _peak_current_after_lift(core, mcu, gui_cur=False, host_driven=True)
            assert peak >= LIFT_ADC_MIN, (
                f"expected SDK current near {DEFAULT_LIFT_LOAD_ADC}, peak {peak}"
            )
        finally:
            mcu.running = False


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_gui_cur_column_shows_lift_load_band(gravity_front_board, monkeypatch):
    """``JointRow`` Cur column (``python -m firmware.gui``) shows ~30 under lift load."""
    pytest.importorskip("serial")
    skip_unless_firmware_isaac()
    _fast_connect_sleep(monkeypatch)
    core, _robot, _config = gravity_front_board
    core.reset()

    with IsaacMcuHost(core) as port:
        mcu = KrabbyMCUSDK(port=port)
        try:
            assert mcu.connect() is True
            _wait_joint_current(mcu, "FLHL", lambda _c: True, timeout=8.0, core=core)
            peak = _peak_current_after_lift(core, mcu, gui_cur=True, host_driven=True)
            assert peak >= LIFT_ADC_MIN, (
                f"expected GUI Cur near {DEFAULT_LIFT_LOAD_ADC}, peak {peak}"
            )
        finally:
            mcu.running = False

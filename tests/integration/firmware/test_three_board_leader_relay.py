"""Leader forwards commands and aggregates FRONT/LEFT/RIGHT telemetry for the SDK."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.three_board_mcu_host import ThreeBoardMcuHost

FOLLOWER_JOG_CASES = (
    ("RLHY", "FLHY", 100),
    ("RRHY", "FRHY", 100),
)


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


def _wait_for_joints(mcu: KrabbyMCUSDK, names: tuple[str, ...], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(mcu.joints.get(name) is not None for name in names):
            return
        time.sleep(0.02)
    missing = [name for name in names if mcu.joints.get(name) is None]
    pytest.fail(f"timed out waiting for joints: {missing}")


def _joint_pwm_active(joint) -> bool:
    return joint.pwm[0] >= 100 or joint.pwm[1] >= 100


def _joint_pwm_idle(joint) -> bool:
    return joint.pwm[0] < 50 and joint.pwm[1] < 50


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

        assert {"FRONT", "LEFT", "RIGHT"}.issubset(mcu.role_last_seen)
    finally:
        mcu.running = False
        host.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
@pytest.mark.parametrize("follower_joint,front_joint,min_pwm", FOLLOWER_JOG_CASES)
def test_follower_jog_updates_target_joint_only(
    bridge, monkeypatch, follower_joint, front_joint, min_pwm
):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True
        _wait_for_joints(mcu, (follower_joint, front_joint))

        mcu.send_command_jog(follower_joint, 200)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            follower = mcu.joints.get(follower_joint)
            front = mcu.joints.get(front_joint)
            if (
                follower is not None
                and _joint_pwm_active(follower)
                and front is not None
                and _joint_pwm_idle(front)
            ):
                break
            time.sleep(0.02)

        follower = mcu.joints[follower_joint]
        front = mcu.joints[front_joint]
        assert _joint_pwm_active(follower)
        assert _joint_pwm_idle(front)
    finally:
        mcu.running = False
        host.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_batch_jog_drives_left_and_right_follower_joints(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True
        _wait_for_joints(mcu, ("RLHY", "RRHY", "FLHY"))

        mcu.send_commands_jog({"RLHY": 200, "RRHY": 200})

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            left = mcu.joints.get("RLHY")
            right = mcu.joints.get("RRHY")
            front = mcu.joints.get("FLHY")
            if (
                left is not None
                and right is not None
                and _joint_pwm_active(left)
                and _joint_pwm_active(right)
                and front is not None
                and _joint_pwm_idle(front)
            ):
                break
            time.sleep(0.02)

        assert _joint_pwm_active(mcu.joints["RLHY"])
        assert _joint_pwm_active(mcu.joints["RRHY"])
        assert _joint_pwm_idle(mcu.joints["FLHY"])
    finally:
        mcu.running = False
        host.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_trajectory_command_drives_left_and_right_follower_joints(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True
        _wait_for_joints(mcu, ("RLHY", "RRHY"))

        mcu.send_command_joints({"RLHY": 0.85, "RRHY": 0.85})

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            left = mcu.joints.get("RLHY")
            right = mcu.joints.get("RRHY")
            if (
                left is not None
                and right is not None
                and (left.pwm[0] > 0 or left.pwm[1] > 0)
                and (right.pwm[0] > 0 or right.pwm[1] > 0)
            ):
                break
            time.sleep(0.02)

        left = mcu.joints["RLHY"]
        right = mcu.joints["RRHY"]
        assert left.pwm[0] > 0 or left.pwm[1] > 0
        assert right.pwm[0] > 0 or right.pwm[1] > 0
    finally:
        mcu.running = False
        host.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_hold_coasts_jogged_left_and_right_follower_joints(bridge, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True
        _wait_for_joints(mcu, ("RLHY", "RRHY"))

        mcu.send_commands_jog({"RLHY": 200, "RRHY": 200})

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            left = mcu.joints.get("RLHY")
            right = mcu.joints.get("RRHY")
            if left is not None and right is not None and _joint_pwm_active(left) and _joint_pwm_active(
                right
            ):
                break
            time.sleep(0.02)

        mcu.send_command_joints_hold()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            left = mcu.joints.get("RLHY")
            right = mcu.joints.get("RRHY")
            if left is not None and right is not None and _joint_pwm_idle(left) and _joint_pwm_idle(
                right
            ):
                break
            time.sleep(0.02)

        assert _joint_pwm_idle(mcu.joints["RLHY"])
        assert _joint_pwm_idle(mcu.joints["RRHY"])
    finally:
        mcu.running = False
        host.close()

"""Parametrized ``KrabbyMCUSDK`` jog over all 18 HAL ``mcu_joints`` (M21 3h)."""

from __future__ import annotations

import os
import time

import pytest

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import KRABBY_HEX_FIRMWARE_ACTUATOR_IDS
from firmware.sim.serial_pty_bridge import SimMcuPtyBridge
from firmware.sim.three_board_mcu_host import ThreeBoardMcuHost
from tests.integration.firmware.conftest import park_at_default_pose
from tests.integration.firmware.test_mcu_sdk_pty_jog import _fast_connect_sleep

# Keep in sync with ``KRABBY_HEX_DEFINITION.mcu_joints`` (hal/server/robot_definition_krabby_hex.py).
HAL_MCU_JOINTS: tuple[str, ...] = (
    "FL_hip_yaw",
    "FL_hip_pitch",
    "FL_knee",
    "FR_hip_yaw",
    "FR_hip_pitch",
    "FR_knee",
    "ML_hip_yaw",
    "ML_hip_pitch",
    "ML_knee",
    "MR_hip_yaw",
    "MR_hip_pitch",
    "MR_knee",
    "RL_hip_yaw",
    "RL_hip_pitch",
    "RL_knee",
    "RR_hip_yaw",
    "RR_hip_pitch",
    "RR_knee",
)

_HAL_TO_FW_SUFFIX = {"hip_yaw": "HY", "hip_pitch": "HL", "knee": "KL"}


def hal_joint_to_firmware_id(hal_name: str) -> str:
    leg, _, suffix = hal_name.partition("_")
    return leg + _HAL_TO_FW_SUFFIX[suffix]


assert len(HAL_MCU_JOINTS) == 18
assert {hal_joint_to_firmware_id(name) for name in HAL_MCU_JOINTS} == set(
    KRABBY_HEX_FIRMWARE_ACTUATOR_IDS
)

JOG_PWM = 255
JOG_STEPS = 40
MIN_COMMANDED_TRAVEL = 0.04
OTHER_JOINT_RATIO = 4


@pytest.fixture
def bridge():
    try:
        host = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return host


def _wait_for_joint(mcu: KrabbyMCUSDK, firmware_id: str, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mcu.joints.get(firmware_id) is not None:
            return
        time.sleep(0.02)
    pytest.fail(f"timed out waiting for telemetry joint {firmware_id!r}")


def _wait_for_joint_with_firmware_ticks(
    core, mcu: KrabbyMCUSDK, firmware_id: str, timeout: float = 8.0
) -> None:
    """Isaac rig: USB telemetry is emitted only when ``core`` ticks host firmware."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mcu.joints.get(firmware_id) is not None:
            return
        core.tick_firmware_without_physics()
        time.sleep(0.002)
    pytest.fail(f"timed out waiting for telemetry joint {firmware_id!r}")


def _joint_pwm_active(joint) -> bool:
    return joint.pwm[0] >= 100 or joint.pwm[1] >= 100


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
@pytest.mark.parametrize("hal_joint_name", HAL_MCU_JOINTS)
def test_krabby_mcu_sdk_jog_activates_each_hal_joint_on_three_board_host(
    bridge, monkeypatch, hal_joint_name: str
):
    pytest.importorskip("serial")
    firmware_id = hal_joint_to_firmware_id(hal_joint_name)
    _fast_connect_sleep(monkeypatch)
    host = ThreeBoardMcuHost(bridge)
    host.start()
    mcu = KrabbyMCUSDK(port=host.slave_path)
    try:
        assert mcu.connect() is True
        _wait_for_joint(mcu, firmware_id)

        mcu.send_command_jog(firmware_id, JOG_PWM)

        deadline = time.monotonic() + 2.5
        while time.monotonic() < deadline:
            joint = mcu.joints.get(firmware_id)
            if joint is not None and _joint_pwm_active(joint):
                break
            time.sleep(0.02)

        joint = mcu.joints.get(firmware_id)
        assert joint is not None, f"no telemetry for {hal_joint_name} ({firmware_id})"
        assert _joint_pwm_active(joint), (
            f"jog did not energize {hal_joint_name} ({firmware_id}): pwm={joint.pwm}"
        )
    finally:
        mcu.running = False
        host.close()


def _joint_deltas_after_sdk_jog(rig, hal_joint_name: str, pwm: int, steps: int) -> dict[str, float]:
    pytest.importorskip("serial")
    core, robot, configs, indices = rig
    firmware_id = hal_joint_to_firmware_id(hal_joint_name)

    park_at_default_pose(core, robot)
    before = {name: robot.data.joint_pos[0, idx].item() for name, idx in indices.items()}

    # PTY relay only — ``ThreeBoardMcuHost``'s background firmware loop would double-tick
    # against ``core.step()`` and can couple motion across same-leg DOFs.
    with SimMcuPtyBridge(core.bridge) as pty:
        mcu = KrabbyMCUSDK(port=pty.slave_path)
        try:
            assert mcu.connect() is True
            _wait_for_joint_with_firmware_ticks(core, mcu, firmware_id)
            mcu.send_command_jog(firmware_id, pwm)
            for _ in range(steps):
                core.step()
        finally:
            mcu.running = False

    return {
        name: robot.data.joint_pos[0, idx].item() - before[name]
        for name, idx in indices.items()
    }


@pytest.mark.isaacsim
@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
@pytest.mark.parametrize("hal_joint_name", HAL_MCU_JOINTS)
def test_krabby_mcu_sdk_jog_each_hal_joint_moves_only_its_isaac_dof(
    eighteen_joint_rig, monkeypatch, hal_joint_name: str
):
    _fast_connect_sleep(monkeypatch)
    firmware_id = hal_joint_to_firmware_id(hal_joint_name)
    commanded_joint = eighteen_joint_rig[2].slot_by_firmware_id(firmware_id).isaac_joint_name

    moved = _joint_deltas_after_sdk_jog(
        eighteen_joint_rig,
        hal_joint_name,
        JOG_PWM,
        JOG_STEPS,
    )
    cmd_travel = moved[commanded_joint]
    assert abs(cmd_travel) > MIN_COMMANDED_TRAVEL, (
        f"{hal_joint_name} ({firmware_id}) did not move {commanded_joint} enough: {moved}"
    )

    for name, delta in moved.items():
        if name == commanded_joint:
            continue
        assert abs(delta) * OTHER_JOINT_RATIO < abs(cmd_travel), (
            f"SDK jog on {hal_joint_name} also moved {name}: {moved}"
        )

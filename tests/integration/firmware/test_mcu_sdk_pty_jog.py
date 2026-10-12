"""KrabbyMCUSDK over PTY: connect, jog, Isaac joint motion, parsed telemetry."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("serial")

from firmware.krabby_mcu import KrabbyMCUSDK
from firmware.sim.isaac_scene import configure_open_loop_velocity_drive
from firmware.sim.robot_core import RobotCore
from firmware.sim.simulated_mcu_host import SimulatedMcuHost
from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

JOINT = "FLHY"
MAX_JOINT_SPEED = 0.5
DRIVE_WAIT_S = 2.0


def _fast_connect_sleep(monkeypatch) -> None:
    real_sleep = time.sleep

    def sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", sleep)


@pytest.fixture(scope="module")
def sdk_rig(front_board_rig):
    core, robot, config, indices = front_board_rig
    configure_open_loop_velocity_drive(robot, list(indices.values()))
    rig = RobotCore.front_board(
        env=core.env,
        articulation=robot,
        joint_index_for_name=indices,
        bridge=core.bridge,
        config=config,
        max_joint_speed=MAX_JOINT_SPEED,
    )
    return rig, robot, config, indices


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_krabby_mcu_sdk_jog_moves_isaac_and_updates_joints(sdk_rig, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    core, robot, config, indices = sdk_rig
    slot = config.slot_by_firmware_id(JOINT)
    joint_index = indices[slot.isaac_joint_name]

    park_at_default_pose(core, robot)
    before = robot.data.joint_pos[0, joint_index].item()

    with SimulatedMcuHost(core) as pty_path:
        mcu = KrabbyMCUSDK(port=pty_path)
        try:
            assert mcu.connect() is True

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if mcu.joints.get(JOINT) is not None:
                    break
                time.sleep(0.02)
            assert mcu.joints.get(JOINT) is not None

            mcu.send_command_jog(JOINT, 200)
            time.sleep(DRIVE_WAIT_S)

            after = robot.data.joint_pos[0, joint_index].item()
            jt = mcu.joints.get(JOINT)
            assert jt is not None
            assert after - before > 0.03
            assert jt.pwm[1] >= 100
        finally:
            mcu.running = False
            mcu.close()

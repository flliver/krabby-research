"""Cam-shaft -> Body_Hip slaving for firmware Isaac (plant USD only)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from firmware.sim.cam_hip_slave import (
    CamHipSlaveBinding,
    apply_cam_hip_slave,
    build_cam_hip_slave_binding,
)
from firmware.sim.isaac_scene import _crab_hex_articulation_uses_ref_joint_names


def _mdp_dir() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "parkour"
        / "parkour_tasks"
        / "parkour_tasks"
        / "crab_hex_forward_task"
        / "mdp"
    )


def _yaw_k() -> float:
    mdp = _mdp_dir()
    if str(mdp) not in sys.path:
        sys.path.insert(0, str(mdp))
    from crab_hex_dimensions import YAW_K

    return float(YAW_K)


def test_build_binding_on_plant_usd():
    class Robot:
        SHAFT_ID = 10
        HIP_ID = 3

        def find_joints(self, names, preserve_order=True):
            del preserve_order
            if names == ["FL_Body_CamShaft_RevoluteJoint"]:
                return ([self.SHAFT_ID], ["FL_Body_CamShaft_RevoluteJoint"])
            if names == [".*_Body_CamShaft_RevoluteJoint"]:
                return ([self.SHAFT_ID], ["FL_Body_CamShaft_RevoluteJoint"])
            if names == ["FL_Body_Hip_RevoluteJoint"]:
                return ([self.HIP_ID], ["FL_Body_Hip_RevoluteJoint"])
            raise ValueError(names)

    binding = build_cam_hip_slave_binding(Robot())
    assert binding == CamHipSlaveBinding(shaft_joint_ids=[10], hip_joint_ids=[3])


def test_build_binding_skips_ref_usd_profile():
    robot = MagicMock()

    def find_joints(names, preserve_order=True):
        del preserve_order
        if names == ["FL_Body_CamShaft_RevoluteJoint"]:
            raise ValueError("ref USD has no cam shaft joint")
        if names == ["lf_hip_yaw"]:
            return ([0], ["lf_hip_yaw"])
        return ([0], ["lf_hip_yaw"])

    robot.find_joints.side_effect = find_joints
    assert _crab_hex_articulation_uses_ref_joint_names(robot) is True
    assert build_cam_hip_slave_binding(robot) is None


@pytest.fixture
def plant_articulation():
    torch = pytest.importorskip("torch")

    class PlantCamHipArticulation:
        SHAFT_ID = 10
        HIP_ID = 3

        def __init__(self) -> None:
            self.joint_pos = torch.zeros(1, 24)
            self.joint_vel = torch.zeros(1, 24)
            self.data = SimpleNamespace(
                joint_pos=self.joint_pos,
                joint_vel=self.joint_vel,
            )
            self.position_target: torch.Tensor | None = None
            self.velocity_target: torch.Tensor | None = None

        def find_joints(self, names, preserve_order=True):
            del preserve_order
            if names == ["FL_Body_CamShaft_RevoluteJoint"]:
                return ([self.SHAFT_ID], ["FL_Body_CamShaft_RevoluteJoint"])
            if names == [".*_Body_CamShaft_RevoluteJoint"]:
                return ([self.SHAFT_ID], ["FL_Body_CamShaft_RevoluteJoint"])
            if names == ["FL_Body_Hip_RevoluteJoint"]:
                return ([self.HIP_ID], ["FL_Body_Hip_RevoluteJoint"])
            raise ValueError(names)

        def set_joint_position_target(self, positions, joint_ids=None, env_ids=None):
            del env_ids
            assert joint_ids == [self.HIP_ID]
            self.position_target = positions.detach().clone()

        def set_joint_velocity_target(self, velocities, joint_ids=None, env_ids=None):
            del env_ids
            assert joint_ids == [self.HIP_ID]
            self.velocity_target = velocities.detach().clone()

        def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None):
            del joint_ids, env_ids
            self.joint_pos.copy_(position)
            self.joint_vel.copy_(velocity)

    return PlantCamHipArticulation(), torch


def test_apply_slaves_hip_to_whitworth_mapping(plant_articulation):
    robot, torch = plant_articulation
    mdp = _mdp_dir()
    if str(mdp) not in sys.path:
        sys.path.insert(0, str(mdp))
    from crab_hex_cam_mapping import cam_shaft_to_hip

    shaft_angle = 0.42
    shaft_speed = 1.5
    robot.joint_pos[0, robot.SHAFT_ID] = shaft_angle
    robot.joint_vel[0, robot.SHAFT_ID] = shaft_speed
    binding = CamHipSlaveBinding(
        shaft_joint_ids=[robot.SHAFT_ID],
        hip_joint_ids=[robot.HIP_ID],
    )

    apply_cam_hip_slave(binding, robot)

    want_hip, want_omega = cam_shaft_to_hip(
        torch.tensor([shaft_angle]), torch.tensor([shaft_speed])
    )
    assert robot.position_target is not None
    assert robot.velocity_target is not None
    assert robot.position_target.item() == pytest.approx(want_hip.item(), abs=1e-6)
    assert robot.velocity_target.item() == pytest.approx(want_omega.item(), abs=1e-6)


def test_cam_shaft_jog_updates_hip_position_target(plant_articulation):
    """FLHY drives cam shaft; hip PD targets follow ``cam_shaft_to_hip`` after apply."""
    robot, torch = plant_articulation
    mdp = _mdp_dir()
    if str(mdp) not in sys.path:
        sys.path.insert(0, str(mdp))
    from crab_hex_cam_mapping import cam_shaft_to_hip

    binding = CamHipSlaveBinding(
        shaft_joint_ids=[robot.SHAFT_ID],
        hip_joint_ids=[robot.HIP_ID],
    )
    k = _yaw_k()
    for shaft_angle in (0.0, 0.2, -0.15, math.pi / 4):
        robot.joint_pos[0, robot.SHAFT_ID] = shaft_angle
        robot.joint_vel[0, robot.SHAFT_ID] = 0.0
        apply_cam_hip_slave(binding, robot)
        want_hip, _ = cam_shaft_to_hip(torch.tensor([shaft_angle]), torch.tensor([0.0]))
        math_hip = math.atan2(k * math.sin(shaft_angle), 1.0 + k * math.cos(shaft_angle))
        assert robot.position_target is not None
        assert robot.position_target.item() == pytest.approx(want_hip.item(), abs=1e-6)
        assert robot.position_target.item() == pytest.approx(math_hip, abs=1e-6)

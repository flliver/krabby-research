"""Host hall shim: ``saf`` edges track Isaac joint travel (OVERVIEW 2d)."""

from __future__ import annotations

import pytest
import torch

from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"
HALL_TOLERANCE_FRAC = 0.02
HALL_TOLERANCE_MIN = 2


def _hall_slot(config, actuator_id: str) -> int:
    return config.slot_by_firmware_id(actuator_id).slot_index


def _write_joint_angle(robot, joint_index: int, angle: float) -> None:
    with torch.inference_mode():
        pos = robot.data.joint_pos.clone()
        vel = robot.data.default_joint_vel.clone()
        pos[0, joint_index] = angle
        robot.write_joint_state_to_sim(pos, vel)


def test_stop_to_stop_travel_matches_configured_hall_edges(front_board_rig):
    core, robot, config, indices = front_board_rig
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]
    hall_slot = _hall_slot(config, SLOT)
    limits = robot.data.soft_joint_pos_limits[0, joint_index]
    lo = limits[0].item()
    hi = limits[1].item()

    park_at_default_pose(core, robot)
    _write_joint_angle(robot, joint_index, lo)
    core.step()
    _write_joint_angle(robot, joint_index, hi)
    core.step()

    edges = core.bridge.hall_edge_count(hall_slot)
    expected = config.hall_edges_per_full_travel
    tol = max(HALL_TOLERANCE_MIN, int(HALL_TOLERANCE_FRAC * expected))
    assert abs(edges - expected) <= tol, f"got {edges}, expected {expected} ± {tol}"


def test_hall_count_holds_when_joint_is_stationary(front_board_rig):
    core, robot, config, indices = front_board_rig
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]
    hall_slot = _hall_slot(config, SLOT)

    park_at_default_pose(core, robot)
    angle = robot.data.joint_pos[0, joint_index].item()
    _write_joint_angle(robot, joint_index, angle)
    core.step()
    core.step()
    frozen = core.bridge.hall_edge_count(hall_slot)

    for _ in range(8):
        core.jog(SLOT, 0)
        core.step()
    assert core.bridge.hall_edge_count(hall_slot) == frozen


def test_jog_increases_hall_monotonically(front_board_rig):
    core, robot, config, indices = front_board_rig
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]
    hall_slot = _hall_slot(config, SLOT)

    park_at_default_pose(core, robot)
    core.step()
    start = core.bridge.hall_edge_count(hall_slot)
    start_pos = robot.data.joint_pos[0, joint_index].item()

    core.jog(SLOT, 255)
    last = start
    for _ in range(40):
        core.step()
        count = core.bridge.hall_edge_count(hall_slot)
        assert count >= last
        last = count

    moved = robot.data.joint_pos[0, joint_index].item() - start_pos
    assert moved > 0.04
    assert last > start + 50

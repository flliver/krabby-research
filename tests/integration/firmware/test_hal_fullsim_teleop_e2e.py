"""E2E: ZMQ joint command → fullsim HAL → firmware → env.step → joint + observation."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import JointCommand
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.integration.firmware.conftest import park_at_default_pose

pytestmark = pytest.mark.isaacsim

HAL_JOINT = "FL_hip_pitch"
DRIVE_STEPS = 40
COMMAND_RAD = 0.25
MIN_JOINT_DELTA_RAD = 0.03


def _make_teleop_command(values: list[float]) -> JointCommand:
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    return JointCommand(
        _joint_positions=np.array(values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )


@pytest.fixture(scope="module")
def fullsim_hal_server(crab_hex_play_env):
    from firmware.sim.isaac_scene import (
        configure_open_loop_velocity_drive,
        resolve_eighteen_joint_indices,
    )
    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
    from hal.server import HalServerConfig
    from hal.server.isaac.hal_server import IsaacSimHalServer

    env, get_robot, _resolve_front = crab_hex_play_env
    robot = get_robot(env)
    configs = ThreeBoardSimConfigs.rev3()
    indices = resolve_eighteen_joint_indices(robot, configs)
    configure_open_loop_velocity_drive(robot, list(indices.values()))

    obs_dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    server_cfg = HalServerConfig(
        observation_bind="inproc://hal_fullsim_teleop_obs",
        command_bind="inproc://hal_fullsim_teleop_cmd",
    )
    server = IsaacSimHalServer(
        server_cfg,
        KRABBY_HEX_DEFINITION,
        env=env,
        observation_dimensions=obs_dims,
        command_backend="fullsim",
        firmware_transport="in_process",
        command_timeout_s=10.0,
    )
    server.initialize()
    yield server, env, robot, indices
    server.close()


def test_hal_fullsim_teleop_command_e2e(fullsim_hal_server):
    """Teleop path: PUSH JointCommand → apply_command → env.step → joint and HAL observation move."""
    server, env, robot, indices = fullsim_hal_server
    assert server._three_board_core is not None

    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs

    configs = ThreeBoardSimConfigs.rev3()
    slot = configs.front.slot_by_firmware_id("FLHL")
    joint_index = indices[slot.isaac_joint_name]
    hal_joint_idx = KRABBY_HEX_DEFINITION.get_joint_names().index(HAL_JOINT)

    park_at_default_pose(server._three_board_core, robot)

    client_cfg = HalClientConfig(
        observation_endpoint="inproc://hal_fullsim_teleop_obs",
        command_endpoint="inproc://hal_fullsim_teleop_cmd",
    )
    client = HalClient(client_cfg, context=server.get_transport_context())
    client.initialize()

    try:
        server.set_observation()
        obs_before = client.poll(timeout_ms=3000)
        assert obs_before is not None
        joint_obs_before = float(obs_before.joint_positions[hal_joint_idx])

        before_sim = robot.data.joint_pos[0, joint_index].item()

        values = [0.0] * 18
        values[hal_joint_idx] = COMMAND_RAD
        command = _make_teleop_command(values)

        client.put_joint_command(command)
        action = server.apply_command()
        assert action.shape == (1, 18)
        assert abs(action[0, hal_joint_idx].item()) > 0.05

        # Play-task action terms fight open-loop velocity; step firmware→articulation directly.
        core = server._three_board_core
        with torch.inference_mode():
            for _ in range(DRIVE_STEPS):
                core.step()

        after_sim = robot.data.joint_pos[0, joint_index].item()
        assert abs(after_sim - before_sim) > MIN_JOINT_DELTA_RAD

        server._latest_obs_dict = None
        server._latest_obs_tensor = None
        server.set_observation()
        obs_after = client.poll(timeout_ms=3000)
        assert obs_after is not None
        joint_obs_after = float(obs_after.joint_positions[hal_joint_idx])
        assert joint_obs_after != pytest.approx(joint_obs_before, abs=1e-5)
    finally:
        client.close()

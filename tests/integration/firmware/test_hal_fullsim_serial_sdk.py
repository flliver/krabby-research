"""Fullsim HAL over KrabbyMCUSDK + PTY (Jetson wire protocol)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch

from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import JointCommand
from hal.server import HalServerConfig
from hal.server.isaac.firmware_serial_loop import start_serial_sdk_session
from hal.server.isaac.hal_server import IsaacSimHalServer
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK as HalKrabbyMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.integration.firmware.conftest import park_at_default_pose
from tests.integration.firmware.test_mcu_sdk_pty_jog import _fast_connect_sleep

pytestmark = pytest.mark.isaacsim

HAL_JOINT = "FL_hip_pitch"
DRIVE_STEPS = 40
COMMAND_RAD = 0.25
MIN_JOINT_DELTA_RAD = 0.03


def _make_hex_command(joint_values: list[float]) -> JointCommand:
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    return JointCommand(
        _joint_positions=np.array(joint_values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )


@pytest.fixture(scope="module")
def serial_sdk_hal_server(crab_hex_play_env):
    if not hasattr(os, "openpty"):
        pytest.skip("PTY requires Unix")

    from firmware.sim.isaac_scene import (
        configure_open_loop_velocity_drive,
        resolve_eighteen_joint_indices,
    )
    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs

    env, get_robot, _resolve_front = crab_hex_play_env
    robot = get_robot(env)
    configs = ThreeBoardSimConfigs.rev3()
    indices = resolve_eighteen_joint_indices(robot, configs)
    configure_open_loop_velocity_drive(robot, list(indices.values()))

    obs_dims = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        KRABBY_HEX_DEFINITION
    )
    server_cfg = HalServerConfig(
        observation_bind="inproc://hal_fullsim_serial_sdk_obs",
        command_bind="inproc://hal_fullsim_serial_sdk_cmd",
    )

    real_sleep = time.sleep

    def fast_sleep(seconds: float) -> None:
        if seconds >= 5.0:
            real_sleep(0.05)
        else:
            real_sleep(seconds)

    time.sleep = fast_sleep
    try:
        server = IsaacSimHalServer(
            server_cfg,
            KRABBY_HEX_DEFINITION,
            env=env,
            observation_dimensions=obs_dims,
            command_backend="fullsim",
            firmware_transport="serial_sdk",
            command_timeout_s=10.0,
        )
    finally:
        time.sleep = real_sleep

    server.initialize()
    session = server._fullsim_session
    assert session is not None
    assert session.firmware_transport == "serial_sdk"
    assert session.hal_krabby_sdk is not None
    assert isinstance(session.hal_krabby_sdk, HalKrabbyMCUSDK)
    assert session.hal_krabby_sdk.is_connected()

    yield server, env, robot, indices, session
    server.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_fullsim_serial_sdk_drives_firmware_pwm(eighteen_joint_rig, monkeypatch):
    _fast_connect_sleep(monkeypatch)
    core, robot, _configs, _indices = eighteen_joint_rig

    loop, session = start_serial_sdk_session(
        core,
        KRABBY_HEX_DEFINITION,
        connect_timeout_s=5.0,
    )
    try:
        sdk = IsaacSimFirmwareMCUSDK.from_robot_definition(
            KRABBY_HEX_DEFINITION,
            device=robot.device,
            firmware_loop=loop,
        )
        hal_joint_idx = KRABBY_HEX_DEFINITION.get_joint_names().index(HAL_JOINT)
        values = [0.0] * 18
        values[hal_joint_idx] = COMMAND_RAD

        sdk.apply_command(_make_hex_command(values))

        assert core.command_pwm_for_actuator("FLHL") != 0
        assert session.hal_krabby_sdk is not None
        assert session.pty_slave_path
    finally:
        session.close()


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_hal_fullsim_serial_sdk_teleop_command_e2e(serial_sdk_hal_server):
    """ZMQ JointCommand → HAL fullsim (serial_sdk) → KrabbyMCUSDK → joint + observation."""
    server, env, robot, indices, session = serial_sdk_hal_server
    assert server._three_board_core is not None

    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs

    configs = ThreeBoardSimConfigs.rev3()
    slot = configs.front.slot_by_firmware_id("FLHL")
    joint_index = indices[slot.isaac_joint_name]
    hal_joint_idx = KRABBY_HEX_DEFINITION.get_joint_names().index(HAL_JOINT)

    park_at_default_pose(server._three_board_core, robot)

    client_cfg = HalClientConfig(
        observation_endpoint="inproc://hal_fullsim_serial_sdk_obs",
        command_endpoint="inproc://hal_fullsim_serial_sdk_cmd",
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
        command = _make_hex_command(values)

        client.put_joint_command(command)
        action = server.apply_command()
        assert action.shape == (1, 18)
        assert abs(action[0, hal_joint_idx].item()) > 0.05
        assert session.hal_krabby_sdk is not None
        assert session.hal_krabby_sdk.is_connected()

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

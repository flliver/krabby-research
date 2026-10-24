"""Regression: fullsim command backend must not change HAL observation keys/shapes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.isaac.hal_server import IsaacSimHalServer
from hal.server.isaac.observation_schema import hardware_observation_schema
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

_HEX_OBS_DIMS = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
    KRABBY_HEX_DEFINITION
)
_NUM_JOINTS = KRABBY_HEX_DEFINITION.get_total_joint_count()


def _mock_hex_isaac_env() -> MagicMock:
    mock_contact_sensor = MagicMock()
    mock_contact_sensor.cfg = MagicMock()
    mock_contact_sensor.cfg.body_ids = list(range(6))
    mock_contact_sensor.data = MagicMock()
    mock_contact_sensor.data.net_forces_w_history = torch.zeros(
        (1, 10, 6, 3), dtype=torch.float32
    )

    mock_robot = MagicMock()
    mock_robot.data = MagicMock()
    mock_robot.data.joint_pos = torch.zeros((1, _NUM_JOINTS), dtype=torch.float32)
    mock_robot.data.root_ang_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_lin_vel_b = torch.zeros((1, 3), dtype=torch.float32)
    mock_robot.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    mock_robot.data.joint_vel = torch.zeros((1, _NUM_JOINTS), dtype=torch.float32)

    mock_scene = MagicMock()
    mock_scene.__getitem__ = MagicMock(return_value=mock_robot)
    mock_scene.__contains__ = MagicMock(return_value=True)
    mock_scene.keys = MagicMock(return_value=["robot"])
    mock_scene.sensors = {"contact_forces": mock_contact_sensor}

    env = MagicMock()
    env.scene = mock_scene
    env.unwrapped = env
    env.unwrapped.num_envs = 1
    env.device = torch.device("cpu")
    env.observation_manager = MagicMock()
    env.action_manager = MagicMock()
    env.action_manager.get_term = MagicMock(
        return_value=MagicMock(
            action_history_buf=torch.zeros((1, 10, _NUM_JOINTS), dtype=torch.float32)
        )
    )
    return env


def _publish_one_observation(
    command_backend: str,
    *,
    obs_bind: str,
    cmd_bind: str,
) -> dict:
    env = _mock_hex_isaac_env()
    server_cfg = HalServerConfig(observation_bind=obs_bind, command_bind=cmd_bind)
    client_cfg = HalClientConfig(
        observation_endpoint=obs_bind,
        command_endpoint=cmd_bind,
    )

    patch_target = "hal.server.isaac.firmware_three_board_setup.attach_fullsim_firmware_to_env"
    with patch(patch_target) as mock_attach:
        if command_backend == "fullsim":
            from hal.server.isaac.firmware_serial_loop import FullsimFirmwareSession

            mock_attach.return_value = FullsimFirmwareSession(
                core=MagicMock(name="ThreeBoardRobotCore"),
                firmware_loop=MagicMock(name="FirmwareCommandLoop"),
                firmware_transport="in_process",
            )

        server = IsaacSimHalServer(
            server_cfg,
            KRABBY_HEX_DEFINITION,
            env=env,
            observation_dimensions=_HEX_OBS_DIMS,
            command_backend=command_backend,  # type: ignore[arg-type]
        )
        server.initialize()
        obs_tensor = torch.ones(_HEX_OBS_DIMS.obs_dim, dtype=torch.float32) * 0.1
        server.observation_manager.compute = MagicMock(return_value={"policy": obs_tensor})

        client = HalClient(client_cfg, context=server.get_transport_context())
        client.initialize()
        try:
            server.set_observation()
            hw_obs = client.poll(timeout_ms=2000)
            assert hw_obs is not None
            return hardware_observation_schema(hw_obs)
        finally:
            client.close()
            server.close()


@pytest.mark.parametrize(
    ("direct_bind", "fullsim_bind"),
    [
        ("inproc://obs_direct", "inproc://obs_fullsim"),
    ],
)
def test_fullsim_observation_schema_matches_direct(direct_bind, fullsim_bind):
    direct_schema = _publish_one_observation(
        "direct",
        obs_bind=direct_bind,
        cmd_bind=direct_bind.replace("obs_", "cmd_"),
    )
    fullsim_schema = _publish_one_observation(
        "fullsim",
        obs_bind=fullsim_bind,
        cmd_bind=fullsim_bind.replace("obs_", "cmd_"),
    )

    assert direct_schema.keys() == fullsim_schema.keys()
    assert direct_schema == fullsim_schema
    assert direct_schema["joint_positions"][0] == (_NUM_JOINTS,)

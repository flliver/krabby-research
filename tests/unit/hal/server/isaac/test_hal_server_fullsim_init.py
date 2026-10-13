"""IsaacSimHalServer fullsim backend wiring (mocked host firmware)."""

from unittest.mock import MagicMock, patch

import torch

from hal.server import HalServerConfig
from hal.server.isaac.hal_server import IsaacSimHalServer
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.isaac.isaacsim_mcusdk import IsaacSimMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.unit.hal.server.isaac.test_hal_server_mcusdk import MockEnv


@patch("hal.server.isaac.firmware_three_board_setup.attach_three_board_firmware_to_env")
def test_fullsim_initializes_firmware_mcusdk_on_existing_env(mock_attach):
    mock_core = MagicMock(name="ThreeBoardRobotCore")
    mock_loop = MagicMock(name="ThreeBoardCommandLoop")
    mock_attach.return_value = (mock_core, mock_loop)

    env = MockEnv(num_envs=1)
    config = HalServerConfig(
        observation_bind="inproc://test_fullsim_obs",
        command_bind="inproc://test_fullsim_cmd",
    )
    server = IsaacSimHalServer(
        config,
        KRABBY_HEX_DEFINITION,
        env=env,
        command_backend="fullsim",
    )

    mock_attach.assert_called_once()
    call_env, call_robot, call_def = mock_attach.call_args[0]
    assert call_env is env
    assert call_robot is server.robot
    assert call_def is KRABBY_HEX_DEFINITION
    assert server._three_board_core is mock_core
    assert isinstance(server._mcusdk, IsaacSimFirmwareMCUSDK)


def test_direct_backend_still_uses_isaacsim_mcusdk():
    env = MockEnv(num_envs=1)
    config = HalServerConfig(
        observation_bind="inproc://test_direct_obs",
        command_bind="inproc://test_direct_cmd",
    )
    server = IsaacSimHalServer(
        config,
        KRABBY_HEX_DEFINITION,
        env=env,
        command_backend="direct",
    )
    assert server._three_board_core is None
    assert isinstance(server._mcusdk, IsaacSimMCUSDK)

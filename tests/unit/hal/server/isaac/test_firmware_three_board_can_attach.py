"""attach_fullsim_firmware_to_env can transport delegation."""

from unittest.mock import MagicMock, patch

from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION


@patch("hal.server.isaac.firmware_can_loop.start_can_stm_session")
def test_attach_fullsim_can_delegates(mock_start):
    from hal.server.isaac.firmware_three_board_setup import attach_fullsim_firmware_to_env

    env = MagicMock()
    robot = MagicMock()
    expected = MagicMock(name="session")
    mock_start.return_value = expected

    session = attach_fullsim_firmware_to_env(
        env,
        robot,
        KRABBY_HEX_DEFINITION,
        firmware_transport="can",
    )

    mock_start.assert_called_once_with(env, robot, KRABBY_HEX_DEFINITION)
    assert session is expected
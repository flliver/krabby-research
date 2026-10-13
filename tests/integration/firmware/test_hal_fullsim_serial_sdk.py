"""Fullsim HAL command path over KrabbyMCUSDK + PTY (Jetson wire protocol)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_serial_loop import start_serial_sdk_session
from hal.server.isaac.isaacsim_firmware_mcusdk import IsaacSimFirmwareMCUSDK
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from tests.integration.firmware.test_mcu_sdk_pty_jog import _fast_connect_sleep

pytestmark = pytest.mark.isaacsim


def _make_hex_command(joint_values: list[float]) -> JointCommand:
    names = KRABBY_HEX_DEFINITION.get_joint_names()
    return JointCommand(
        _joint_positions=np.array(joint_values, dtype=np.float32),
        timestamp_ns=time.time_ns(),
        observation_timestamp_ns=time.time_ns(),
        joint_names=names,
    )


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
        names = KRABBY_HEX_DEFINITION.get_joint_names()
        values = [0.0] * 18
        values[names.index("ML_hip_yaw")] = 0.25

        sdk.apply_command(_make_hex_command(values))

        assert core.command_pwm_for_actuator("MLHY") != 0
        assert session.hal_krabby_sdk is not None
        assert session.pty_slave_path
    finally:
        session.close()

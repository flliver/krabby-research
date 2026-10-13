"""Unit tests for three-board firmware attachment logging."""

import logging

import pytest

from hal.server.isaac.firmware_three_board_setup import log_firmware_joint_bindings
from hal.server.robot_definition import ObservationScalingDefinition, RobotDefinition

_MINI_HEX = RobotDefinition(
    name="mini_hex",
    legs=("FL",),
    joint_types=("hip_yaw", "hip_pitch"),
    observation_scaling=ObservationScalingDefinition(
        base_ang_vel=0.25,
        joint_vel=0.05,
        base_lin_vel=2.0,
    ),
    mcu_joints=("FL_hip_yaw", "FL_hip_pitch"),
)


def test_log_firmware_joint_bindings_emits_per_joint(caplog):
    caplog.set_level(logging.INFO)
    indices = {
        "FL_Body_Hip_RevoluteJoint": 0,
        "FL_Hip_Femur_RevoluteJoint": 1,
    }
    log_firmware_joint_bindings(
        _MINI_HEX,
        indices,
        firmware_actuator_id_for_hal_joint={
            "FL_hip_yaw": "FLHY",
            "FL_hip_pitch": "FLHL",
        },
        isaac_joint_for_firmware_id={
            "FLHY": "FL_Body_Hip_RevoluteJoint",
            "FLHL": "FL_Hip_Femur_RevoluteJoint",
        },
    )
    text = caplog.text
    assert "one process" in text
    assert "FL_hip_yaw" in text and "FLHY" in text
    assert "FL_hip_pitch" in text and "FLHL" in text
    assert "dof_index=0" in text
    assert "dof_index=1" in text


def test_attach_requires_eighteen_mcu_joints():
    from hal.server.isaac.firmware_three_board_setup import attach_three_board_firmware_to_env
    from hal.server.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION

    with pytest.raises(ValueError, match="18 mcu_joints"):
        attach_three_board_firmware_to_env(object(), object(), KRABBY_QUAD_DEFINITION)

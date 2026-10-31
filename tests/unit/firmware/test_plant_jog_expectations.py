"""Plant vs ref jog coupling skip rules."""

from unittest.mock import MagicMock

from firmware.sim.plant_jog_expectations import (
    cross_leg_jog_coupling_ratio,
    skip_plant_jog_coupling_check,
)


def test_skip_same_leg_and_body_hip_on_plant():
    robot = MagicMock()
    robot.find_joints.side_effect = lambda names, preserve_order=True: (
        ([0], ["FL_Body_CamShaft_RevoluteJoint"])
        if names == ["FL_Body_CamShaft_RevoluteJoint"]
        else ([], [])
    )
    cmd = "FL_Hip_Femur_RevoluteJoint"
    assert skip_plant_jog_coupling_check(robot, cmd, "FL_Femur_Tibia_RevoluteJoint")
    assert skip_plant_jog_coupling_check(robot, cmd, "FL_Body_Hip_RevoluteJoint")
    assert not skip_plant_jog_coupling_check(robot, cmd, "FR_Hip_Femur_RevoluteJoint")


def test_cross_leg_ratio_stricter_on_ref():
    plant = MagicMock()
    plant.find_joints.side_effect = lambda names, preserve_order=True: (
        ([0], ["FL_Body_CamShaft_RevoluteJoint"])
        if names == ["FL_Body_CamShaft_RevoluteJoint"]
        else ([], [])
    )
    ref = MagicMock()
    ref.find_joints.side_effect = lambda names, preserve_order=True: (
        ([0], ["lf_hip_yaw"]) if names == ["lf_hip_yaw"] else ([], [])
    )
    assert cross_leg_jog_coupling_ratio(plant) < cross_leg_jog_coupling_ratio(ref)

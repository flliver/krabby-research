"""Three-board REV3 configs cover all 18 Krabby hex MCU joints."""

from firmware.sim.simulated_robot_config import (
    KRABBY_HEX_FIRMWARE_ACTUATOR_IDS,
    ThreeBoardSimConfigs,
    isaac_joint_name_for_firmware_id,
    isaac_joint_name_for_firmware_id_crab_hex_ref,
)


def test_eighteen_firmware_ids_map_to_distinct_isaac_joints():
    configs = ThreeBoardSimConfigs.rev3()
    configs.validate_eighteen_distinct_joints()
    assert configs.firmware_actuator_ids() == (
        configs.front.firmware_actuator_ids()
        + configs.left.firmware_actuator_ids()
        + configs.right.firmware_actuator_ids()
    )
    assert set(configs.firmware_actuator_ids()) == set(KRABBY_HEX_FIRMWARE_ACTUATOR_IDS)


def test_isaac_joint_name_helper_matches_slot_table():
    configs = ThreeBoardSimConfigs.rev3()
    for slot in configs.all_slots():
        assert slot.isaac_joint_name == isaac_joint_name_for_firmware_id(
            slot.firmware_actuator_id
        )


def test_crab_hex_ref_joint_names_are_distinct_and_cover_eighteen_actuators():
    configs = ThreeBoardSimConfigs.rev3()
    ref_names = tuple(
        isaac_joint_name_for_firmware_id_crab_hex_ref(slot.firmware_actuator_id)
        for slot in configs.all_slots()
    )
    assert len(ref_names) == 18
    assert len(set(ref_names)) == 18
    assert isaac_joint_name_for_firmware_id_crab_hex_ref("FLHY") == "lf_hip_yaw"
    assert isaac_joint_name_for_firmware_id_crab_hex_ref("FRKL") == "rf_knee_actuator_joint"

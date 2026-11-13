"""Six leg nodes, eighteen joints, shared C/Python identity map."""

from __future__ import annotations

from pathlib import Path

import pytest

from firmware.stm import hal_joint_names, ids
from tests.unit.firmware.stm.test_krabby_leg_dbc import EXPECTED_FRAME_IDS, LEGS

REPO_ROOT = Path(__file__).resolve().parents[4]
BOARD_IDS_H = REPO_ROOT / "firmware" / "stm" / "include" / "krabby" / "board_ids.h"


def test_leg_and_joint_id_ranges() -> None:
    assert ids.LEG_IDS == (1, 2, 3, 4, 5, 6)
    assert ids.LEG_IDS_LEFT == (1, 3, 5)
    assert ids.LEG_IDS_RIGHT == (2, 4, 6)
    assert ids.JOINT_IDS == (1, 2, 3)
    assert ids.JOINT_LETTERS == "YHK"


def test_leg_id_index_round_trip() -> None:
    for leg_id in ids.LEG_IDS:
        assert ids.leg_index_to_id(ids.leg_id_to_index(leg_id)) == leg_id


def test_joint_id_index_round_trip() -> None:
    for joint_id in ids.JOINT_IDS:
        assert ids.joint_index_to_id(ids.joint_id_to_index(joint_id)) == joint_id


def test_hal_joint_names_match_robot_definition() -> None:
    assert len(hal_joint_names.MCU_JOINTS) == 18


@pytest.mark.parametrize("leg_id", ids.LEG_IDS)
@pytest.mark.parametrize("joint_id", ids.JOINT_IDS)
def test_hal_name_round_trip(leg_id: int, joint_id: int) -> None:
    name = hal_joint_names.mcu_joint_name(leg_id, joint_id)
    assert hal_joint_names.parse_mcu_joint_name(name) == (leg_id, joint_id)
    bus, leg_index, joint_index = hal_joint_names.joint_to_leg_bus(leg_id, joint_id)
    assert leg_index == ids.leg_id_to_index(leg_id)
    assert joint_index == ids.joint_id_to_index(joint_id)
    assert bus == ids.LEG_BUS_BY_ID[leg_id]


def test_leg_can_message_ids_no_collisions() -> None:
    cmd = {ids.leg_command_can_id(i) for i in range(6)}
    tel = {ids.leg_telemetry_can_id(i) for i in range(6)}
    stat = {ids.node_status_can_id(i) for i in range(6)}
    qimu = {ids.qwiic_imu_can_id(i) for i in range(6)}
    assert len(cmd) == len(tel) == len(stat) == len(qimu) == 6
    assert cmd.isdisjoint(tel)
    assert cmd.isdisjoint(stat)
    assert cmd.isdisjoint(qimu)
    assert tel.isdisjoint(stat)
    assert tel.isdisjoint(qimu)
    assert stat.isdisjoint(qimu)


def test_leg_can_ids_match_dbc() -> None:
    for i, leg in enumerate(LEGS):
        leg_id = ids.LEG_ID_BY_DBC_SUFFIX[leg]
        assert ids.leg_id_to_index(leg_id) == i
        assert ids.leg_command_can_id_for_leg(leg_id) == (
            EXPECTED_FRAME_IDS[f"LegCommand_{leg}"] & 0x1FFFFFFF
        )
        assert ids.leg_telemetry_can_id_for_leg(leg_id) == (
            EXPECTED_FRAME_IDS[f"LegTelemetry_{leg}"] & 0x1FFFFFFF
        )
        assert ids.node_status_can_id_for_leg(leg_id) == (
            EXPECTED_FRAME_IDS[f"NodeStatus_{leg}"] & 0x1FFFFFFF
        )
        assert ids.qwiic_imu_can_id_for_leg(leg_id) == (
            EXPECTED_FRAME_IDS[f"QwiicImu_{leg}"] & 0x1FFFFFFF
        )


def test_provisioning_switch_equals_leg_id() -> None:
    for leg_id in ids.LEG_IDS:
        assert ids.leg_index_to_id(ids.leg_id_to_index(leg_id)) == leg_id
    assert ids.LEG_INDEX_UNASSIGNED == 6
    assert ids.LEG_CODE_UNASSIGNED == 255


def test_board_ids_header_defines_match_python() -> None:
    text = BOARD_IDS_H.read_text(encoding="utf-8")
    assert "KRABBY_LEG_COUNT 6" in text
    assert "KRABBY_LEG_CODE_UNASSIGNED 255" in text
    assert "KRABBY_CAN_ID_LEG_COMMAND_BASE 0x0C000000" in text
    assert "KRABBY_CAN_ID_PROVISION_SET 0x0C0007E0" in text

"""Six leg nodes, eighteen joints, shared C/Python identity map."""

from __future__ import annotations

from pathlib import Path

import pytest

from firmware.stm import hal_joint_names, ids
from tests.unit.firmware.stm.test_krabby_dbc import EXPECTED_FRAME_IDS

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


def test_telemetry_and_board_info_frame_leg_id_round_trip() -> None:
    for leg_id in ids.LEG_IDS:
        assert ids.telemetry_leg_id_from_frame(ids.telemetry_can_id(leg_id)) == leg_id
        assert ids.board_info_leg_id_from_frame(ids.board_info_can_id(leg_id)) == leg_id
    assert ids.telemetry_leg_id_from_frame(ids.CAN_ID_TELEMETRY_BASE) is None
    assert ids.board_info_leg_id_from_frame(ids.CAN_ID_BOARD_INFO_BASE) is None
    assert ids.telemetry_leg_id_from_frame(ids.set_duty_can_id()) is None


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


def test_grant_can_message_ids_unique() -> None:
    frames = {ids.set_duty_can_id()}
    for leg_id in ids.LEG_IDS:
        frames.add(ids.calibrate_can_id(leg_id))
        frames.add(ids.telemetry_can_id(leg_id))
        frames.add(ids.board_info_can_id(leg_id))
    assert len(frames) == 1 + 6 * 3


def test_ids_match_krabby_dbc() -> None:
    assert ids.set_duty_can_id() == EXPECTED_FRAME_IDS["SET_DUTY"]
    for leg_id in ids.LEG_IDS:
        assert ids.calibrate_can_id(leg_id) == EXPECTED_FRAME_IDS[f"CALIBRATE_{leg_id}"]
        assert ids.telemetry_can_id(leg_id) == EXPECTED_FRAME_IDS[f"TELEMETRY_{leg_id}"]
        assert ids.board_info_can_id(leg_id) == EXPECTED_FRAME_IDS[f"BOARD_INFO_{leg_id}"]


def test_provisioning_switch_equals_leg_id() -> None:
    for leg_id in ids.LEG_IDS:
        assert ids.leg_index_to_id(ids.leg_id_to_index(leg_id)) == leg_id
    assert ids.LEG_INDEX_UNASSIGNED == 6
    assert ids.LEG_CODE_UNASSIGNED == 255


def test_board_ids_header_defines_match_python() -> None:
    text = BOARD_IDS_H.read_text(encoding="utf-8")
    assert "KRABBY_LEG_COUNT 6" in text
    assert "KRABBY_CAN_ID_SET_DUTY 0x100" in text
    assert "KRABBY_CAN_ID_CALIBRATE_BASE 0x180" in text
    assert "KRABBY_CAN_ID_TELEMETRY_BASE 0x200" in text
    assert "KRABBY_CAN_ID_BOARD_INFO_BASE 0x300" in text
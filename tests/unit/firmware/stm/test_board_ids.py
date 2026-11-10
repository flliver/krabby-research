"""Six leg nodes, eighteen joints, shared C/Python identity map."""

from __future__ import annotations

from pathlib import Path

import pytest

from firmware.stm import ids
from tests.unit.firmware.stm.test_krabby_leg_dbc import EXPECTED_FRAME_IDS, LEGS

# Keep in sync with ``KRABBY_HEX_DEFINITION.mcu_joints`` (hal/server/robot_definition_krabby_hex.py).
EXPECTED_MCU_JOINTS: tuple[str, ...] = (
    "FL_hip_yaw",
    "FL_hip_pitch",
    "FL_knee",
    "FR_hip_yaw",
    "FR_hip_pitch",
    "FR_knee",
    "ML_hip_yaw",
    "ML_hip_pitch",
    "ML_knee",
    "MR_hip_yaw",
    "MR_hip_pitch",
    "MR_knee",
    "RL_hip_yaw",
    "RL_hip_pitch",
    "RL_knee",
    "RR_hip_yaw",
    "RR_hip_pitch",
    "RR_knee",
)

REPO_ROOT = Path(__file__).resolve().parents[4]
BOARD_IDS_H = REPO_ROOT / "firmware" / "stm" / "include" / "krabby" / "board_ids.h"


def test_mcu_joint_count_six_by_three() -> None:
    assert len(ids.LEG_CODES) == 6
    assert len(ids.JOINT_SUFFIXES) == 3
    assert len(ids.MCU_JOINTS) == 18


def test_mcu_joints_match_robot_definition() -> None:
    assert ids.MCU_JOINTS == EXPECTED_MCU_JOINTS


def test_joint_names_unique() -> None:
    assert len(set(ids.MCU_JOINTS)) == 18


@pytest.mark.parametrize("name", ids.MCU_JOINTS)
def test_joint_to_leg_bus_index_round_trip(name: str) -> None:
    leg, joint_index = ids.parse_mcu_joint_name(name)
    bus, leg_index, ji = ids.joint_to_leg_bus_index(name)
    assert ji == joint_index
    assert leg_index == ids.LEG_CODE_BY_NAME[leg]
    assert bus == ids.LEG_BUS[leg]


def test_leg_can_message_ids_no_collisions() -> None:
    cmd = {ids.leg_command_can_id(i) for i in range(6)}
    tel = {ids.leg_telemetry_can_id(i) for i in range(6)}
    stat = {ids.node_status_can_id(i) for i in range(6)}
    assert len(cmd) == len(tel) == len(stat) == 6
    assert cmd.isdisjoint(tel)
    assert cmd.isdisjoint(stat)
    assert tel.isdisjoint(stat)


def test_leg_can_ids_match_dbc() -> None:
    for i, leg in enumerate(LEGS):
        assert ids.leg_command_can_id(i) == (EXPECTED_FRAME_IDS[f"LegCommand_{leg}"] & 0x1FFFFFFF)
        assert ids.leg_telemetry_can_id(i) == (EXPECTED_FRAME_IDS[f"LegTelemetry_{leg}"] & 0x1FFFFFFF)
        assert ids.node_status_can_id(i) == (EXPECTED_FRAME_IDS[f"NodeStatus_{leg}"] & 0x1FFFFFFF)


def test_provisioning_leg_code_table() -> None:
    assert ids.LEG_CODE_BY_NAME["FL"] == 0
    assert ids.LEG_CODE_BY_NAME["RR"] == 5
    assert ids.LEG_INDEX_UNASSIGNED == 6
    assert ids.LEG_CODE_UNASSIGNED == 255


def test_board_ids_header_defines_match_python() -> None:
    text = BOARD_IDS_H.read_text(encoding="utf-8")
    assert "KRABBY_LEG_COUNT 6" in text
    assert "KRABBY_LEG_CODE_UNASSIGNED 255" in text
    assert "KRABBY_CAN_ID_LEG_COMMAND_BASE 0x0C000000" in text
    assert "KRABBY_CAN_ID_PROVISION_SET 0x0C0007E0" in text

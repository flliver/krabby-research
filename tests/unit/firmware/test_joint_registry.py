"""Guards for the static joint registry (firmware/joints.py).

The registry is a host-side table of per-joint facts (sensor type, board, slot,
direction vocab, jog cap). Its authority is the firmware sketch — these tests
parse arduino.ino as text and fail if the two ever disagree, so the registry
can't silently drift from what's flashed.
"""
import re
from pathlib import Path

import pytest

from firmware import joints
from firmware.joints import JOINTS, spec, board_joints
from firmware.krabby_mcu import ALL_JOINT_NAMES, JOINT_GROUP_NAMES

ARDUINO = Path(__file__).resolve().parents[3] / "firmware" / "arduino"


def _ino() -> str:
    return (ARDUINO / "arduino.ino").read_text()


class TestRegistryShape:
    def test_all_18_joints(self):
        assert len(JOINTS) == 18
        assert {js.leg for js in JOINTS.values()} == set(joints.LEGS)
        assert {js.kind for js in JOINTS.values()} == set(joints.KINDS)

    def test_spec_lookup(self):
        js = spec("FLHY")
        assert js.is_yaw and js.sensor == "HALL" and js.board == "FRONT" and js.slot == 0

    def test_spec_rejects_unknown(self):
        with pytest.raises(ValueError):
            spec("BOGUS")

    def test_kind_facts(self):
        # Yaw: left/right vocab, storm-safe jog cap, never end-stop-swept.
        for leg in joints.LEGS:
            hy, hl, kl = spec(leg + "HY"), spec(leg + "HL"), spec(leg + "KL")
            assert hy.cal_directions == ("left", "right")
            assert hy.jog_pwm_max == 150 and not hy.end_stop_calibratable
            assert hl.cal_directions == kl.cal_directions == ("extend", "retract")
            assert hl.sensor == "HALL" and kl.sensor == "POT"
            assert kl.position_absolute_at_boot and not hl.position_absolute_at_boot

    def test_sdk_name_lists_derive_from_registry(self):
        assert ALL_JOINT_NAMES == frozenset(JOINTS)
        assert {b for b, _ in JOINT_GROUP_NAMES} == set(joints.BOARD_LEGS)
        for board, names in JOINT_GROUP_NAMES:
            assert names == [js.name for js in board_joints(board)]


class TestRegistryMatchesFirmware:
    """Parse arduino.ino's actuator declarations — the ground truth the registry mirrors."""

    # e.g.: LinearActuator flhy("FLHY", ..., 0, SENSOR_HALL);
    _DECL = re.compile(
        r'LinearActuator\s+\w+\("(\w{4})",[^;]*?,\s*(-?\d+),\s*(SENSOR_HALL|SENSOR_POT)\)')

    def _declared(self) -> dict:
        found = {m.group(1): (m.group(2), m.group(3)) for m in self._DECL.finditer(_ino())}
        assert len(found) == 18, f"expected 18 actuator declarations, parsed {len(found)}"
        return found

    def test_sensor_types_match_sketch(self):
        for name, (_, sensor_token) in self._declared().items():
            expected = "SENSOR_HALL" if spec(name).sensor == "HALL" else "SENSOR_POT"
            assert sensor_token == expected, f"{name}: registry says {spec(name).sensor}"

    def test_board_slot_order_matches_act_lists(self):
        # ACT_LIST_FRONT[] = { &flhy, &flhl, ... } — index in the list IS the slot.
        ino = _ino()
        for board in joints.BOARD_LEGS:
            m = re.search(rf"ACT_LIST_{board}\[\]\s*=\s*\{{([^}}]*)\}}", ino)
            assert m, f"ACT_LIST_{board} not found"
            listed = [tok.strip().lstrip("&").upper() for tok in m.group(1).split(",")]
            assert listed == [js.name for js in board_joints(board)], board

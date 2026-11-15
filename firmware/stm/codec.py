"""cantools load/encode/decode for krabby.dbc."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from firmware.stm import ids

_DBC_PATH = Path(__file__).resolve().parent / "dbc" / "krabby.dbc"

SET_DUTY_SIGNALS: Final = tuple(
    f"L{leg}_J{joint}" for leg in ids.LEG_IDS for joint in ids.JOINT_IDS
)


def dbc_path() -> Path:
    return _DBC_PATH


@lru_cache(maxsize=1)
def load_database():
    import cantools

    return cantools.database.load_file(_DBC_PATH, strict=False)


def empty_set_duty_values() -> dict[str, float]:
    return {name: 0.0 for name in SET_DUTY_SIGNALS}


def set_duty_values_from_grid(duties: tuple[tuple[float, float, float], ...]) -> dict[str, float]:
    """``duties`` is six rows (leg_id 1..6) of three joint duties."""
    if len(duties) != 6:
        raise ValueError(f"expected 6 legs, got {len(duties)}")
    payload = empty_set_duty_values()
    for leg_id, row in enumerate(duties, start=1):
        if len(row) != 3:
            raise ValueError(f"leg {leg_id}: expected 3 joints, got {len(row)}")
        for joint_id, duty in enumerate(row, start=1):
            payload[f"L{leg_id}_J{joint_id}"] = float(duty)
    return payload


def set_duty_values_flat(duties: tuple[float, ...]) -> dict[str, float]:
    """Eighteen duties in leg_id 1..6, joint 1..3 order."""
    if len(duties) != 18:
        raise ValueError(f"expected 18 duties, got {len(duties)}")
    rows: list[tuple[float, float, float]] = []
    for i in range(0, 18, 3):
        rows.append((duties[i], duties[i + 1], duties[i + 2]))
    return set_duty_values_from_grid(tuple(rows))


def encode_set_duty(payload: dict[str, float] | None = None) -> tuple[int, bytes]:
    db = load_database()
    msg = db.get_message_by_name("SET_DUTY")
    data = msg.encode(payload if payload is not None else empty_set_duty_values())
    return msg.frame_id, data


def decode_frame(arbitration_id: int, data: bytes) -> tuple[str, dict[str, Any]] | None:
    db = load_database()
    try:
        message = db.get_message_by_frame_id(arbitration_id)
    except KeyError:
        return None
    return message.name, message.decode(data)


def decode_telemetry(leg_id: int, data: bytes) -> dict[str, Any]:
    name = f"TELEMETRY_{leg_id}"
    db = load_database()
    msg = db.get_message_by_name(name)
    if msg.frame_id != ids.telemetry_can_id(leg_id):
        raise ValueError(f"DBC/id mismatch for {name}")
    return msg.decode(data)


def encode_calibrate(leg_id: int, joint_id: int) -> tuple[int, bytes]:
    if leg_id not in ids.LEG_IDS:
        raise ValueError(f"leg_id out of range: {leg_id!r}")
    db = load_database()
    message = db.get_message_by_name(f"CALIBRATE_{leg_id}")
    data = message.encode({"joint": joint_id})
    return message.frame_id, data


def decode_board_info(leg_id: int, data: bytes) -> dict[str, Any]:
    name = f"BOARD_INFO_{leg_id}"
    db = load_database()
    msg = db.get_message_by_name(name)
    if msg.frame_id != ids.board_info_can_id(leg_id):
        raise ValueError(f"DBC/id mismatch for {name}")
    return msg.decode(data)
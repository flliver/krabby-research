"""codec.py encode/decode for krabby.dbc."""

from __future__ import annotations

from firmware.stm import codec, ids


def test_encode_set_duty_matches_dbc_test_vector() -> None:
    payload = codec.set_duty_values_from_grid(((0.5, 0.0, 0.0),) + ((0.0, 0.0, 0.0),) * 5)
    frame_id, data = codec.encode_set_duty(payload)
    assert frame_id == ids.set_duty_can_id()
    assert len(data) == 36
    assert data == (500).to_bytes(2, "little", signed=True) + b"\x00" * 34


def test_set_duty_flat_roundtrip_grid() -> None:
    duties = tuple(0.1 * i for i in range(18))
    payload = codec.set_duty_values_flat(duties)
    grid = codec.set_duty_values_from_grid(
        tuple(tuple(payload[f"L{leg}_J{j}"] for j in ids.JOINT_IDS) for leg in ids.LEG_IDS)
    )
    assert grid == payload


def test_decode_telemetry_message_name() -> None:
    db = codec.load_database()
    msg = db.get_message_by_name("TELEMETRY_1")
    zeros = bytes(msg.length)
    fields = codec.decode_telemetry(1, zeros)
    assert "J1_DUTY" in fields
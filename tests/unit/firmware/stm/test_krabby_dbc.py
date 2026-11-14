"""krabby.dbc: IDs, FD format, integrity pin."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cantools

REPO_ROOT = Path(__file__).resolve().parents[4]
DBC_DIR = REPO_ROOT / "firmware" / "stm" / "dbc"
DBC_PATH = DBC_DIR / "krabby.dbc"
SHA_PATH = DBC_DIR / "krabby.sha256"
GENERATED_C = DBC_DIR / "generated" / "krabby.c"
GENERATED_H = DBC_DIR / "generated" / "krabby.h"

BO_RE = re.compile(r"^BO_\s+(\d+)\s+(\w+):")

EXPECTED_FRAME_IDS = {
    "SET_DUTY": 0x100,
    **{f"CALIBRATE_{n}": 0x180 + n for n in range(1, 7)},
    **{f"TELEMETRY_{n}": 0x200 + n for n in range(1, 7)},
    **{f"BOARD_INFO_{n}": 0x300 + n for n in range(1, 7)},
}


def _parse_bo_names_and_ids(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = BO_RE.match(line.strip())
        if m:
            out[m.group(2)] = int(m.group(1))
    return out


def test_dbc_files_exist() -> None:
    assert DBC_PATH.is_file()
    assert SHA_PATH.is_file()


def test_dbc_sha256_pin() -> None:
    digest = hashlib.sha256(DBC_PATH.read_bytes()).hexdigest()
    pinned = SHA_PATH.read_text().strip()
    assert digest == pinned, "krabby.sha256 out of date; run make -C firmware stm-dbc"


def test_dbc_generated_c_sources_committed() -> None:
    assert GENERATED_C.is_file() and GENERATED_H.is_file()
    header = GENERATED_H.read_text(encoding="utf-8")
    assert "krabby_set_duty" in header.lower() or "SET_DUTY" in header


def test_dbc_message_ids_match_grant() -> None:
    parsed = _parse_bo_names_and_ids(DBC_PATH.read_text())
    assert len(parsed) == 19
    assert set(parsed) == set(EXPECTED_FRAME_IDS)
    for name, frame_id in EXPECTED_FRAME_IDS.items():
        assert parsed[name] == frame_id, name


def test_dbc_loads_and_set_duty_is_fd() -> None:
    db = cantools.database.load_file(DBC_PATH, strict=False)
    msg = db.get_message_by_name("SET_DUTY")
    assert msg.length == 36
    attrs = msg.dbc.attributes or {}
    assert attrs.get("VFrameFormat") is not None
    assert attrs["VFrameFormat"].value == 14


def test_set_duty_encode_leg1_yaw_half() -> None:
    db = cantools.database.load_file(DBC_PATH, strict=False)
    msg = db.get_message_by_name("SET_DUTY")
    assert msg.frame_id == 0x100
    data = msg.encode(
        {
            "L1_J1": 0.5,
            "L1_J2": 0.0,
            "L1_J3": 0.0,
            **{f"L{i}_J{j}": 0.0 for i in range(2, 7) for j in range(1, 4)},
        }
    )
    assert len(data) == 36
    # int16 scale 0.001 → 0.5 duty = 500; grant acceptance 1a layout
    assert data == (500).to_bytes(2, "little", signed=True) + b"\x00" * 34
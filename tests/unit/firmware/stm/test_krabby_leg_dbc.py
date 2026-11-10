"""Canonical leg-bus DBC: IDs, joint coverage, integrity pin."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
DBC_DIR = REPO_ROOT / "firmware" / "stm" / "dbc"
DBC_PATH = DBC_DIR / "krabby_leg.dbc"
SHA_PATH = DBC_DIR / "krabby_leg.sha256"
VERSION_PATH = DBC_DIR / "VERSION"
GENERATED_C = DBC_DIR / "generated" / "krabby_leg.c"
GENERATED_H = DBC_DIR / "generated" / "krabby_leg.h"

LEGS = ("FL", "FR", "ML", "MR", "RL", "RR")

JOINT_SUFFIXES = ("hip_yaw", "hip_pitch", "knee")

# Extended CAN IDs (29-bit base | 0x80000000)
def _ext(base: int) -> int:
    return base | 0x80000000


EXPECTED_FRAME_IDS = {
    **{f"LegCommand_{leg}": _ext(0x0C000000 + i) for i, leg in enumerate(LEGS)},
    **{f"LegTelemetry_{leg}": _ext(0x0C000100 + i) for i, leg in enumerate(LEGS)},
    **{f"NodeStatus_{leg}": _ext(0x0C000200 + i) for i, leg in enumerate(LEGS)},
    "DiscoveryRequest": _ext(0x0C0007F0),
    "ProvisionSet": _ext(0x0C0007E0),
    "FlashSessionOpen": _ext(0x0C0007F1),
    "FlashSessionClose": _ext(0x0C0007F2),
}
for i, leg in enumerate(LEGS):
    EXPECTED_FRAME_IDS[f"FlashXcpRequest_{leg}"] = _ext(0x0C000300 + i)
    EXPECTED_FRAME_IDS[f"FlashXcpResponse_{leg}"] = _ext(0x0C000380 + i)
    EXPECTED_FRAME_IDS[f"FlashProgress_{leg}"] = _ext(0x0C0003C0 + i)


BO_RE = re.compile(r"^BO_\s+(\d+)\s+(\w+):")


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
    assert VERSION_PATH.read_text().strip() == "1.0.0"


def test_dbc_sha256_pin() -> None:
    digest = hashlib.sha256(DBC_PATH.read_bytes()).hexdigest()
    pinned = SHA_PATH.read_text().strip()
    assert digest == pinned, "krabby_leg.sha256 out of date; refresh per firmware/stm/dbc/README.md"


def test_dbc_generated_c_sources_committed() -> None:
    assert GENERATED_C.is_file() and GENERATED_H.is_file()
    header = GENERATED_H.read_text(encoding="utf-8")
    assert "Krabby_leg" in header or "krabby_leg" in header
    assert "KRABBY_LEG_LEG_COMMAND_FL_NAME" in header


def test_dbc_message_ids_match_protocol() -> None:
    parsed = _parse_bo_names_and_ids(DBC_PATH.read_text())
    assert set(parsed) == set(EXPECTED_FRAME_IDS)
    for name, frame_id in EXPECTED_FRAME_IDS.items():
        assert parsed[name] == frame_id, name


def test_dbc_joint_command_and_telemetry_signals() -> None:
    text = DBC_PATH.read_text()
    for leg in LEGS:
        cmd_block = _message_signal_region(text, f"LegCommand_{leg}")
        tel_block = _message_signal_region(text, f"LegTelemetry_{leg}")
        for joint in JOINT_SUFFIXES:
            assert f"{joint}_jog_pwm" in cmd_block, leg
            assert f"{joint}_position_norm" in cmd_block, leg
            assert f"{joint}_position_valid" in cmd_block, leg
            assert f"{joint}_position_norm" in tel_block, leg
            assert f"{joint}_pot_raw" in tel_block, leg
            assert f"{joint}_connection_state" in tel_block, leg
            assert f"{joint}_faults" in tel_block, leg


def test_dbc_discovery_provision_node_status() -> None:
    text = DBC_PATH.read_text()
    assert "DiscoveryRequest" in text
    disc = _message_signal_region(text, "DiscoveryRequest")
    assert "trigger" in disc
    prov = _message_signal_region(text, "ProvisionSet")
    assert "new_leg_code" in prov and "session_nonce" in prov and "crc32" in prov
    for leg in LEGS:
        status = _message_signal_region(text, f"NodeStatus_{leg}")
        assert "leg_index" in status and "run_mode" in status and "firmware_identity_0" in status


def test_dbc_loads_with_cantools() -> None:
    cantools = pytest.importorskip("cantools")
    db = cantools.database.load_file(DBC_PATH)
    assert len(db.messages) == len(EXPECTED_FRAME_IDS)
    for joint_idx, leg in enumerate(LEGS):
        msg = db.get_message_by_name(f"LegCommand_{leg}")
        assert (msg.frame_id & 0x1FFFFFFF) == (0x0C000000 + joint_idx)
        names = {s.name for s in msg.signals}
        assert "hip_yaw_jog_pwm" in names and "knee_position_valid" in names


def _message_signal_region(text: str, message_name: str) -> str:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"BO_ ") and f" {message_name}:" in line:
            start = i + 1
            break
    if start is None:
        raise AssertionError(f"message {message_name} not found")
    chunk: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("BO_ "):
            break
        if stripped.startswith("SG_"):
            chunk.append(stripped)
    return "\n".join(chunk)

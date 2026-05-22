"""Smoke test: firmware update → show → VER comparison against S3 manifest."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import requests

S3_BASE = "https://krabby-firmware-public.s3.amazonaws.com"
BOARD_COUNT = 3

if "" not in sys.path:
    sys.path.insert(0, "")


@dataclass
class SmokeResult:
    ok: bool
    step: str = ""
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    ver_observed: list[str] = field(default_factory=list)
    ver_expected: Optional[str] = None


def run_smoke(firmware_channel: str, image_ref: str) -> SmokeResult:
    """Run the full smoke sequence against the currently installed image."""
    # Step 1: discover board ports
    rc, out, err = _run_firmware(image_ref, ["show"])
    if rc != 0:
        return SmokeResult(ok=False, step="firmware_show_ports", detail=f"exit {rc}", stdout=out, stderr=err)

    ports = _parse_ports(out)
    if len(ports) < BOARD_COUNT:
        return SmokeResult(
            ok=False, step="firmware_show_ports",
            detail=f"expected {BOARD_COUNT} ports, got {len(ports)}",
            stdout=out, stderr=err,
        )

    # Step 2: update each port
    for port in ports:
        rc, out_u, err_u = _run_firmware(image_ref, ["update", firmware_channel, port])
        if rc != 0:
            return SmokeResult(ok=False, step="firmware_update", detail=f"exit {rc} ({port})", stdout=out_u, stderr=err_u)

    # Step 3: re-show to get post-update versions
    rc, out, err = _run_firmware(image_ref, ["show"])
    if rc != 0:
        return SmokeResult(ok=False, step="firmware_show", detail=f"exit {rc}", stdout=out, stderr=err)

    ver_observed = _parse_versions(out)
    if len(ver_observed) < BOARD_COUNT:
        return SmokeResult(
            ok=False, step="firmware_show",
            detail=f"expected 3 boards, got {len(ver_observed)}",
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    if len(set(ver_observed)) != 1:
        return SmokeResult(
            ok=False, step="ver_mismatch",
            detail=f"boards disagree: {ver_observed}",
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    # Step 4: compare against S3 manifest
    try:
        ver_expected = _fetch_expected_ver(firmware_channel)
    except Exception as exc:
        return SmokeResult(
            ok=False, step="s3_fetch",
            detail=str(exc),
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    if ver_observed[0] != ver_expected:
        return SmokeResult(
            ok=False, step="ver_mismatch_s3",
            detail=f"boards={ver_observed[0]!r} s3={ver_expected!r}",
            stdout=out, stderr=err, ver_observed=ver_observed, ver_expected=ver_expected,
        )

    return SmokeResult(ok=True, ver_observed=ver_observed, ver_expected=ver_expected)


def _run_firmware(image_ref: str, args: list[str]) -> tuple[int, str, str]:
    sys.path.insert(0, "")  # ensure krabby package is importable when installed
    from krabby._docker import firmware_cmd  # type: ignore[import]
    cmd = firmware_cmd(image_ref, args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return result.returncode, result.stdout, result.stderr


def _parse_ports(show_output: str) -> list[str]:
    """Extract unique serial port paths from `krabby-firmware show` output."""
    return re.findall(r"^\s+(/dev/tty\S+)", show_output, re.MULTILINE)


def _parse_versions(show_output: str) -> list[str]:
    """Extract version strings from `krabby-firmware show` output.

    Lines look like: '  /dev/ttyACM0  primary: 0.2.0 (mainline abc1234)'
    """
    return re.findall(r":\s+(\d+\.\d+\.\d+)\s+\(", show_output)


def _fetch_expected_ver(channel: str) -> str:
    latest = requests.get(f"{S3_BASE}/{channel}/latest.json", timeout=10)
    latest.raise_for_status()
    manifest_url = latest.json()["manifest_url"]
    manifest = requests.get(manifest_url, timeout=10)
    manifest.raise_for_status()
    ver_string = manifest.json()["ver_string"]
    return ver_string.split()[0]

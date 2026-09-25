"""Smoke test: firmware update → show → VER comparison against S3 manifest."""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import requests

S3_BASE = "https://krabby-firmware-public.s3.amazonaws.com"
BOARD_COUNT = 3
# PIN_REV=2 is local-testing only (Uno v0.1) and will not be used going forward.
# Boards report "dev-local"; do not flash the published S3 hex over them.
DEV_LOCAL_VER = "dev-local"

if "" not in sys.path:
    sys.path.insert(0, "")

log = logging.getLogger(__name__)


@dataclass
class SmokeResult:
    ok: bool
    step: str = ""
    detail: str = ""
    stdout: str = ""
    stderr: str = ""
    ver_observed: list[str] = field(default_factory=list)
    ver_expected: Optional[str] = None
    skipped_flash: bool = False


def run_smoke(firmware_channel: str, image_ref: str) -> SmokeResult:
    """Run the full smoke sequence against the currently installed image."""
    log.debug("Smoke test starting (channel=%s, image=%s)", firmware_channel, image_ref)

    # Step 1: discover board ports (unique USB paths — often 1 with a UART hub)
    log.debug("Step 1: discovering board ports")
    rc, out, err = _run_firmware(image_ref, ["show"])
    if rc != 0:
        return SmokeResult(ok=False, step="firmware_show_ports", detail=f"exit {rc}", stdout=out, stderr=err)

    ports = _parse_ports(out)
    log.debug("Found %d unique port(s): %s", len(ports), ports)
    if len(ports) < 1:
        return SmokeResult(
            ok=False, step="firmware_show_ports",
            detail=f"expected at least 1 MCU serial port, got {len(ports)}",
            stdout=out, stderr=err,
        )

    ver_before = _parse_versions(out)
    # PIN_REV=2 is local-testing only (not future/published). Skip S3 update so
    # we do not overwrite "dev-local" boards with the published artifact.
    if ver_before and all(v == DEV_LOCAL_VER for v in ver_before):
        if len(ver_before) < BOARD_COUNT:
            return SmokeResult(
                ok=False, step="firmware_show",
                detail=f"dev-local present but only {len(ver_before)} roles (need {BOARD_COUNT})",
                stdout=out, stderr=err, ver_observed=ver_before, skipped_flash=True,
            )
        detail = (
            f"skip flash+S3: all {len(ver_before)} roles are {DEV_LOCAL_VER} "
            f"(PIN_REV=2 local-only — do not take S3 firmware); ports={ports}"
        )
        log.warning("%s", detail)
        return SmokeResult(
            ok=True,
            step="dev_local_skip",
            detail=detail,
            stdout=out,
            stderr=err,
            ver_observed=ver_before,
            ver_expected=DEV_LOCAL_VER,
            skipped_flash=True,
        )
    if any(v == DEV_LOCAL_VER for v in ver_before):
        detail = (
            f"mixed versions include {DEV_LOCAL_VER}: {ver_before} — refusing to flash "
            f"(PIN_REV=2 local-only; would overwrite with S3 firmware)"
        )
        log.warning("%s", detail)
        return SmokeResult(
            ok=False, step="dev_local_mixed",
            detail=detail,
            stdout=out, stderr=err, ver_observed=ver_before, skipped_flash=True,
        )

    # Step 2: update each unique port (leader-only USB flashes all three roles)
    for port in ports:
        log.debug("Step 2: flashing %s (channel=%s)", port, firmware_channel)
        rc, out_u, err_u = _run_firmware(image_ref, ["update", firmware_channel, port])
        if rc != 0:
            return SmokeResult(ok=False, step="firmware_update", detail=f"exit {rc} ({port})", stdout=out_u, stderr=err_u)
        log.debug("Flashed %s successfully", port)

    # Step 3: re-show to get post-update versions (expect 3 roles)
    log.debug("Step 3: reading post-update versions")
    rc, out, err = _run_firmware(image_ref, ["show"])
    if rc != 0:
        return SmokeResult(ok=False, step="firmware_show", detail=f"exit {rc}", stdout=out, stderr=err)

    ver_observed = _parse_versions(out)
    log.debug("Versions observed: %s", ver_observed)
    if len(ver_observed) < BOARD_COUNT:
        return SmokeResult(
            ok=False, step="firmware_show",
            detail=f"expected {BOARD_COUNT} board roles, got {len(ver_observed)}",
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    if len(set(ver_observed)) != 1:
        return SmokeResult(
            ok=False, step="ver_mismatch",
            detail=f"boards disagree: {ver_observed}",
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    # Step 4: compare against S3 manifest
    log.debug("Step 4: fetching expected version from S3 (channel=%s)", firmware_channel)
    try:
        ver_expected = _fetch_expected_ver(firmware_channel)
    except Exception as exc:
        return SmokeResult(
            ok=False, step="s3_fetch",
            detail=str(exc),
            stdout=out, stderr=err, ver_observed=ver_observed,
        )

    log.debug("Version expected (S3): %s", ver_expected)
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
    """Extract unique serial port paths from `krabby-firmware show` output.

    Supports:
      - New: ``  front (/dev/ttyUSB0): 0.2.15 (…)`` (hub / leader-only USB)
      - Legacy: ``  /dev/ttyACM0  primary: 0.2.0 (…)``
    """
    paren = re.findall(r"\((/dev/tty[^)]+)\)", show_output)
    if paren:
        seen: list[str] = []
        for p in paren:
            if p not in seen:
                seen.append(p)
        return seen
    legacy = re.findall(r"^\s+(/dev/tty\S+)", show_output, re.MULTILINE)
    seen = []
    for p in legacy:
        if p not in seen:
            seen.append(p)
    return seen


def _parse_versions(show_output: str) -> list[str]:
    """Extract per-role version strings from `krabby-firmware show` output.

    Supports:
      - New: ``  front (/dev/ttyUSB0): 0.2.15 (release/0.2.15 abc)``
        or ``  left: 0.2.15 (…)`` when roles share the leader USB
      - Legacy: ``  /dev/ttyACM0  primary: 0.2.0 (mainline abc1234)``
      - Aggregated legacy: ``primary: 0.2.9 (…) | left: 0.2.9 (…)``
    """
    role_lines = re.findall(
        r"^\s+(?:front|left|right|primary|leader)\s*(?:\(/dev/tty[^)]+\))?:\s+(\S+)\s+\(",
        show_output,
        re.MULTILINE | re.IGNORECASE,
    )
    if role_lines:
        return role_lines
    # Legacy / pipe-aggregated numeric versions
    return re.findall(r":\s+(\d+\.\d+\.\d+)\s+\(", show_output)


def _fetch_expected_ver(channel: str) -> str:
    latest = requests.get(f"{S3_BASE}/{channel}/latest.json", timeout=10)
    latest.raise_for_status()
    manifest_url = latest.json()["manifest_url"]
    manifest = requests.get(manifest_url, timeout=10)
    manifest.raise_for_status()
    ver_string = manifest.json()["ver_string"]
    return ver_string.split()[0]

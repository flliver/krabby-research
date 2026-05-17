"""krabby-firmware --show / --update CLI commands."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

from firmware.krabby_mcu import parse_ver_reply
from firmware.manifest import FirmwareIndex, parse_index, latest_release_branch

BUCKET_BASE = "https://krabby-firmware-public.s3.amazonaws.com"
CACHE_DIR = Path.home() / ".cache" / "krabby-firmware"

_MEGA_USB_IDS = {
    ("2341", "0042"), ("2341", "0010"), ("2341", "0110"),  # Arduino Mega native USB
    ("1a86", "7523"), ("1a86", "5523"),                    # CH340 / CH341 (Krabby-Uno shield)
}


# --- port detection ---

def _all_mega_ports() -> list[str]:
    """Return device paths for all attached Arduino Mega 2560 boards."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    results = []
    for p in list_ports.comports():
        vid = f"{p.vid:04x}" if p.vid else ""
        pid = f"{p.pid:04x}" if p.pid else ""
        if (vid, pid) in _MEGA_USB_IDS:
            results.append(p.device)
            continue
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if any(k in desc or k in manuf for k in ("arduino", "dfrobot", "dfduino", "ch340", "cp210")):
            results.append(p.device)
    return results



def _probe_version(port: str, timeout: float = 6.0) -> Optional[str]:
    """Open port, wait for board to finish booting, send V, return VER line or None.

    Reads startup lines until "Krabby Ready" appears (board done booting), then
    sends V and returns the VER response. Avoids fixed sleeps that may be too short.
    """
    try:
        import serial
    except ImportError:
        return None
    try:
        with serial.Serial(port, 115200, timeout=0.2) as ser:
            ready = False
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    if ready:
                        ser.write(b"V\n")
                        ser.flush()
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if "Krabby Ready" in line:
                    ready = True
                    ser.write(b"V\n")
                    ser.flush()
                elif line.startswith("VER "):
                    return line
    except Exception:
        pass
    return None


# --- S3 fetch helpers ---

def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _fetch_index() -> FirmwareIndex:
    return parse_index(_fetch_json(f"{BUCKET_BASE}/index.json"))


# --- --show ---

def cmd_show() -> None:
    ports = _all_mega_ports()
    if ports:
        print("Attached boards:")
        for port in ports:
            ver_line = _probe_version(port)
            if boards := (parse_ver_reply(ver_line) if ver_line else None):
                roles = ("primary", "left", "right")
                parts = " | ".join(
                    f"{roles[i] if i < len(roles) else f'board{i}'}: {v} ({b} {c})"
                    for i, (v, b, c) in enumerate(boards) if v != "-"
                )
                print(f"  {port}  {parts}")
            else:
                print(f"  {port}  (no version response)")
    else:
        print("No attached Mega boards detected.")

    print()
    try:
        index = _fetch_index()
    except Exception as exc:
        print(f"Could not fetch S3 index: {exc}", file=sys.stderr)
        return

    if not index.branches:
        print("S3 bucket has no builds yet.")
        return

    print("Available S3 builds:")
    for name in sorted(index.branches):
        entry = index.branches[name]
        print(f"  {name:<30}  build {entry.build_key}")


# --- --update ---

def _is_port(s: str) -> bool:
    return s.startswith("/dev/") or s.startswith("COM") or s.upper().startswith("COM")


def _cached_hex(branch: str, commit: str, hex_filename: str) -> Path:
    return CACHE_DIR / branch / commit / hex_filename


def _download_hex(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def _flash(hex_path: Path, port: str) -> None:
    if shutil.which("avrdude"):
        cmd = ["avrdude", "-p", "m2560", "-c", "wiring", "-P", port,
               "-b", "115200", "-D", "-U", f"flash:w:{hex_path}:i"]
    elif shutil.which("arduino-cli"):
        cmd = ["arduino-cli", "upload", "--fqbn", "arduino:avr:mega",
               "--port", port, "--input-file", str(hex_path)]
    else:
        sys.exit("avrdude or arduino-cli required to flash. Run krabby-firmware --install first.")
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        sys.exit(f"Flash failed (exit {ret})")


def cmd_update(branch_or_port: Optional[str] = None, port_arg: Optional[str] = None) -> None:
    branch: Optional[str] = None
    port: Optional[str] = port_arg

    if branch_or_port is not None:
        if _is_port(branch_or_port):
            port = branch_or_port
        else:
            branch = branch_or_port

    try:
        index = _fetch_index()
    except Exception as exc:
        sys.exit(f"Could not fetch S3 index: {exc}")

    if branch is None:
        entry = latest_release_branch(index)
        if entry is None:
            sys.exit("No release/* branches found in S3 index. Use --update <branch> to specify one.")
        branch = entry.branch
    elif branch not in index.branches:
        sys.exit(f"Branch '{branch}' not found in S3 index. Available: {', '.join(sorted(index.branches))}")
    else:
        entry = index.branches[branch]

    print(f"Branch: {branch}  build: {entry.build_key}")

    hex_filename = "firmware.hex"
    commit = entry.build_key.rsplit("-", 1)[-1]
    cached = _cached_hex(branch, commit, hex_filename)

    if cached.exists():
        print(f"Using cached HEX: {cached}")
    else:
        print(f"Downloading {entry.hex_url} ...")
        _download_hex(entry.hex_url, cached)
        print(f"Saved to {cached}")

    if port is None:
        ports = _all_mega_ports()
        if not ports:
            sys.exit("No Mega boards detected. Connect a board or specify a port.")
        if len(ports) > 1:
            sys.exit(
                f"Multiple boards detected: {', '.join(ports)}. "
                "Specify a port with --update [branch] /dev/ttyACMx"
            )
        port = ports[0]

    print(f"Flashing {port} ...")
    _flash(cached, port)
    print("Flash complete.")

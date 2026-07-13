"""krabby-firmware show / update / set / get CLI commands."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from firmware.gui.remote import DEFAULT_SERIAL_DEV, start_bridge
from firmware.krabby_mcu import (
    _TELEMETRY_LINE_PREFIXES,
    ALL_JOINT_NAMES,
    KrabbyMCUSDK,
    build_get_line,
    build_set_line,
    parse_ver_reply,
)
from firmware.manifest import FirmwareIndex, parse_index, latest_release_branch
from firmware.mcu_port import MEGA_USB_IDS

BUCKET_BASE = "https://krabby-firmware-public.s3.amazonaws.com"
CACHE_DIR = Path.home() / ".cache" / "krabby-firmware"


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
        if (vid, pid) in MEGA_USB_IDS:
            results.append(p.device)
            continue
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if any(k in desc or k in manuf for k in ("arduino", "dfrobot", "dfduino", "ch340", "cp210")):
            results.append(p.device)
    return results



# Follower boards (ROLE_LEFT/RIGHT) respond to V on their UART uplink, not USB.
# After this many empty readline() timeouts post-V, give up rather than waiting
# the full timeout. Each readline timeout is 0.2 s → 8 × 0.2 s = 1.6 s cutoff.
_PROBE_V_RETRY_LIMIT = 8


def _probe_version(port: str, timeout: float = 6.0) -> tuple[Optional[str], Optional[str]]:
    """Open port, wait for boot, send V. Return (ver_line, role_hint). Either may be None.

    Captures the ROLE_HINT line the firmware prints at boot (from its EEPROM role)
    so the caller can label each board's port, including followers probed directly.
    """
    try:
        import serial
    except ImportError:
        return None, None
    try:
        # serial_for_url handles both local devices and the remote bridge's
        # socket://host:port URLs (plain paths fall through to normal Serial).
        with serial.serial_for_url(port, baudrate=115200, timeout=0.2) as ser:
            # Opening a local port toggles DTR and resets the board, so we wait for
            # its "Krabby Ready" banner before sending V. Over the TCP bridge the
            # banner is unreliable: the board reset when the *bridge* opened the
            # real port, so by the time we connect it may be mid-boot (banner still
            # coming) or long past it (banner gone, telemetry streaming). Writing V
            # blind is worse — during the boot window it pokes the bootloader, not
            # the sketch (and on pre-EEPROM-role firmware, V sent during SYNC
            # election was dropped outright). So for socket:// ports a telemetry
            # line also counts as ready (it proves the sketch's main loop is
            # running); boot-sequence chatter still waits for the banner.
            socket_port = port.startswith("socket://")
            ready = False
            role_hint: Optional[str] = None
            v_retries = 0
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    if ready:
                        if v_retries >= _PROBE_V_RETRY_LIMIT:
                            return None, role_hint
                        ser.write(b"V\n")
                        ser.flush()
                        v_retries += 1
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("ROLE_HINT: "):
                    role_hint = line[len("ROLE_HINT: "):].strip().lower()
                elif ("Krabby Ready" in line
                      or (socket_port and not ready
                          and line.startswith(_TELEMETRY_LINE_PREFIXES))):
                    ready = True
                    ser.write(b"V\n")
                    ser.flush()
                elif line.startswith("VER "):
                    return line, role_hint
    except Exception:
        pass
    return None, None


# --- S3 fetch helpers ---

def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _fetch_index() -> FirmwareIndex:
    return parse_index(_fetch_json(f"{BUCKET_BASE}/index.json"))


# --- --show ---

def cmd_show(remote: Optional[str] = None, remote_serial: str = DEFAULT_SERIAL_DEV) -> None:
    # The S3 index fetch is independent of the board probes and of the
    # multi-second ssh bridge startup — kick it off first so it overlaps both.
    index_executor = ThreadPoolExecutor(max_workers=1)
    index_future = index_executor.submit(_fetch_index)
    index_executor.shutdown(wait=False)

    # --remote: probe the single board behind the ssh/TCP bridge instead of
    # scanning local USB ports.
    bridge = None
    labels: dict[str, str] = {}
    if remote:
        bridge, bridged_port = start_bridge(remote, serial_dev=remote_serial)
        ports = [bridged_port]
        # Label the board by where it physically lives, not the tunnel URL.
        # Prefer the device the bridge actually opened — it globs a fallback
        # when the requested path is absent, so the request can be wrong.
        labels[bridged_port] = f"{remote}:{bridge.resolved_serial or remote_serial}"
    else:
        ports = _all_mega_ports()

    try:
        _show_status(ports, labels, index_future)
    finally:
        if bridge:
            bridge.stop()


def _show_status(ports: list[str], labels: dict[str, str], index_future) -> None:
    # Probe all boards in parallel (the S3 index fetch is already in flight).
    with ThreadPoolExecutor(max_workers=max(len(ports), 1)) as executor:
        probe_futures = [(port, executor.submit(_probe_version, port)) for port in ports]

    probe_results = {port: fut.result() for port, fut in probe_futures}

    if ports:
        # Leader returns combined VER (slot 0=front, 1=left, 2=right via UART).
        # Display role slots directly so old firmware without ROLE_HINT still shows
        # correct per-board versions instead of all mapping to slot 0.
        combined: list[tuple[str, str, str]] | None = None
        for port in ports:
            ver_line, _ = probe_results[port]
            if ver_line:
                parsed = parse_ver_reply(ver_line)
                if parsed and any(v != "-" for v, _, _ in parsed[1:]):
                    combined = parsed
                    break

        print("Attached boards:")
        if combined:
            # Annotate with port only when ROLE_HINT is available (firmware >= M14 step 9).
            role_to_port: dict[str, str] = {}
            for port in ports:
                _, role_hint = probe_results[port]
                if role_hint:
                    role_to_port.setdefault(role_hint, port)

            for role, slot in [("front", 0), ("left", 1), ("right", 2)]:
                v, b, c = combined[slot] if slot < len(combined) else ("-", "-", "-")
                p = role_to_port.get(role)
                port_label = f" ({labels.get(p, p)})" if p else ""
                print(f"  {role}{port_label}: {v} ({b} {c})")
        else:
            for port in ports:
                ver_line, role_hint = probe_results[port]
                role = role_hint or "front"
                parsed = parse_ver_reply(ver_line) if ver_line else None
                if parsed:
                    v, b, c = parsed[0]
                    print(f"  {labels.get(port, port)}  {role}: {v} ({b} {c})")
                else:
                    print(f"  {labels.get(port, port)}  {role}: (no version response)")
    else:
        print("No attached Mega boards detected.")

    print()
    try:
        index = index_future.result()
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


# --- set / get (board config) ---

def _parse_assignments(assignments: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a in assignments:
        key, sep, val = a.partition("=")
        if not sep or not key or not val:
            raise ValueError(f"expected key=value, got {a!r}")
        pairs.append((key, val))
    return pairs


def _open_config_sdk(port: Optional[str]) -> KrabbyMCUSDK:
    """Open a board for a quick config exchange: short settle, no hold-on-connect."""
    sdk = KrabbyMCUSDK(port=port)
    # Opening a local CH340 port resets the board — wait out its (election-free)
    # boot. Opening the socket:// bridge never touches the board: no wait needed.
    settle = 0.5 if "://" in sdk.port else 2.0
    if not sdk.connect(settle=settle, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")
    return sdk


def cmd_set(port: Optional[str], board: Optional[str], assignments: list[str]) -> None:
    # Validate client-side before touching the port (the SDK is the validation layer).
    try:
        pairs = _parse_assignments(assignments)
        build_set_line(board, pairs)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    keys = [k for k, _ in pairs]
    sdk = _open_config_sdk(port)
    try:
        sdk.send_set(board=board, **dict(pairs))
        result = sdk.send_get(*keys, board=board, timeout=2.0)  # best-effort read-back
    finally:
        sdk.close()

    label = f" ({board})" if board else ""
    if result:
        print(f"set{label}: " + "  ".join(f"{k}={result.get(k, '?')}" for k in keys))
    else:
        sent = " ".join(f"{k}={v}" for k, v in pairs)
        print(f"set{label}: sent {sent} (no read-back — run `get` to confirm)")


def cmd_get(port: Optional[str], board: Optional[str], keys: list[str]) -> None:
    try:
        build_get_line(board, keys)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    sdk = _open_config_sdk(port)
    try:
        result = sdk.send_get(*keys, board=board, timeout=2.0)
    finally:
        sdk.close()

    label = f" ({board})" if board else ""
    if result is None:
        sys.exit(f"get{label}: no response from board")
    print("  ".join(f"{k}={result.get(k, '?')}" for k in keys))


# --- calibrate-joint ---

def cmd_calibrate_joint(port: Optional[str], joint: str) -> None:
    joint = joint.upper()
    # Validate client-side before touching the port (opening it resets the board).
    if joint not in ALL_JOINT_NAMES:
        sys.exit(f"error: unknown joint {joint!r}; expected one of {', '.join(sorted(ALL_JOINT_NAMES))}")

    sdk = _open_config_sdk(port)
    try:
        print(f"calibrating {joint} — it will sweep to both stops "
              "(takes seconds; telemetry pauses)...")
        result = sdk.calibrate_joint(joint)
    finally:
        sdk.close()

    if result is None:
        sys.exit(f"calibrate {joint.upper()}: no reply from board")
    if not result["ok"]:
        sys.exit(f"calibrate {result['joint']}: FAILED ({result['why']})")
    print(f"calibrated {result['joint']}: min={result['min']} max={result['max']} (saved to EEPROM)")


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
        sys.exit("avrdude or arduino-cli required to flash. Run: krabby-firmware install")
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        raise RuntimeError(f"flash failed on {port} (exit {ret})")


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
            sys.exit("No release/* branches found in S3 index. Use: update <branch>")
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

    if port is not None:
        ports = [port]
    else:
        ports = _all_mega_ports()
        if not ports:
            sys.exit("No Mega boards detected. Connect a board or specify a port.")

    failed = []
    for p in ports:
        print(f"Flashing {p} ...")
        try:
            _flash(cached, p)
            print(f"  done")
        except RuntimeError as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            failed.append(p)

    if failed:
        sys.exit(f"Flash failed on: {', '.join(failed)}")
    print(f"Flashed {len(ports)} board(s).")

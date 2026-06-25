"""krabby-firmware show / update CLI commands."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from firmware.krabby_mcu import (
    ALL_JOINT_NAMES,
    KrabbyMCUSDK,
    build_get_line,
    build_set_line,
    parse_ver_reply,
)
from firmware.mcu_port import MEGA_USB_IDS
from firmware.manifest import (
    BranchBuild,
    FirmwareIndex,
    latest_release_branch,
    parse_builds,
    parse_index,
)

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

    Captures the ROLE_HINT line the board prints at boot so the caller can label
    follower boards correctly even when probed alone.
    """
    try:
        import serial
    except ImportError:
        return None, None
    try:
        with serial.Serial(port, 115200, timeout=0.2) as ser:
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
                elif "Krabby Ready" in line:
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


def _fetch_builds(branch: str) -> list[BranchBuild]:
    return parse_builds(_fetch_json(f"{BUCKET_BASE}/{branch}/builds.json"))


def _page(text: str) -> None:
    """Write text, routing through a pager when stdout is an interactive TTY.

    Falls back to plain stdout if the pager can't be launched (OSError) or exits
    non-zero — e.g. `less`/$PAGER missing makes the shell exit 127 — so the build
    list is never silently swallowed (the locomotion image has no `less`).
    """
    if not text.endswith("\n"):
        text += "\n"
    if sys.stdout.isatty():
        pager = os.environ.get("PAGER") or "less -FRX"
        try:
            proc = subprocess.Popen(pager, shell=True, stdin=subprocess.PIPE, text=True)
            proc.communicate(text)
            if proc.returncode == 0:
                return
        except (OSError, BrokenPipeError):
            pass
    sys.stdout.write(text)


# --- show ---

def cmd_show(branch: Optional[str] = None) -> None:
    # `show <branch>` lists that branch's full build history, newest-first and paged.
    if branch is not None:
        _show_branch_builds(branch)
        return

    ports = _all_mega_ports()

    # Probe all boards and fetch S3 index in parallel.
    with ThreadPoolExecutor(max_workers=len(ports) + 1) as executor:
        index_future = executor.submit(_fetch_index)
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
                port_label = f" ({role_to_port[role]})" if role in role_to_port else ""
                print(f"  {role}{port_label}: {v} ({b} {c})")
        else:
            for port in ports:
                ver_line, role_hint = probe_results[port]
                role = role_hint or "front"
                parsed = parse_ver_reply(ver_line) if ver_line else None
                if parsed:
                    v, b, c = parsed[0]
                    print(f"  {port}  {role}: {v} ({b} {c})")
                else:
                    print(f"  {port}  {role}: (no version response)")
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

    print("Available S3 builds (latest per branch):")
    for name in sorted(index.branches):
        entry = index.branches[name]
        print(f"  {name:<30}  build {entry.build_key}")
    print("\nRun `krabby-firmware show <branch>` to list a branch's builds newest-first.")


def _show_branch_builds(branch: str) -> None:
    """List every build for one branch, newest-first, through a pager."""
    try:
        builds = _fetch_builds(branch)
    except urllib.error.HTTPError as exc:
        # The public bucket grants s3:GetObject but not s3:ListBucket, so a GET of an
        # absent key returns 403 (Access Denied), not 404 — treat both as "no history".
        if exc.code in (403, 404):
            sys.exit(f"No build history for branch '{branch}'. Run `krabby-firmware show` to list branches.")
        sys.exit(f"Could not fetch builds for '{branch}': {exc}")
    except Exception as exc:
        sys.exit(f"Could not fetch builds for '{branch}': {exc}")

    if not builds:
        print(f"No builds found for branch '{branch}'.")
        return

    lines = [f"Builds for {branch} (newest first, {len(builds)} total):", ""]
    for b in builds:
        ver = b.ver_string or "?"
        date = b.commit_date or "?"
        lines.append(f"  {b.build_key:<26}  {date:<12}  {ver}")
    _page("\n".join(lines) + "\n")


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


# --- set / get (board config: role, serial) ---

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
    if not sdk.connect(settle=2.0, hold=False):
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


def cmd_calibrate_joint(port: Optional[str], name: str) -> None:
    # Validate client-side before opening the port (the SDK is the validation layer).
    if name not in ALL_JOINT_NAMES:
        sys.exit(f"error: unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")

    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=5.0, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")

    # Cal is fire-and-forget on the wire; hold the connection open briefly to surface
    # any ERR <joint> <code> the firmware emits (e.g. motor_did_not_move). The full
    # sweep runs in firmware regardless of whether we stay connected.
    wait = 10.0
    seen: set = set()
    cal = None
    try:
        sdk.clear_errors()
        sdk.calibrate_joint(name)
        print(f"calibrate-joint: {name} — sweeping to both stops (~{wait:.0f}s); watching for errors…")
        deadline = time.time() + wait
        while time.time() < deadline:
            for e in sdk.get_errors():
                key = (e.token, e.code)
                if key not in seen:
                    seen.add(key)
                    print(f"  ERR {e.token} {e.code}")
            time.sleep(0.2)
        cal = sdk.get_calibration(name, timeout=2.0)  # read back what landed in EEPROM
    finally:
        sdk.close()

    print(_format_cal(name, cal))
    if seen:
        print(f"calibrate-joint: {name} reported the error(s) above — not trusted.")


def _format_cal(name: str, cal: Optional[dict]) -> str:
    """One-line summary of a joint's stored calibration (the CAL read-back dict)."""
    if cal is None:
        return f"{name}: no calibration read-back (no response)"
    span = ""
    try:
        span = f"  span={int(cal['max']) - int(cal['min'])}"
    except (KeyError, ValueError):
        pass
    trusted = cal.get("cal") == "1"
    flag = "" if trusted else "   ⚠ NOT TRUSTED (sweep range too small / sensor not tracking)"
    return (f"{name}: type={cal.get('type','?')} reversed={cal.get('rev','?')} "
            f"min={cal.get('min','?')} max={cal.get('max','?')}{span}  calibrated={cal.get('cal','?')}{flag}")


def cmd_get_calibration(port: Optional[str], name: str) -> None:
    if name not in ALL_JOINT_NAMES:
        sys.exit(f"error: unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")
    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=2.0, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")
    try:
        cal = sdk.get_calibration(name, timeout=2.0)
    finally:
        sdk.close()
    if cal is None:
        sys.exit(f"get-calibration: no response for {name}")
    print(_format_cal(name, cal))

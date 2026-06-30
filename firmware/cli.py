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
    JOINT_GROUP_NAMES,
    KrabbyMCUSDK,
    build_get_line,
    build_set_line,
    parse_cal_reply,
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


def _probe_version(
    port: str, timeout: float = 6.0, cal_joints: Optional[list[str]] = None
) -> tuple[Optional[str], Optional[str], dict[str, bool]]:
    """Open port, wait for boot, send V. Return (ver_line, role_hint, cals). ver_line/
    role_hint may be None; cals maps joint -> stored EEPROM `calibrated` flag (bool).

    Captures the ROLE_HINT line the board prints at boot so the caller can label
    follower boards correctly even when probed alone. When cal_joints is given, after the
    VER reply it issues `Q<joint>` for each (the firmware forwards these to followers and
    relays their replies up) and collects the persisted `calibrated` flag for each that
    answers. cals only contains joints that replied — callers treat absent ones as unknown.
    """
    cals: dict[str, bool] = {}
    try:
        import serial
    except ImportError:
        return None, None, cals
    want = list(cal_joints or [])
    try:
        with serial.Serial(port, 115200, timeout=0.2) as ser:
            ready = False
            role_hint: Optional[str] = None
            ver_line: Optional[str] = None
            v_retries = 0
            cal_deadline: Optional[float] = None  # set once Qs are sent; bounds the cal wait
            deadline = time.time() + timeout
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    if ready and ver_line is None:
                        if v_retries >= _PROBE_V_RETRY_LIMIT:
                            return None, role_hint, cals
                        ser.write(b"V\n")
                        ser.flush()
                        v_retries += 1
                    if cal_deadline is not None and time.time() > cal_deadline:
                        break
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if line.startswith("ROLE_HINT: "):
                    role_hint = line[len("ROLE_HINT: "):].strip().lower()
                elif "Krabby Ready" in line:
                    ready = True
                    ser.write(b"V\n")
                    ser.flush()
                elif line.startswith("VER "):
                    ver_line = line
                    if not want:
                        return ver_line, role_hint, cals
                    for j in want:
                        ser.write(f"Q{j}\n".encode())
                    ser.flush()
                    cal_deadline = time.time() + 3.0
                else:
                    parsed = parse_cal_reply(line)
                    if parsed:
                        name, kv = parsed
                        if name in want and "cal" in kv:
                            cals[name] = kv["cal"] == "1"
                            if len(cals) >= len(want):
                                break
                if cal_deadline is not None and time.time() > cal_deadline:
                    break
            return ver_line, role_hint, cals
    except Exception:
        pass
    return None, None, cals


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

# Display layout for the calibration matrix: one row per leg, one column per joint.
# A joint name is <leg-prefix><joint-suffix>, e.g. FL + HL = FLHL.
_CAL_LEGS = (
    ("Front Left", "FL"), ("Front Right", "FR"),
    ("Middle Left", "ML"), ("Middle Right", "MR"),
    ("Rear Left", "RL"), ("Rear Right", "RR"),
)
_CAL_COLS = (("Hip", "HL"), ("Knee", "KL"), ("Yaw", "HY"))


def _cal_cell(joint: str, board_cals: dict[str, bool]) -> str:
    """One cell of the calibration matrix, by the persisted EEPROM `calibrated` flag:
    ✓ = calibrated, ✗ = not calibrated, N/A = yaw (no calibratable end-stops here),
    ? = no reply (board unreachable / joint not wired / firmware without cal read-back)."""
    if joint[3] == "Y":
        return "N/A"
    if joint not in board_cals:
        return "?"
    return "✓" if board_cals[joint] else "✗"


def _print_cal_status(board_cals: dict[str, bool]) -> None:
    """Print joint calibration as a leg × joint grid — the persisted EEPROM `calibrated`
    flag, not the runtime state (which resets to PARTIAL on a Hall every boot)."""
    label_w = max(len(name) for name, _ in _CAL_LEGS)
    col_w = 7
    print()
    print("Joint calibration:")
    header = " " * (label_w + 4) + "".join(f"{title:<{col_w}}" for title, _ in _CAL_COLS)
    print(header.rstrip())
    for name, leg in _CAL_LEGS:
        cells = "".join(f"{_cal_cell(leg + suf, board_cals):<{col_w}}" for _, suf in _CAL_COLS)
        print(f"  {name:<{label_w}}  {cells}".rstrip())
    print("  ✓ calibrated   ✗ not calibrated   N/A no end-stops   ? no reply")


def cmd_show(branch: Optional[str] = None) -> None:
    # `show <branch>` lists that branch's full build history, newest-first and paged.
    if branch is not None:
        _show_branch_builds(branch)
        return

    ports = _all_mega_ports()

    # Yaw joints (4th char 'Y') have no calibratable end-stops here, so we don't query them.
    cal_joints = [n for _, names in JOINT_GROUP_NAMES for n in names if n[3] != "Y"]

    # Probe all boards and fetch S3 index in parallel. Each probe also collects the stored
    # `calibrated` flag for the non-yaw joints (Q forwards to followers; replies relay up).
    with ThreadPoolExecutor(max_workers=len(ports) + 1) as executor:
        index_future = executor.submit(_fetch_index)
        probe_futures = [
            (port, executor.submit(_probe_version, port, cal_joints=cal_joints))
            for port in ports
        ]

    probe_results = {port: fut.result() for port, fut in probe_futures}

    # Union the per-port cal replies — a board answers its own joints, and a leader relays
    # its followers', so across all ports we collect every joint that responded.
    board_cals: dict[str, bool] = {}
    for _, _, cals in probe_results.values():
        board_cals.update(cals)

    if ports:
        # Leader returns combined VER (slot 0=front, 1=left, 2=right via UART).
        # Display role slots directly so old firmware without ROLE_HINT still shows
        # correct per-board versions instead of all mapping to slot 0.
        combined: list[tuple[str, str, str]] | None = None
        for port in ports:
            ver_line, _, _ = probe_results[port]
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
                _, role_hint, _ = probe_results[port]
                if role_hint:
                    role_to_port.setdefault(role_hint, port)

            for role, slot in [("front", 0), ("left", 1), ("right", 2)]:
                v, b, c = combined[slot] if slot < len(combined) else ("-", "-", "-")
                port_label = f" ({role_to_port[role]})" if role in role_to_port else ""
                print(f"  {role}{port_label}: {v} ({b} {c})")
        else:
            for port in ports:
                ver_line, role_hint, _ = probe_results[port]
                role = role_hint or "front"
                parsed = parse_ver_reply(ver_line) if ver_line else None
                if parsed:
                    v, b, c = parsed[0]
                    print(f"  {port}  {role}: {v} ({b} {c})")
                else:
                    print(f"  {port}  {role}: (no version response)")
    else:
        print("No attached Mega boards detected.")

    if ports:
        _print_cal_status(board_cals)

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
    # `get calibration` is the whole-board cal dump (Task 4 §4), a different reply shape
    # (one CAL line per joint) than the single-line role/serial/version GETs.
    if keys == ["calibration"]:
        _cmd_get_calibration_board(port, board)
        return
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


def _cmd_get_calibration_board(port: Optional[str], board: Optional[str]) -> None:
    """Pretty-print one board's stored calibration (every joint), via `GET calibration`."""
    sdk = _open_config_sdk(port)
    try:
        cals = sdk.get_all_calibration(board=board, timeout=2.5)
    finally:
        sdk.close()
    label = board or "front"
    if not cals:
        sys.exit(f"get calibration ({label}): no response from board")
    print(f"Calibration ({label}):")
    for joint in sorted(cals):
        c = cals[joint]
        span = abs(int(c["max"]) - int(c["min"]))
        print(f"  {joint}  {c['type']:<4} min={c['min']:>6} max={c['max']:>6} "
              f"span={span:<5} reversed={c['reversed']}")


def _watch_until_idle(sdk, label: str, timeout: float) -> list:
    """Stream ERR lines as the firmware emits them and return once the joints go idle after
    running (no completion line on the wire, spec §3h). Returns the ErrorEvents collected."""
    seen: set = set()
    errs: list = []
    deadline = time.time() + timeout
    started, idle_since = False, None
    while time.time() < deadline:
        for e in sdk.get_errors():
            key = (e.token, e.code)
            if key not in seen:
                seen.add(key)
                errs.append(e)
                print(f"  ERR {e.token} {e.code}")
        active = any(jt is not None and jt.pwm != (0, 0) for jt in sdk.joints.values())
        if active:
            started, idle_since = True, None
        elif started:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > 4.0:
                break
        time.sleep(0.3)
    return errs


def _read_cal_grid(sdk) -> dict:
    cals: dict = {}
    for j in sorted(ALL_JOINT_NAMES):
        if j[3] == "Y":
            continue
        c = sdk.get_calibration(j, timeout=0.8)
        if c is not None and "cal" in c:
            cals[j] = c["cal"] == "1"
    return cals


def _print_failures(sdk, errs: list) -> None:
    for instruction in sdk.explain_failures(errs):
        print(f"  → {instruction}")


def cmd_calibrate_all(port: Optional[str], no_yaw: bool = False,
                      skip_validation: bool = False, timeout: float = 300.0) -> None:
    """Run the whole-board calibration sequence + (unless --skip-validation) the Task-4
    neutral pose and current-sense lift validation. Streams ERR lines, then prints the cal
    grid and operator-facing fix instructions for any failures."""
    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=5.0, hold=True):
        sys.exit(f"could not open serial port {sdk.port}")
    try:
        sdk.clear_errors()
        sdk.calibrate_all(include_yaw=not no_yaw, validate=not skip_validation)
        extras = (" (no yaw)" if no_yaw else "") + (" (skip validation)" if skip_validation else "")
        print(f"calibrate-all: running whole-board sequence{extras} "
              f"— watching for errors (up to ~{timeout:.0f}s)…")
        errs = _watch_until_idle(sdk, "calibrate-all", timeout)
        cals = _read_cal_grid(sdk)
    finally:
        sdk.close()

    _print_cal_status(cals)
    if errs:
        print("calibrate-all: failures reported —")
        _print_failures(sdk, errs)


def cmd_validate_current_sense(port: Optional[str], timeout: float = 120.0) -> None:
    """Run the current-sense validation standalone (Task 4 §6): neutral pose, then per-leg
    body lift. Assumes per-joint cal already ran. Chassis-required — on a bench the lifts
    can't bear weight, so expect current_sense_no_signal/no_spike."""
    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=5.0, hold=True):
        sys.exit(f"could not open serial port {sdk.port}")
    try:
        sdk.clear_errors()
        sdk.validate_current_sense()
        print(f"validate-current-sense: lifting each leg — watching for errors (up to ~{timeout:.0f}s)…")
        errs = _watch_until_idle(sdk, "validate-current-sense", timeout)
    finally:
        sdk.close()

    if errs:
        print("validate-current-sense: failures reported —")
        _print_failures(sdk, errs)
    else:
        print("validate-current-sense: all legs produced a current spike.")


def cmd_calibrate_joint(port: Optional[str], name: str, direction: Optional[str] = None) -> None:
    # Validate client-side before opening the port (the SDK is the validation layer).
    if name not in ALL_JOINT_NAMES:
        sys.exit(f"error: unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")
    try:
        direction = KrabbyMCUSDK._validate_cal_direction(name, direction)
    except ValueError as e:
        sys.exit(f"error: {e}")

    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=5.0, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")

    # Cal is fire-and-forget on the wire — there's no completion message, so we poll the
    # stored calibration. A FULL sweep is retract→extend→retract (triple stroke for the 2c
    # drift check) and resolves to FULL (ok) / UNCAL (range fail). A directional cal drives
    # ONE end-stop (~one stroke) and on a fresh joint leaves the other end unrecorded, so it
    # won't reach FULL — for that case we just wait out the stroke and read back whatever
    # landed. ~8 s/stroke on this slow actuator, so give it generous room either way.
    timeout = 45.0 if direction is None else 20.0
    sweep_desc = "sweeping both stops" if direction is None else f"driving the {direction} stop"
    seen: set = set()
    cal = None
    try:
        sdk.clear_errors()
        sdk.calibrate_joint(name, direction)
        print(f"calibrate-joint: {name} — {sweep_desc} (slow actuator; up to ~{timeout:.0f}s)…")
        deadline = time.time() + timeout
        while time.time() < deadline:
            for e in sdk.get_errors():
                key = (e.token, e.code)
                if key not in seen:
                    seen.add(key)
                    print(f"  ERR {e.token} {e.code}")
            cal = sdk.get_calibration(name, timeout=1.0)
            # A full sweep is done when it reaches FULL (success) or an ERR fires (every
            # failure path emits one). UNCAL is NOT terminal — it's also the *start* state
            # of a never-calibrated joint, so breaking on it returns the stale pre-cal read
            # while the sweep is still running. A directional cal has no FULL endpoint (one
            # end only), so it just polls for ERR until the stroke timeout, then reads back.
            if direction is None and ((cal and cal.get("state") == "FULL") or seen):
                break
            if direction is not None and seen:
                break
            time.sleep(0.5)
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
        # absolute travel: a reversed sensor (e.g. a pot that reads high→low as it extends)
        # stores max < min, so the raw difference goes negative — the magnitude is the span.
        span = f"  span={abs(int(cal['max']) - int(cal['min']))}"
    except (KeyError, ValueError):
        pass
    trusted = cal.get("cal") == "1"
    flag = "" if trusted else "   ⚠ NOT TRUSTED (sweep range too small / sensor not tracking)"
    state = cal.get("state")
    state_str = f" state={state}" if state else ""
    if state == "PARTIAL":
        flag += "   (Hall: jog to an end-stop to anchor it)"
    return (f"{name}: type={cal.get('type','?')} reversed={cal.get('rev','?')} "
            f"min={cal.get('min','?')} max={cal.get('max','?')}{span}  "
            f"calibrated={cal.get('cal','?')}{state_str}{flag}")


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


def cmd_jog(port: Optional[str], name: str, pwm: int, ms: int) -> None:
    """Drive one joint open-loop at `pwm` for `ms` milliseconds, re-sending at a 100 ms
    heartbeat (< the firmware's 300 ms jog watchdog) so it runs the full duration, then
    stop. Reports the joint's normalized position and calibration state afterward —
    useful for driving a Hall joint into an end-stop to exercise the self-heal."""
    if name not in ALL_JOINT_NAMES:
        sys.exit(f"error: unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")
    pwm = max(-255, min(255, int(pwm)))
    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=5.0, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")
    cal = None
    pos = "?"
    try:
        deadline = time.time() + ms / 1000.0
        next_beat = 0.0
        while time.time() < deadline:
            now = time.time()
            if now >= next_beat:
                sdk.send_command_jog(name, pwm)
                next_beat = now + 0.1
            time.sleep(0.01)
        sdk.send_command_jog(name, 0)
        time.sleep(0.25)
        jt = sdk.joints.get(name)
        if jt:
            pos = f"{jt.pos:.3f}"
        cal = sdk.get_calibration(name, timeout=1.0)
    finally:
        sdk.send_command_jog(name, 0)
        sdk.close()
    print(f"jog {name}: pwm={pwm} ms={ms} -> pos={pos}")
    if cal:
        print("  " + _format_cal(name, cal))


def cmd_observe(port: Optional[str], hz: float = 0.0, count: int = 1) -> None:
    """Dump the assembled model observation built from live MCU telemetry — the
    HAL-layer view (normalized joint position [0,1], Python-side velocity EMA, and
    the current→contact_forces mapping), not just raw board telemetry. Bench tool
    for verifying M17 Task 6 (6b/6c/6e) without running the inference stack.

    ``count <= 0`` streams until Ctrl-C. Velocity needs ≥2 snapshots to read
    non-zero (the first tick for a joint has no prior sample), so a single-shot
    call shows vel 0 — pass --hz to stream."""
    from firmware.observation_mapping import (
        CONTACT_DROPPED_LEG,
        CONTACT_LEGS,
        JointVelocityEstimator,
        contact_forces_from_leg_currents,
        leg_prefix,
    )

    sdk = KrabbyMCUSDK(port=port)
    if not sdk.connect(settle=2.0, hold=False):
        sys.exit(f"could not open serial port {sdk.port}")

    names = [n for _, group in JOINT_GROUP_NAMES for n in group]
    est = JointVelocityEstimator()
    period = 1.0 / hz if hz and hz > 0 else 0.0

    # The reader thread starts after connect(); give the board a moment to stream
    # its first telemetry frame so the first snapshot isn't all dashes.
    deadline = time.time() + 3.0
    while not sdk.joints and time.time() < deadline:
        time.sleep(0.05)

    try:
        n_done = 0
        while count <= 0 or n_done < count:
            now = time.time()
            snap = dict(sdk.joints)
            leg_currents: dict[str, float] = {}
            print(f"=== observation @ {now:.2f}s ===")
            print(f"{'joint':<6} {'pos[0,1]':>8} {'vel':>7} {'cur':>5}  cal")
            for name in names:
                jt = snap.get(name)
                if jt is None:
                    print(f"{name:<6} {'-':>8} {'-':>7} {'-':>5}  -")
                    continue
                vel = est.update(name, jt.pos, now)
                leg = leg_prefix(name)
                leg_currents[leg] = leg_currents.get(leg, 0.0) + float(jt.current)
                print(f"{name:<6} {jt.pos:>8.3f} {vel:>+7.3f} {jt.current:>5}  {jt.cal_state_name}")
            contacts = contact_forces_from_leg_currents(leg_currents)
            contact_str = ", ".join(f"{leg} {v:+.2f}" for leg, v in zip(CONTACT_LEGS, contacts))
            print(f"contact_forces: [{contact_str}]   ({CONTACT_DROPPED_LEG} dropped)")
            n_done += 1
            if (count <= 0 or n_done < count) and period:
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        sdk.close()

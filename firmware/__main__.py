"""
Interactive MCU menu. Run with: python -m firmware [--debug]
Works over SSH / headless — uses termios + select, no X11 required.
"""
import argparse
import select
import sys
import tty
import termios
import logging
import time
from typing import NoReturn

from firmware.krabby_mcu import BOARDS, JOINT_GROUP_NAMES, KrabbyMCUSDK, parse_ver_reply, logger

# Joint order per leg pair: LKL, LHL, LHY, RHY, RHL, RKL
JOINTS_FRONT = ["FLKL", "FLHL", "FLHY", "FRHY", "FRHL", "FRKL"]
JOINTS_LEFT = ["RLKL", "RLHL", "RLHY", "MLHY", "MLHL", "MLKL"]
JOINTS_RIGHT = ["RRKL", "RRHL", "RRHY", "MRHY", "MRHL", "MRKL"]
# Extend: Q W E R T Y  |  Retract: A S D F G H
EXTEND_KEYS = ["q", "w", "e", "r", "t", "y"]
RETRACT_KEYS = ["a", "s", "d", "f", "g", "h"]
# analogWrite is 0-255 duty; 200 is ~78% -> ~18.8 V average from a 24 V rail. 255 is ~100% duty.
JOG_PWM = 255

# Tracks the last time each key was seen; is_pressed() uses a 150ms recency
# window so auto-repeat (≈33ms) keeps keys "held" while release lets them expire.
_last_seen: dict[str, float] = {}
_quit = False
_HOLD_WINDOW = 0.15  # seconds
_BOARD_ROLES = ("front  ", "left   ", "right  ")  # padded for log column alignment


def _read_keys(fd: int) -> None:
    """Drain all bytes currently in stdin and refresh _last_seen timestamps."""
    global _quit
    while select.select([fd], [], [], 0)[0]:
        ch = sys.stdin.buffer.read(1)
        if not ch:
            break
        if ch == b"\x1b":
            _quit = True
        else:
            c = ch.decode("utf-8", errors="ignore").lower()
            if c:
                _last_seen[c] = time.monotonic()


def is_pressed(k: str) -> bool:
    return time.monotonic() - _last_seen.get(k, 0.0) < _HOLD_WINDOW


# --- Live telemetry table (headless bench readout) ------------------------
# The SDK's reader thread keeps mcu.joints up to date; we just render it in
# place so an operator on the headless Jetson can jog a motor and watch its
# pot / current / Hall move in the same SSH screen (AC 1b).
_TELEM_HEADER = "        JOINT   POS   POT   CUR   HALL  CAL"


def _fmt_row(board_lbl: str, sel: str, name: str, jt) -> str:
    head = f"{board_lbl:6s}{sel}"
    if jt is None:
        return f"{head} {name:5s}  ----    ---   ---   ---   ----"
    # STATE flags whether POS is trustworthy: PARTIAL = Hall not yet anchored (pos is
    # relative — jog to an end-stop to self-heal), FULL = absolute, UNCAL = no cal.
    return (f"{head} {name:5s} {jt.pos:6.3f}  {jt.pot:4d}  {jt.current:4d}  {jt.saf:5d}  "
            f"{jt.cal_state_name}")


def _render_telemetry(joints, selected: str, status: str = "", err_line: str = "") -> list[str]:
    """Build the live-telemetry frame as a list of lines (pure → unit-testable)."""
    lines = [
        "=== Krabby MCU — jog + live telemetry (18 joints) ===",
        "Extend: Q W E R T Y   Retract: A S D F G H",
        f"Board: {selected:5s}   hold 1=LEFT  2=RIGHT  1+2=all  none=FRONT",
        "0=neutral  V=version  ESC=quit",
        status or " ",
        err_line or " ",
        "",
        _TELEM_HEADER,
    ]
    for board, names in JOINT_GROUP_NAMES:
        for i, name in enumerate(names):
            board_lbl = board if i == 0 else ""
            sel = ">" if selected in (board, "ALL") else " "
            lines.append(_fmt_row(board_lbl, sel, name, joints.get(name)))
        lines.append("")
    return lines


def _draw(lines: list[str]) -> None:
    """Repaint the frame in place: home the cursor, clear each line to EOL, then
    clear anything below so a shorter frame leaves no stale rows."""
    sys.stdout.write("\033[H" + "".join(f"{ln}\033[K\n" for ln in lines) + "\033[J")
    sys.stdout.flush()


def _err_line(mcu) -> str:
    if errs := mcu.get_errors():
        e = errs[-1]
        return f"last ERR: {e.token} {e.code}"
    return "last ERR: (none)"


def _selected_board(key1: bool, key2: bool) -> str:
    if key1 and key2:
        return "ALL"
    if key1:
        return "LEFT"
    if key2:
        return "RIGHT"
    return "FRONT"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(f"krabby-firmware: {message}\n", file=sys.stderr)
        self.print_help(sys.stderr)
        sys.exit(2)


# One row per subcommand: (usage, description). Required args are shown bare
# (e.g. JOINT, KEY=VAL...), optional ones in [brackets]. Kept here — rather than
# leaning on argparse's terse auto-help — so `help` can show each command's
# required arguments inline. Run `krabby-firmware <cmd> --help` for full flags.
_COMMAND_HELP: list[tuple[str, str]] = [
    ("help",
     "Show this command summary and exit."),
    ("install",
     "Set up host udev rules and serial permissions (one-time host setup)."),
    ("show [BRANCH]",
     "List attached boards and firmware versions; with BRANCH, list that "
     "branch's builds newest-first."),
    ("update [CHANNEL] [PORT]",
     "Flash firmware from an S3 channel to attached board(s). Defaults to the "
     "latest release channel and all detected boards."),
    ("set [--port P] [--board B] KEY=VAL [KEY=VAL ...]",
     "Write board config (role, serial) to EEPROM. Requires one or more "
     "KEY=VAL, e.g. role=FRONT serial=FRT-0042."),
    ("get [--port P] [--board B] KEY [KEY ...]",
     "Read board config from a board. Requires one or more KEY, "
     "e.g. role serial version."),
    ("calibrate-joint [--port P] [--direction D] JOINT",
     "Calibrate one joint: sweep end-stop(s), auto-detect sensor + direction, "
     "persist to EEPROM. Requires JOINT (e.g. FLHL)."),
    ("calibrate-all [--port P] [--no-yaw] [--skip-validation]",
     "Run the whole-board cal sequence + current-sense validation (M17 Task 3/4). "
     "--no-yaw skips hip-yaw; --skip-validation skips the chassis-required lifts."),
    ("validate-current-sense [--port P]",
     "Run the per-leg current-sense lift validation standalone (assumes cal done, M17 Task 4)."),
    ("get-calibration [--port P] JOINT",
     "Read back a joint's stored calibration (sensor type, min/max, flag). "
     "Requires JOINT."),
    ("jog [--port P] [--ms MS] --joint JOINT --pwm -255..255",
     "Drive one joint open-loop for a bounded time, then stop. "
     "Requires --joint and --pwm."),
    ("observe [--port P] [--hz HZ] [--count N]",
     "Dump the assembled model observation (normalized pos, velocity, "
     "contact_forces) from live telemetry. --hz streams (M17 Task 6 bench check)."),
]


def _command_help() -> str:
    """Curated `krabby-firmware help` summary: each subcommand, its required
    arguments, and a one-line description (pure → unit-testable)."""
    lines = [
        "krabby-firmware — Krabby firmware tools",
        "",
        "Usage: krabby-firmware [--debug] [<command> ...]",
        "",
        "With no command, launches the interactive MCU jog + live-telemetry menu.",
        "--debug enables debug logging (interactive menu only).",
        "",
        "Commands:",
    ]
    for usage, desc in _COMMAND_HELP:
        lines.append(f"  {usage}")
        lines.append(f"      {desc}")
    lines.append("")
    lines.append("Run `krabby-firmware <command> --help` for a command's full options.")
    return "\n".join(lines)


def main():
    parser = _Parser(
        prog="krabby-firmware",
        description="Krabby firmware tools. With no subcommand, launches the interactive MCU key-control menu.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (interactive menu only)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("help", help="Show this help message and exit.")
    subparsers.add_parser("install", help="Set up host udev rules and serial permissions.")
    show_p = subparsers.add_parser(
        "show",
        help="List attached boards and firmware versions; `show <branch>` lists that branch's builds.",
    )
    show_p.add_argument(
        "branch", nargs="?", default=None, metavar="BRANCH",
        help="Optional branch (e.g. release/0.2.9) — list all its builds newest-first, paged.",
    )
    update_p = subparsers.add_parser("update", help="Flash firmware from S3 channel to board(s).")
    update_p.add_argument("channel", nargs="?", default=None, metavar="CHANNEL")
    update_p.add_argument("port", nargs="?", default=None, metavar="PORT")

    set_p = subparsers.add_parser(
        "set", help="Write board config (role, serial) to EEPROM. Fire-and-forget.")
    set_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    set_p.add_argument("--board", default=None, choices=BOARDS,
                       help="Target board (default: the board on --port). left/right forward via the front board.")
    set_p.add_argument("assignments", nargs="+", metavar="KEY=VAL",
                       help="One or more key=value, e.g. role=FRONT serial=FRT-0042.")

    get_p = subparsers.add_parser("get", help="Read board config (role, serial, version) from a board.")
    get_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    get_p.add_argument("--board", default=None, choices=BOARDS,
                       help="Target board (default: the board on --port).")
    get_p.add_argument("keys", nargs="+", metavar="KEY",
                       help="One or more keys to read, e.g. role serial version.")

    cal_p = subparsers.add_parser(
        "calibrate-joint",
        help="Calibrate one joint: sweep both end-stops, auto-detect sensor + direction, persist to EEPROM.")
    cal_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    cal_p.add_argument("name", metavar="JOINT",
                       help="Joint to calibrate, e.g. FLHL. Forwarded to followers via the FRONT board.")
    cal_p.add_argument("--direction", default=None,
                       choices=["extend", "retract", "left", "right"],
                       help="Calibrate one end-stop only (Task 3): extend/retract for linear "
                            "joints, left/right for yaw. Omit for a full both-ends sweep.")

    calall_p = subparsers.add_parser(
        "calibrate-all",
        help="Run the whole-board calibration sequence (auto per-leg sweep, M17 Task 3).")
    calall_p.add_argument("--port", default=None, metavar="PORT",
                          help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    calall_p.add_argument("--no-yaw", action="store_true",
                          help="Skip the hip-yaw steps (bench rig without yaw motors, spec §8).")
    calall_p.add_argument("--skip-validation", action="store_true",
                          help="Stop after the neutral pose; skip the chassis-required current-sense lifts.")

    vcs_p = subparsers.add_parser(
        "validate-current-sense",
        help="Run the current-sense lift validation standalone (assumes cal already done, M17 Task 4).")
    vcs_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")

    getcal_p = subparsers.add_parser(
        "get-calibration",
        help="Read back a joint's stored calibration (sensor type, min/max, calibrated flag).")
    getcal_p.add_argument("--port", default=None, metavar="PORT",
                          help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    getcal_p.add_argument("name", metavar="JOINT", help="Joint to read, e.g. FLHL.")

    jog_p = subparsers.add_parser(
        "jog", help="Drive one joint open-loop for a bounded time, then stop (reports pos + cal state).")
    jog_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    jog_p.add_argument("--joint", required=True, metavar="JOINT")
    jog_p.add_argument("--pwm", required=True, type=int, metavar="-255..255")
    jog_p.add_argument("--ms", type=int, default=1000, help="duration in ms (default 1000)")

    observe_p = subparsers.add_parser(
        "observe",
        help="Dump the assembled model observation (pos[0,1]/vel/contact_forces) from live telemetry (M17 Task 6).")
    observe_p.add_argument("--port", default=None, metavar="PORT",
                           help="Serial port of the board (default: auto-detect / $KRABBY_MCU_PORT).")
    observe_p.add_argument("--hz", type=float, default=0.0, metavar="HZ",
                           help="Stream at this rate (default: one snapshot). Velocity needs ≥2 snapshots.")
    observe_p.add_argument("--count", type=int, default=None, metavar="N",
                           help="Number of snapshots (default 1, or stream until Ctrl-C when --hz is set).")

    args = parser.parse_args()

    if args.command == "help":
        print(_command_help())
        return

    if args.command == "install":
        from firmware.install import run_install
        run_install()
        return

    if args.command == "show":
        from firmware.cli import cmd_show
        cmd_show(args.branch)
        return

    if args.command == "update":
        from firmware.cli import cmd_update
        cmd_update(args.channel, args.port)
        return

    if args.command == "set":
        from firmware.cli import cmd_set
        cmd_set(args.port, args.board, args.assignments)
        return

    if args.command == "get":
        from firmware.cli import cmd_get
        cmd_get(args.port, args.board, args.keys)
        return

    if args.command == "calibrate-joint":
        from firmware.cli import cmd_calibrate_joint
        cmd_calibrate_joint(args.port, args.name, args.direction)
        return

    if args.command == "calibrate-all":
        from firmware.cli import cmd_calibrate_all
        cmd_calibrate_all(args.port, args.no_yaw, args.skip_validation)
        return

    if args.command == "validate-current-sense":
        from firmware.cli import cmd_validate_current_sense
        cmd_validate_current_sense(args.port)
        return

    if args.command == "get-calibration":
        from firmware.cli import cmd_get_calibration
        cmd_get_calibration(args.port, args.name)
        return

    if args.command == "jog":
        from firmware.cli import cmd_jog
        cmd_jog(args.port, args.joint, args.pwm, args.ms)
        return

    if args.command == "observe":
        from firmware.cli import cmd_observe
        # No --count: one shot normally, or stream forever (count<=0) when --hz set.
        count = args.count if args.count is not None else (0 if args.hz else 1)
        cmd_observe(args.port, args.hz, count)
        return

    if args.debug:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    status = ""
    last_render = 0.0

    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[2J")  # clear screen once; frames repaint in place after

        while True:
            _read_keys(fd)

            if _quit:
                break

            key1 = is_pressed("1")
            key2 = is_pressed("2")
            selected = _selected_board(key1, key2)

            if is_pressed("0"):
                mcu.send_command_joints({
                    "FLHY": 0.5, "FRHY": 0.5, "FLHL": 0.5, "FLKL": 0.5, "FRHL": 0.5, "FRKL": 0.5,
                    "RLHY": 0.5, "MLHY": 0.5, "RLHL": 0.5, "RLKL": 0.5, "MLHL": 0.5, "MLKL": 0.5,
                    "RRHY": 0.5, "MRHY": 0.5, "RRHL": 0.5, "RRKL": 0.5, "MRHL": 0.5, "MRKL": 0.5,
                })
                status = "neutral sent — all joints → 0.5"
                _draw(_render_telemetry(mcu.joints, selected, status, _err_line(mcu)))
                time.sleep(0.3)  # debounce
                continue
            if is_pressed("v"):
                reply = mcu.read_version()
                if boards := (parse_ver_reply(reply) if reply else None):
                    parts = []
                    for i, (v, b, c) in enumerate(boards):
                        role = _BOARD_ROLES[i].strip() if i < len(_BOARD_ROLES) else f"board{i}"
                        if v != "-":
                            parts.append(f"{role}={v}|{b}|{c}")
                    status = "VER  " + ("  ".join(parts) if parts else "no response")
                else:
                    status = "VER  no response from MCU"
                _draw(_render_telemetry(mcu.joints, selected, status, _err_line(mcu)))
                time.sleep(0.3)
                continue

            # No 1/2 → FRONT; 1 → LEFT; 2 → RIGHT; 1+2 → all 18
            drive_front = (not key1 and not key2) or (key1 and key2)
            jog_cmds = {}
            for i in range(6):
                if is_pressed(EXTEND_KEYS[i]):
                    pwm = JOG_PWM
                elif is_pressed(RETRACT_KEYS[i]):
                    pwm = -JOG_PWM
                else:
                    pwm = 0
                jog_cmds[JOINTS_FRONT[i]] = pwm if drive_front else 0
                jog_cmds[JOINTS_LEFT[i]] = pwm if key1 else 0
                jog_cmds[JOINTS_RIGHT[i]] = pwm if key2 else 0

            mcu.send_commands_jog(jog_cmds)

            now = time.monotonic()
            if now - last_render >= 0.12:  # ~8 Hz repaint; control loop stays ~25 Hz
                _draw(_render_telemetry(mcu.joints, selected, status, _err_line(mcu)))
                last_render = now

            time.sleep(0.04)  # ~25 Hz

    except KeyboardInterrupt:
        mcu.send_command_joints_hold()
    finally:
        sys.stdout.write("\033[2J\033[H")  # clear the live frame so the shell prompt is clean
        sys.stdout.flush()
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        mcu.close()


if __name__ == "__main__":
    main()

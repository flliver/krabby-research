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

from firmware.gui.remote import DEFAULT_SERIAL_DEV
from firmware.krabby_mcu import BOARDS, KrabbyMCUSDK, parse_ver_reply, logger

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


def _log_jog(jog_cmds):
    active = {k: v for k, v in jog_cmds.items() if v != 0}
    if active:
        parts = "  ".join(f"{k} {v:+d}" for k, v in active.items())
        logger.info("JOG  %s", parts)
    else:
        logger.info("JOG  (hold)")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        print(f"krabby-firmware: {message}\n", file=sys.stderr)
        self.print_help(sys.stderr)
        sys.exit(2)


def main():
    parser = _Parser(
        prog="krabby-firmware",
        description="Krabby firmware tools. With no subcommand, launches the interactive MCU key-control menu.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (interactive menu only)")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("help", help="Show this help message and exit.")
    subparsers.add_parser("install", help="Set up host udev rules and serial permissions.")
    show_p = subparsers.add_parser("show", help="List attached boards and their firmware versions.")
    show_p.add_argument("--remote", default=None, metavar="HOST",
                        help="ssh host the board is attached to; auto-starts the serial/TCP "
                             "bridge there and probes through a tunnel instead of scanning "
                             "local USB ports (like the GUI's --remote).")
    show_p.add_argument("--serial", default=DEFAULT_SERIAL_DEV,
                        help="Serial device on the remote host (only with --remote).")
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
        help="Sweep ONE joint to both stops and persist its travel limits to EEPROM. MOVES THE JOINT.")
    cal_p.add_argument("joint", metavar="JOINT",
                       help="Joint name, e.g. FLHL (works for follower joints too — the front board forwards).")
    cal_p.add_argument("--port", default=None, metavar="PORT",
                       help="Serial port of the front board (default: auto-detect / $KRABBY_MCU_PORT).")

    args = parser.parse_args()

    if args.command == "help":
        parser.print_help()
        return

    if args.command == "install":
        from firmware.install import run_install
        run_install()
        return

    if args.command == "show":
        from firmware.cli import cmd_show
        cmd_show(remote=args.remote, remote_serial=args.serial)
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
        cmd_calibrate_joint(args.port, args.joint)
        return

    if args.debug:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        print("\n=== Krabby MCU — Direct key control (18 joints) ===")
        print("Extend: Q W E R T Y  |  Retract: A S D F G H")
        print("Hold 1: LEFT set  |  Hold 2: RIGHT set  |  Hold 1+2: all 18  |  No 1/2: FRONT")
        print("0: Neutral (0.5)  |  V: firmware version  |  ESC: Quit")
        print()

        prev_jog = {}

        while True:
            _read_keys(fd)

            if _quit:
                logger.info("ESC — quitting")
                break
            if is_pressed("0"):
                mcu.send_command_joints({
                    "FLHY": 0.5, "FRHY": 0.5, "FLHL": 0.5, "FLKL": 0.5, "FRHL": 0.5, "FRKL": 0.5,
                    "RLHY": 0.5, "MLHY": 0.5, "RLHL": 0.5, "RLKL": 0.5, "MLHL": 0.5, "MLKL": 0.5,
                    "RRHY": 0.5, "MRHY": 0.5, "RRHL": 0.5, "RRKL": 0.5, "MRHL": 0.5, "MRKL": 0.5,
                })
                time.sleep(0.3)  # debounce
                continue
            if is_pressed("v"):
                reply = mcu.read_version()
                if boards := (parse_ver_reply(reply) if reply else None):
                    for i, (v, b, c) in enumerate(boards):
                        role = _BOARD_ROLES[i] if i < len(_BOARD_ROLES) else f"board{i}"
                        if v != "-":
                            logger.info("VER  %s  %s  %s  %s", role, v, b, c)
                else:
                    logger.warning("VER  no response from MCU")
                time.sleep(0.3)
                continue

            key1 = is_pressed("1")
            key2 = is_pressed("2")
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

            if jog_cmds != prev_jog:
                _log_jog(jog_cmds)
                prev_jog = jog_cmds.copy()

            mcu.send_commands_jog(jog_cmds)

            time.sleep(0.04)  # ~25 Hz

    except KeyboardInterrupt:
        mcu.send_command_joints_hold()
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        mcu.close()


if __name__ == "__main__":
    main()

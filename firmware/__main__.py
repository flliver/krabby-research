"""
Interactive MCU menu. Run with: python -m firmware [--debug]
Works over SSH / headless — uses termios + select (msvcrt on Windows), no X11 required.
"""
import argparse
import select
import sys
import logging
import time
from typing import NoReturn, Optional

from firmware.krabby_mcu import (
    FULL_CALIBRATION_STATES,
    KrabbyMCUSDK,
    cal_robot_finished,
    logger,
    parse_ver_reply,
)

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


if sys.platform == "win32":
    import msvcrt

    def _next_key() -> Optional[str]:
        return msvcrt.getwch() if msvcrt.kbhit() else None
else:
    def _next_key() -> Optional[str]:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.buffer.read(1)
        return ch.decode("utf-8", errors="ignore") if ch else None


def _read_keys() -> None:
    """Drain all pending keypresses and refresh _last_seen timestamps."""
    global _quit
    while (ch := _next_key()) is not None:
        if ch == "\x1b":
            _quit = True
        elif ch:
            _last_seen[ch.lower()] = time.monotonic()


def is_pressed(k: str) -> bool:
    return time.monotonic() - _last_seen.get(k, 0.0) < _HOLD_WINDOW


def _log_jog(jog_cmds):
    active = {k: v for k, v in jog_cmds.items() if v != 0}
    if active:
        parts = "  ".join(f"{k} {v:+d}" for k, v in active.items())
        logger.info("JOG  %s", parts)
    else:
        logger.info("JOG  (hold)")


NEUTRAL_NEEDS_FULL_RUN = (
    "Neutral: no hip or knee has a position yet - run auto-calibrate (9) with the "
    "robot on its stand"
)
# Key 0 during a calibration run started here: refused, as the GUI's Neutral is. A T
# command would abort the run on the firmware and then drive joints the run has out
# (a hip mid-sweep, its knee parked).
NEUTRAL_DURING_CALIBRATION = (
    "Neutral: a calibration run is in progress - wait for it to end, or press a jog "
    "key or ESC to stop it"
)
NEUTRAL_NEEDS_CURL_ONLY = (
    "Neutral: no hip or knee has a position yet - home them with Calibrate > Curl only "
    "in the GUI (the menu has no Curl only key; 9 is the full run, on the stand only)"
)


def neutral_message(sent: int, skipped: list, states: Optional[dict] = None) -> str:
    """What to tell the person after key 0; empty when every hip and knee was sent.

    ``states`` are the SENS codes. When nothing was sent and every skipped joint only
    needs homing (e or x), the advice is Curl only, never the full run of key 9.
    """
    if sent == 0:
        states = states or {}
        if states and not any(states.get(n) in FULL_CALIBRATION_STATES for n in skipped):
            return NEUTRAL_NEEDS_CURL_ONLY
        return NEUTRAL_NEEDS_FULL_RUN
    if skipped:
        return f"Neutral sent to {sent} joints; no position, skipped: {', '.join(skipped)}"
    return ""


def _sensor_states(mcu) -> dict:
    """SENS codes by joint name; empty for firmware or test doubles without them."""
    getter = getattr(mcu, "sensor_states", None)
    states = getter() if callable(getter) else None
    return states if isinstance(states, dict) else {}


class CalibrationWatch:
    """Tracks a calibration started from the menu so the key loop does not break it.

    The loop streams a ``B`` jog frame every 40 ms, and the firmware aborts calibration
    on every ``B`` (also an all-zero one). While a run is active the loop therefore
    sends no jog frames; it resumes when the leader reports the run over
    (``CAL ROBOT DONE|FAIL|ABORTED|ERROR``), when no ``CAL`` line answers the start
    command within ``NO_REPLY_S`` (firmware that predates the CAL lines), or when the
    person presses a jog key (which then deliberately aborts the run).
    """

    NO_REPLY_S = 10.0

    def __init__(self, mcu, clock=None):
        self._mcu = mcu
        self._clock = clock if clock is not None else time.monotonic
        self.active = False
        self._started_at = 0.0
        self._count_at_start = 0

    def start(self) -> None:
        self.active = True
        self._started_at = self._clock()
        self._count_at_start = self._mcu.cal_line_count

    def stop(self) -> None:
        self.active = False

    def poll(self) -> Optional[str]:
        """Return a message when the run has ended (and mark it inactive), else None."""
        if not self.active:
            return None
        # Every line since start(), not just the newest: the leader prints its
        # end line after sending H, so a still-running follower's
        # "CAL <ROLE> ABORTED H" can arrive right after "CAL ROBOT FAIL ...".
        count = self._mcu.cal_line_count
        new_lines = count - self._count_at_start
        if new_lines > 0:
            for line in self._mcu.recent_cal_lines(new_lines):
                if cal_robot_finished(line):
                    self.active = False
                    return line
        if (
            count == self._count_at_start
            and self._clock() - self._started_at > self.NO_REPLY_S
        ):
            self.active = False
            return f"no CAL reply within {self.NO_REPLY_S:.0f} s; resuming jog control"
        return None


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
        cmd_show(args.branch)
        return

    if args.command == "update":
        from firmware.cli import cmd_update
        cmd_update(args.channel, args.port)
        return

    if args.debug:
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if not mcu.connect():
        return

    posix = sys.platform != "win32"
    if posix:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

    try:
        if posix:
            tty.setcbreak(fd)

        print("\n=== Krabby MCU — Direct key control (18 joints) ===")
        print("Extend: Q W E R T Y  |  Retract: A S D F G H")
        print("Hold 1: LEFT set  |  Hold 2: RIGHT set  |  Hold 1+2: all 18  |  No 1/2: FRONT")
        print("0: Neutral (0.5)  |  9: Auto-calibrate  |  V: firmware version  |  ESC: Quit")
        print("0 (Neutral) sends every hip to about 534 counts and every knee to mid-stroke; on a")
        print("solid stand keep hip targets at or below 560 counts until a stand envelope is decided.")
        print("0 moves only hips and knees that have a position (never yaw); it names the")
        print("ones it skips. Encoder joints must be homed after every power-up with")
        print("Calibrate > Curl only in the GUI (this menu has no Curl only key), then use")
        print("Neutral there: reopening the port can reset FRONT and lose its homes.")
        print("9 is the FULL calibration: it curls the whole robot (per leg: knee, hip, knee")
        print("again), then sweeps one joint at a time (yaw is not moved) and saves the")
        print("limits. Each leg: knee curl, hip curl, knee curl, knee uncurl (its travel checked),")
        print("knee park 521 counts in from its switch (264 out on the front-left leg),")
        print("hip sweep to 840 with the knee parked, hip back to its curl switch, knee curl.")
        print("The hip's extend limit (1069) is derived, not switch-verified. Flash all three")
        print("boards together with the same firmware. It runs at PWM 128 (range 110-150).")
        print("A drive into a limit switch ends only with the switch-open current signature;")
        print("a stall without it is reported as stuck mid-stroke.")
        print("Before 9, jog every hip inside 0-560 and every knee to its curl end.")
        print("Run 9 only with the robot on its stand; a hip or knee resting fully curled is")
        print("nudged out a little and straight back first. Any jog key or ESC stops it.")
        print()

        prev_jog = {}
        neutral_note = None  # last "0" message, so holding the key does not repeat it
        cal_watch = CalibrationWatch(mcu)

        while True:
            _read_keys()

            if _quit:
                logger.info("ESC — quitting")
                if cal_watch.active:
                    mcu.send_command_joints_hold()  # stop the run before we disconnect
                break
            if is_pressed("0"):
                if cal_watch.active and (message := cal_watch.poll()) is not None:
                    logger.info("CAL  run over: %s", message)
                if cal_watch.active:
                    # Refused (once per press) until the run's end line, a jog key or ESC.
                    if neutral_note != NEUTRAL_DURING_CALIBRATION:
                        print(NEUTRAL_DURING_CALIBRATION)
                    neutral_note = NEUTRAL_DURING_CALIBRATION
                    time.sleep(0.3)  # debounce
                    continue
                targets, skipped = mcu.neutral_targets(0.5)
                if targets:
                    mcu.send_command_joints(targets)
                note = neutral_message(len(targets), skipped, _sensor_states(mcu))
                if note and note != neutral_note:
                    print(note)
                neutral_note = note
                time.sleep(0.3)  # debounce
                continue
            neutral_note = None
            if is_pressed("9"):
                print("WARNING: This moves the hips and knees to find their limits, one")
                print("joint at a time (curl first, then a sweep; yaw is not moved).")
                print("Run it with the robot on its stand. A hip or knee resting fully curled")
                print("is nudged out a little and straight back so its sensor can be checked")
                print("(at most 24 hall counts for a hip, 12 for a knee; a curled knee then")
                print("springs out a little more, to about 20). A joint that still")
                print("shows no movement stops the run, and the run names it.")
                print("Hips sweep out 840 counts and back with the knee parked. If it stops")
                print("with 'no SENS', flash all three boards with the same firmware, or wait")
                print("a few seconds after connecting and press 9 again. If a knee stops with")
                print("'curled back only', press 9 again (do not re-curl it first).")
                print("Press any jog key or ESC to stop it.")
                cal_watch.start()  # before sending, so an instant reply is counted
                if mcu.send_command_calibrate() is None:
                    cal_watch.stop()
                time.sleep(0.5)
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

            if cal_watch.active:
                if any(is_pressed(k) for k in EXTEND_KEYS + RETRACT_KEYS):
                    # The jog frame sent below aborts the run on the firmware.
                    cal_watch.stop()
                    logger.warning("CAL  jog key pressed: calibration aborted")
                else:
                    # No jog frames while it runs: any B would abort calibration.
                    if (message := cal_watch.poll()) is not None:
                        logger.info("CAL  run over: %s", message)
                    time.sleep(0.04)
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
        if posix:
            termios.tcflush(fd, termios.TCIFLUSH)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        mcu.close()


if __name__ == "__main__":
    main()

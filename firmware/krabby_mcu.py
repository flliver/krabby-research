import os
import sys
import serial
import time
import threading
import logging
from collections import deque, namedtuple
from typing import Dict, Optional
from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.mcu_port import default_port

# --- LOGGING SETUP ---
# When run as `python -m firmware --debug`, __main__.py calls basicConfig(DEBUG) before this import.
# If krabby_mcu is imported alone, ensure a default handler exists.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
logger = logging.getLogger("KrabbySDK")


def parse_ver_reply(line: str) -> Optional[list[tuple[str, str, str]]]:
    """Parse a VER reply line. Returns None if not a VER line."""
    if not line.startswith("VER "):
        return None
    parts = line[4:].split()
    if not parts:
        return None
    versions = parts[0].split("|")
    branches = parts[1].split("|") if len(parts) > 1 else []
    commits  = parts[2].split("|") if len(parts) > 2 else []
    result = []
    for i, v in enumerate(versions):
        b = branches[i] if i < len(branches) else "-"
        c = commits[i]  if i < len(commits)  else "-"
        result.append((v, b, c))
    return result


# --- ERR telemetry channel (Task 1 §5) ---
# ERR is a third wire surface alongside command-response (SET/GET) and the joint
# telemetry stream: asynchronous "ERR <token> <code>" lines a board emits while a
# command runs. It is NOT a response to any command — the SDK surfaces each line as
# a tagged event (a bounded ring plus an optional callback). The token scopes the
# error (a joint name like FRKL, or a subsystem like "system"); the code is a
# self-describing string. Codes are owned by the emitting task (Task 4 owns the
# calibration vocabulary), so the SDK stores whatever arrives rather than rejecting
# it — the wire has no error-reply contract to validate against.
ErrorEvent = namedtuple("ErrorEvent", ["token", "code", "ts"])

# Canonical vocabulary from Task 1 §5, for reference and operator-facing display.
# Not enforced on parse (new codes land as their emitting task ships).
ERROR_CODES = frozenset({
    "motor_did_not_move",
    "motor_jammed",
    "pot_value_invalid",
    "hall_no_edges",
    "hall_drift",
    "not_calibrated",
    "current_sense_no_signal",
    "current_sense_no_spike",
})

_ERROR_RING_MAX = 128  # most recent ERR events retained for get_errors()


def parse_err_line(line: str) -> Optional[tuple]:
    """Parse 'ERR <token> <code>' into (token, code), or None if not a well-formed
    ERR line. Per §5 there are always exactly two tokens after ERR; any other arity
    is malformed and ignored (the firmware emits nothing else on this prefix)."""
    if not line.startswith("ERR "):
        return None
    parts = line[4:].split()
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


# --- SET / GET config command path ---
# The SDK is the validation layer: bad keys / roles / boards raise ValueError here,
# client-side, before any bytes hit the wire. The firmware silently ignores anything
# malformed, so there is no ERR reply for SET/GET.
CONFIG_KEYS = ("role", "serial")
ROLE_VALUES = ("FRONT", "LEFT", "RIGHT", "UNKNOWN")
BOARDS = ("front", "left", "right")
_SERIAL_MAX_LEN = 15  # firmware EepromLayout.serial is char[16] (15 chars + NUL)


def _board_suffix(board: Optional[str]) -> str:
    """Wire-command suffix for a target board. None/"front" -> "" (the board on USB)."""
    if board is None or board == "front":
        return ""
    if board == "left":
        return "_LEFT"
    if board == "right":
        return "_RIGHT"
    raise ValueError(f"invalid board {board!r}; expected one of {', '.join(BOARDS)}")


def _validate_value(key: str, val: str) -> None:
    if key not in CONFIG_KEYS:
        raise ValueError(f"unknown config key {key!r}; allowed: {', '.join(CONFIG_KEYS)}")
    if key == "role" and val not in ROLE_VALUES:
        raise ValueError(f"invalid role {val!r}; allowed: {', '.join(ROLE_VALUES)}")
    if key == "serial":
        if not val:
            raise ValueError("serial must be non-empty")
        if len(val) > _SERIAL_MAX_LEN:
            raise ValueError(f"serial {val!r} too long (max {_SERIAL_MAX_LEN} chars)")
        if any(c.isspace() for c in val) or not val.isascii() or not val.isprintable():
            raise ValueError(f"serial {val!r} must be printable ASCII with no spaces")


def build_set_line(board: Optional[str], pairs) -> str:
    """Build a 'SET[_LEFT|_RIGHT] <key> <val> …' wire line. Raises ValueError if invalid."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("set requires at least one key=value")
    parts = ["SET" + _board_suffix(board)]
    for key, val in pairs:
        _validate_value(key, val)
        parts += [key, val]
    return " ".join(parts)


def build_get_line(board: Optional[str], keys) -> str:
    """Build a 'GET[_LEFT|_RIGHT] <key> …' wire line. Raises ValueError if invalid."""
    keys = list(keys)
    if not keys:
        raise ValueError("get requires at least one key")
    for key in keys:
        if key not in CONFIG_KEYS:
            raise ValueError(f"unknown config key {key!r}; allowed: {', '.join(CONFIG_KEYS)}")
    return " ".join(["GET" + _board_suffix(board)] + keys)


def parse_get_reply(line: str):
    """Parse 'GET[_LEFT|_RIGHT] <key> <val> …' into (board, {key: val}). None if not a GET line."""
    parts = line.split()
    if not parts:
        return None
    tag_to_board = {"GET": "front", "GET_LEFT": "left", "GET_RIGHT": "right"}
    board = tag_to_board.get(parts[0])
    if board is None:
        return None
    kv = parts[1:]
    return board, {kv[i]: kv[i + 1] for i in range(0, len(kv) - 1, 2)}


def _raw_rx_to_stderr() -> bool:
    """When True, every non-empty decoded line is printed to stderr (see __main__.py --debug)."""
    v = os.environ.get("KRABBY_MCU_RAW_RX", "").strip().lower()
    return v in ("1", "true", "yes", "on")


# Must match firmware roleName() + "; " in arduino.ino (note "LEFT " has trailing space).
_TELEMETRY_LINE_PREFIXES = (
    "FRONT;",
    "UNKWN;",
    "LEFT ;",
    "RIGHT;",
)

# Joint names by board for readable debug output (FRONT / LEFT / RIGHT)
JOINT_GROUP_NAMES = (
    ("FRONT", ["FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"]),
    ("LEFT", ["RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL"]),
    ("RIGHT", ["RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL"]),
)


class KrabbyMCUSDK:
    def __init__(self, port=None, baud=115200):
        self.port = port or default_port()
        self.baud = baud
        self.ser = None
        self.running = False

        # Structured telemetry per joint
        self.joints: Dict[str, Optional[JointTelemetry]] = {}

        self.last_feedback_ts = None
        self.thread = None
        self._last_debug_log_ts = 0.0
        self.last_error = None
        self.last_cmd: Dict[str, Optional[float]] = {}
        self._last_ver_line: Optional[str] = None
        self._last_get_line: Optional[str] = None

        # ERR telemetry channel: a bounded ring of recent events plus an optional
        # callback invoked as each line arrives (see on_error / get_errors).
        self._errors = deque(maxlen=_ERROR_RING_MAX)
        self._error_callback = None

    def connect(self, settle: float = 5.0, hold: bool = True):
        """Open the serial port and start the reader thread.

        settle: seconds to wait after opening before reading (board boot). The
            interactive/GUI path uses the full 5 s; the config CLI passes a smaller
            value since it only needs the board to finish booting.
        hold: send an 'H' (hold all joints) on connect. The control paths want this so
            the legs don't drift; the config-only CLI (set/get) passes hold=False.
        """
        try:
            # Open without toggling DTR so the board is not reset on connect — we want
            # to talk to the already-running board and read its persisted EEPROM role.
            ser = serial.Serial()
            ser.port = self.port
            ser.baudrate = self.baud
            ser.timeout = 0.5
            ser.dtr = False
            ser.open()
            self.ser = ser
            time.sleep(settle)  # wait for board boot before starting reader
            self.running = True
            self.last_error = None
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")

            # On startup, immediately command the MCU to hold all joints at their
            # current positions so the legs don't drift before the user commands them.
            if hold:
                self.send_command_joints_hold()

            return True
        except Exception:
            logger.exception("Connection Failed")
            return False

    def _reader_loop(self):
        while self.running and self.ser.is_open:
            try:
                raw = self.ser.readline()
                try:
                    line = raw.decode('utf-8').strip()
                except UnicodeDecodeError as e:
                    logger.warning(
                        "Decode error on serial line (port=%s, len=%d): %s raw=%s",
                        self.port,
                        len(raw),
                        e,
                        raw.hex(),
                    )
                    line = raw.decode('utf-8', errors='ignore').strip()
                except Exception:
                    logger.exception("Decode error")
                    continue
                if not line:
                    continue
                if _raw_rx_to_stderr():
                    print(f"[serial rx] {line}", file=sys.stderr, flush=True)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug("serial rx: %s", line)
                if line.startswith(_TELEMETRY_LINE_PREFIXES):
                    self._parse_joint_line(line)
                    self.last_feedback_ts = time.time()
                elif line.startswith("VER "):
                    self._last_ver_line = line
                elif line.startswith("GET"):
                    self._last_get_line = line
                elif line.startswith("ERR "):
                    self._record_error(line)
                elif "Krabby" in line or "CAL" in line or "Saved" in line:
                    logger.info(f"[MCU] {line}")

            except (serial.SerialException, AttributeError) as e:
                if self.running:
                    logger.exception("Reader loop error")
                else:
                    logger.debug("Reader loop stopped: %s", e)
                self.last_error = e
                self.running = False
                break
            except Exception as exc:
                # close() tears down the port mid-readline(); pyserial then reads with
                # a None size (TypeError). When we're already shutting down that's a
                # clean stop, not an error.
                if self.running:
                    logger.exception("Reader loop error")
                else:
                    logger.debug("Reader loop stopped: %s", exc)
                self.last_error = exc
                self.running = False
                break

    def _parse_joint_line(self, line: str):
        jts = JointTelemetry.parse_line(line)
        if not jts:
            return
        for jt in jts:
            self.joints[jt.name] = jt

    # --- ERR telemetry channel ---------------------------------------------

    def _record_error(self, line: str):
        """Parse one ERR line and surface it: append to the ring and invoke the
        registered callback. Malformed lines are dropped. A callback that raises is
        logged, never propagated — an ERR event must not take down the reader."""
        parsed = parse_err_line(line)
        if parsed is None:
            return
        event = ErrorEvent(parsed[0], parsed[1], time.time())
        self._errors.append(event)
        cb = self._error_callback
        if cb is not None:
            try:
                cb(event)
            except Exception:
                logger.exception("on_error callback raised for %s", line)

    def on_error(self, callback):
        """Register a callback invoked with each ErrorEvent(token, code, ts) as it
        arrives. Pass None to clear. Exceptions in the callback are logged, not raised."""
        self._error_callback = callback

    def get_errors(self) -> list:
        """Snapshot of recent ERR events, oldest first (up to the ring capacity)."""
        return list(self._errors)

    def clear_errors(self):
        """Drop all retained ERR events."""
        self._errors.clear()

        # Debug Log: FRONT / LEFT / RIGHT each on its own line
        now = time.time()
        if logger.isEnabledFor(logging.DEBUG) and (now - self._last_debug_log_ts) >= 0.25:
            for group_name, names in JOINT_GROUP_NAMES:
                parts = []
                for name in names:
                    jt = self.joints.get(name)
                    if jt:
                        parts.append(jt.format_compact(self.last_cmd.get(name)))
                if parts:
                    logger.debug("JOINTS %s %s", group_name, "; ".join(parts))
            self._last_debug_log_ts = now

    def send_command_joints(self, cmds_by_joint: Dict[str, float]):
        """
        Send commands keyed by joint name.
        """
        if not self.ser or not self.ser.is_open:
            return

        seq = []
        for key, raw_val in cmds_by_joint.items():
            val = max(0.0, min(1.0, raw_val))
            seq.append((key, val))
            self.last_cmd[key] = val

        parts = ["T"]
        for name, val in seq:
            parts.append(name)
            parts.append(f"{val:.3f}")

        cmd = " ".join(parts) + "\n"
        self.ser.write(cmd.encode('utf-8'))
        self.ser.flush()

        logger.info("CMD -> %s", " ".join(parts))

    def send_commands_jog(self, cmds_by_joint: Dict[str, int]):
        """
        Send all jog commands in one batch (B name pwm name pwm ...) so the leader
        can forward one line to followers instead of 18 separate J lines.
        """
        if not self.ser or not self.ser.is_open:
            return
        parts = ["B"]
        for name, raw_pwm in cmds_by_joint.items():
            pwm = max(-255, min(255, int(raw_pwm)))
            parts.append(name)
            parts.append(str(pwm))
        cmd = " ".join(parts) + " \n"
        self.ser.write(cmd.encode('utf-8'))
        self.ser.flush()

    def send_command_jog(self, joint_name: str, pwm: int):
        """ Send J<name> <pwm> (-255 to 255) """
        if not self.ser or not self.ser.is_open:
            return
        pwm = max(-255, min(255, int(pwm)))
        cmd = f"J{joint_name} {pwm}\n"
        self.ser.write(cmd.encode('utf-8'))
        self.ser.flush()

    def read_version(self, timeout: float = 1.0) -> Optional[str]:
        if not self.ser or not self.ser.is_open:
            return None
        self._last_ver_line = None
        self.ser.write(b"V\n")
        self.ser.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_ver_line is not None:
                return self._last_ver_line
            time.sleep(0.02)
        return None

    def send_set(self, board: Optional[str] = None, **kwargs: str) -> None:
        """Write config keys to a board (fire-and-forget; no reply).

        board=None/"front" targets the board on USB; "left"/"right" forward via the
        front board over the inter-board serials. Validates client-side and raises
        ValueError before anything reaches the wire — to confirm, follow with send_get.

            mcu.send_set(role="FRONT", serial="FRT-0042")
            mcu.send_set(board="left", role="LEFT")
        """
        line = build_set_line(board, list(kwargs.items()))  # raises ValueError if invalid
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write((line + "\n").encode("utf-8"))
        self.ser.flush()
        logger.info("CMD -> %s", line)

    def send_get(self, *keys: str, board: Optional[str] = None,
                 timeout: float = 1.0) -> Optional[Dict[str, str]]:
        """Read config keys from a board; block on the tagged reply. Returns a dict
        (or None on timeout). Same request/reply pattern as read_version.

            mcu.send_get("role", "serial")               # the board on USB
            mcu.send_get("role", board="left")           # the left follower
        """
        line = build_get_line(board, list(keys))  # raises ValueError if invalid
        if not self.ser or not self.ser.is_open:
            return None
        want_board = board or "front"
        self._last_get_line = None
        self.ser.write((line + "\n").encode("utf-8"))
        self.ser.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_get_line is not None:
                parsed = parse_get_reply(self._last_get_line)
                self._last_get_line = None
                if parsed and parsed[0] == want_board:
                    return parsed[1]
            time.sleep(0.02)
        return None

    def send_command_calibrate(self):
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write(b"C\n")
        self.ser.flush()
        logger.info("CMD -> AUTO-CALIBRATE (C)")

    def send_command_joints_hold(self):
        """
        Send the 'H' command to hold all joints at their current positions.
        """
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write(b"H\n")
        self.ser.flush()
        logger.info("CMD -> H")

    def wait_for_move(self, seconds):
        time.sleep(seconds)

    def close(self):
        self.running = False
        if self.ser:
            try:
                # Interrupt any blocking read so the reader thread can exit cleanly
                cancel_read = getattr(self.ser, "cancel_read", None)
                if callable(cancel_read):
                    cancel_read()
            except Exception:
                logger.debug("cancel_read failed during close", exc_info=True)
            try:
                self.ser.close()
            except Exception:
                logger.debug("Serial close failed", exc_info=True)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
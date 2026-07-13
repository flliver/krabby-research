import os
import sys
import serial
import time
import threading
import logging
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


# --- SET / GET config command path ---
# The SDK is the validation layer: bad keys / roles / boards raise ValueError here,
# client-side, before any bytes hit the wire. The firmware silently ignores anything
# malformed, so there is no error reply for SET/GET.
CONFIG_KEYS = ("role", "serial")              # writable via SET
GETTABLE_KEYS = CONFIG_KEYS + ("version",)    # readable via GET; version is read-only
ROLE_VALUES = ("FRONT", "LEFT", "RIGHT", "UNKNOWN")
# Single source of truth for the board <-> wire-suffix mapping: the board on USB
# (front) takes the bare command; a follower gets a side suffix the leader routes on.
# BOARDS, _board_suffix (encode), and parse_get_reply's tag map (decode) all derive
# from this, so the encode and decode sides can't drift.
_BOARD_SUFFIX = {"front": "", "left": "_LEFT", "right": "_RIGHT"}
BOARDS = tuple(_BOARD_SUFFIX)
_TAG_TO_BOARD = {"GET" + suffix: board for board, suffix in _BOARD_SUFFIX.items()}
_SERIAL_MAX_LEN = 15  # firmware EepromLayout.serial is char[16] (15 chars + NUL)


def _board_suffix(board: Optional[str]) -> str:
    """Wire-command suffix for a target board. None/"front" -> "" (the board on USB)."""
    suffix = _BOARD_SUFFIX.get("front" if board is None else board)
    if suffix is None:
        raise ValueError(f"invalid board {board!r}; expected one of {', '.join(BOARDS)}")
    return suffix


def _check_key(key: str, allowed=CONFIG_KEYS) -> None:
    if key not in allowed:
        raise ValueError(f"unknown config key {key!r}; allowed: {', '.join(allowed)}")


def _validate_value(key: str, val: str) -> None:
    _check_key(key)  # SET: only writable keys
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
        _check_key(key, GETTABLE_KEYS)
    return " ".join(["GET" + _board_suffix(board)] + keys)


def parse_get_reply(line: str):
    """Parse 'GET[_LEFT|_RIGHT] <key> <val> …' into (board, {key: val}). None if not a GET line."""
    parts = line.split()
    if not parts:
        return None
    board = _TAG_TO_BOARD.get(parts[0])
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
ALL_JOINT_NAMES = frozenset(n for _, names in JOINT_GROUP_NAMES for n in names)


def parse_cal_reply(line: str):
    """Parse a firmware calibration reply into a dict, or None if not a CAL line.

    "CAL <joint> <min> <max> saved" -> {"joint", "min", "max", "ok": True}
    "CAL <joint> FAIL <why>"        -> {"joint", "why", "ok": False}
    """
    parts = line.split()
    if len(parts) < 3 or parts[0] != "CAL":
        return None
    if parts[2] == "FAIL":
        return {"joint": parts[1], "why": parts[3] if len(parts) > 3 else "?", "ok": False}
    try:
        return {"joint": parts[1], "min": int(parts[2]), "max": int(parts[3]), "ok": True}
    except (ValueError, IndexError):
        return None


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
        self._last_cal_line: Optional[str] = None

    def connect(self, settle: Optional[float] = None, hold: bool = True):
        """Open the serial port and start the reader thread.

        settle: seconds to wait after opening before reading (board boot). Default
            (None) picks by port type: 5 s on a local device (CH340 adapters reset
            the board on open regardless of DTR, so wait out its boot), 0.5 s over a
            socket:// bridge (our open never touches the board). The config CLI
            passes a smaller value since it only needs the board to finish booting.
        hold: send an 'H' (hold all joints) on connect. The control paths want this
            so the legs don't drift; the config-only CLI (set/get) passes hold=False.
        """
        try:
            # serial_for_url opens plain device paths via Serial and socket://host:port
            # URLs via a TCP client (the remote serial/TCP bridge, for running the
            # GUI/SDK on a different host than the MCU). Clear DTR before opening so
            # a local open does not reset the board — we want to talk to the
            # already-running board and read its persisted EEPROM role. Over a
            # socket DTR is a no-op; the bridge owns the real port.
            ser = serial.serial_for_url(self.port, baudrate=self.baud,
                                        timeout=0.5, do_not_open=True)
            ser.dtr = False
            ser.open()
            self.ser = ser
            if settle is None:
                settle = 0.5 if "://" in self.port else 5.0
            time.sleep(settle)
            self.running = True
            self.last_error = None
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")

            # On startup, immediately command the MCU to hold all joints
            # at their current positions so the legs don't drift or move
            # unexpectedly before the user issues a command.
            if hold:
                self.send_command_joints_hold()

            return True
        except Exception:
            logger.exception("Connection Failed")
            return False

    def _reader_loop(self):
        while self.running and self.ser.is_open:
            # The link-death tuple wraps ONLY the read, so a bug in the parse code
            # below still gets a full logger.exception traceback instead of being
            # misreported as a lost link.
            #   SerialException/OSError: the link itself died (unplug, bridge gone).
            #   AttributeError/TypeError: pyserial's fd/handle goes None when close()
            #   lands while readline() is blocked — the normal shutdown race, not a
            #   fault. self.running distinguishes the two: close() clears it first.
            try:
                raw = self.ser.readline()
            except (serial.SerialException, OSError, AttributeError, TypeError) as e:
                if self.running:
                    logger.warning("Serial link lost on %s: %s", self.port, e)
                else:
                    logger.debug("Reader loop stopped: %s", e)
                self.last_error = e
                self.running = False
                break
            try:
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
                    # "GET …" / "GET_LEFT …" / "GET_RIGHT …" — tagged config reply
                    self._last_get_line = line
                elif line.startswith("CAL "):
                    # calibration result ("CAL <joint> ... saved" / "... FAIL <why>")
                    self._last_cal_line = line
                    logger.info(f"[MCU] {line}")
                elif "Krabby" in line or "Saved" in line:
                    logger.info(f"[MCU] {line}")

            except Exception as exc:
                logger.exception("Reader loop error")
                self.last_error = exc
                self.running = False
                break

    def _parse_joint_line(self, line: str):
        jts = JointTelemetry.parse_line(line)
        if not jts:
            return
        for jt in jts:
            self.joints[jt.name] = jt

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

    def calibrate_joint(self, joint: str, timeout: float = 40.0) -> Optional[Dict]:
        """Calibrate ONE joint: the board sweeps it to both stops and persists the
        limits to EEPROM. Blocks until the board's "CAL ..." reply (the sweep takes
        seconds; the owning board pauses telemetry while it runs). Returns
        parse_cal_reply's dict, or None if no reply arrived in `timeout`.

        Validates the joint name client-side (ValueError) before touching the wire.
        Works for follower joints too — the front board broadcasts the command.
        """
        joint = joint.upper()
        if joint not in ALL_JOINT_NAMES:
            raise ValueError(f"unknown joint {joint!r}; expected one of "
                             f"{', '.join(sorted(ALL_JOINT_NAMES))}")
        if not self.ser or not self.ser.is_open:
            return None
        self._last_cal_line = None
        self.ser.write(f"C {joint}\n".encode("utf-8"))
        self.ser.flush()
        logger.info("CMD -> C %s", joint)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_cal_line is not None:
                parsed = parse_cal_reply(self._last_cal_line)
                self._last_cal_line = None
                if parsed and parsed["joint"] == joint:
                    return parsed
            time.sleep(0.05)
        return None

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
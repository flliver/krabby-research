import os
import sys
import serial
import time
import threading
import logging
from collections import deque, namedtuple
from typing import Dict, Optional
from firmware import joints
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

# Canonical calibration reason-code vocabulary (Task 1 §5 / Task 4 §4a, single source of
# truth shared by Tasks 2/3/4). Codes are bare strings on the wire; this set is for
# reference/operator display and is not enforced on parse (new codes land as tasks ship).
ERROR_CODES = frozenset({
    "motor_did_not_move",
    "motor_jammed",
    "pot_value_invalid",
    "hall_no_edges",
    "hall_drift",
    "not_calibrated",
    "not_in_starting_pose",
    "current_sense_no_signal",
    "current_sense_no_spike",
    "sensor_type_mismatch",  # calibration found a sensor that disagrees with the joint's fixed type
    "hall_storm",            # encoder edge rate saturated the MCU; joint coasted, counting paused
})

# Reason-code → operator-facing fix instruction (Task 4 §5 / 4f). `{joint}` is filled per
# event. Kept in lockstep with ERROR_CODES: every code a task emits has an entry here.
_FIX_INSTRUCTIONS = {
    "motor_did_not_move":      "Check motor power wiring on {joint} (H-bridge output + power harness).",
    "motor_jammed":            "Motor on {joint} is drawing current but not moving — mechanical jam, bind, or obstruction. Investigate before re-driving.",
    "pot_value_invalid":       "Check potentiometer wiring on {joint} (3-wire harness: VCC, signal, GND).",
    "hall_no_edges":           "Check Hall encoder wiring on {joint} (A/B channels and power).",
    "hall_drift":              "Hall count drifts between repeat sweeps on {joint}. Check signal integrity (cable shielding, ground loops) and the encoder coupling to the motor shaft.",
    "not_calibrated":          "{joint} is PARTIALLY_CALIBRATED — a position target was rejected. Jog the joint to either end-stop or run `calibrate-joint {joint}`.",
    "not_in_starting_pose":    "Robot is not in the auto-squat starting pose. Re-run calibration from any state — the cal routine drives there automatically.",
    "sensor_type_mismatch":    "Sensor on {joint} disagrees with its expected type (e.g. a Hall actuator in a pot slot, or vice versa). Check that the correct motor is wired to this slot.",
    "current_sense_no_signal": "Check current-sense wiring on {joint} (shunt + analog IS line back to the shield).",
    "current_sense_no_spike":  "Current sense on {joint} reads but does not spike under load. Verify the motor is actually bearing weight; if so, check that the shunt resistor isn't damaged.",
    "hall_storm":              "Encoder on {joint} produced edges faster than the MCU can service; the motor was coasted and counting paused. Drive it at a lower PWM (fast-shaft encoders saturate at full jog speed).",
}

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


def parse_cal_reply(line: str):
    """Parse a stored-calibration reply 'CAL <joint> type <POT|HALL> rev <0|1> min <n>
    max <n> cal <0|1>' into (joint, {type, rev, min, max, cal}). None if not a CAL line."""
    parts = line.split()
    if len(parts) < 2 or parts[0] != "CAL":
        return None
    name = parts[1]
    kv = parts[2:]
    return name, {kv[i]: kv[i + 1] for i in range(0, len(kv) - 1, 2)}


def parse_cal_line(line: str):
    """Parse a whole-board GET-calibration line (Task 4 §4):
    'CAL <joint> <POT|HALL> <min> <max> <reversed>' -> (joint, {type, min, max, reversed}).
    Returns None if not that positional shape — notably the richer, field-labelled per-joint
    Q reply (parse_cal_reply), whose third token is the literal 'type'."""
    parts = line.split()
    if len(parts) != 6 or parts[0] != "CAL" or parts[2] not in ("POT", "HALL"):
        return None
    return parts[1], {"type": parts[2], "min": parts[3], "max": parts[4], "reversed": parts[5]}


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

# Joint names by board for readable debug output, in slot order (FRONT / LEFT / RIGHT).
# Derived from the static joint registry (firmware/joints.py).
JOINT_GROUP_NAMES = tuple(
    (board, [js.name for js in joints.board_joints(board)])
    for board in joints.BOARD_LEGS
)

# Every drivable joint name — for client-side validation of calibrate_joint (Task 2).
ALL_JOINT_NAMES = frozenset(joints.JOINTS)


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
        self._board_cals: Dict[str, dict] = {}   # whole-board GET-calibration accumulator

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
            if "://" in self.port:
                # A URL port (e.g. socket://host:port) — a TCP serial bridge for running
                # the GUI/SDK on a different host than the MCU. No DTR over a socket, so
                # the board isn't reset by the client (the bridge owns the real port).
                ser = serial.serial_for_url(self.port, baudrate=self.baud,
                                            timeout=0.5, do_not_open=True)
                ser.open()
            else:
                # Local device: open without toggling DTR so the board is not reset on
                # connect — we want the already-running board's persisted EEPROM role.
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
                elif line.startswith("CAL "):
                    board_line = parse_cal_line(line)
                    if board_line:                       # whole-board GET-calibration line
                        self._board_cals[board_line[0]] = board_line[1]
                    else:                                # richer per-joint Q reply
                        self._last_cal_line = line
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

    @staticmethod
    def explain_failures(errors) -> list:
        """Turn calibration/validation failures into operator-facing fix instructions
        (Task 4 §5 / 4f). `errors` is an iterable of either ``ErrorEvent`` (as returned by
        ``get_errors()``) or bare ``(joint, code)`` tuples. Unknown codes get a generic
        line so a new firmware code never silently drops out of the operator's view."""
        out = []
        for e in errors:
            joint = getattr(e, "token", None)
            code = getattr(e, "code", None)
            if joint is None:            # not an ErrorEvent — treat as (joint, code)
                joint, code = e
            tpl = _FIX_INSTRUCTIONS.get(code, f"Unknown failure on {{joint}}: {code}")
            out.append(tpl.format(joint=joint))
        return out

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

    def calibrate_joint(self, name: str, direction: Optional[str] = None):
        """Trigger a single-joint calibration (wire command ``K<name> [direction]``, M17 Task 2).

        Fire-and-forget: the firmware sweeps the joint, auto-detects the sensor type +
        direction, and persists the result — it sends no completion reply. Watch the
        joint-telemetry stream and any ``ERR <joint> <code>`` lines (Task 1 §5) for
        failures; read the recorded values back via ``get_calibration`` / ``GET calibration``.

        ``direction`` selects how much of the travel this run sweeps (Task 3 cals one DOF
        at a time): ``None`` = full both-ends sweep; for linear joints ``"extend"`` /
        ``"retract"`` drive one end-stop and park there; for yaw joints ``"left"`` /
        ``"right"``. A mismatched pairing (e.g. ``"extend"`` on a yaw joint) is a client
        bug — raise here rather than letting the firmware silently drop it.
        """
        if name not in ALL_JOINT_NAMES:
            raise ValueError(
                f"unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")
        direction = self._validate_cal_direction(name, direction)
        if not self.ser or not self.ser.is_open:
            return
        wire = f"K{name}\n" if direction is None else f"K{name} {direction}\n"
        self.ser.write(wire.encode('utf-8'))
        self.ser.flush()
        logger.info("CMD -> K %s %s(calibrate joint)", name, f"{direction} " if direction else "")

    def calibrate_all(self, include_yaw: bool = False, validate: bool = True):
        """Trigger the whole-board calibration sequence (wire ``CALL [yaw] [skipval]``,
        M17 Task 3 + Task 4). The firmware runs the standard per-leg cal sequence for its
        own legs (Task-2 directional ``calibrateJoint`` + closed-loop pose moves), then —
        unless ``validate=False`` — the neutral pose (every joint → 0.5) and the per-leg
        current-sense lift validation (Task 4).

        Fire-and-forget like ``calibrate_joint``: no completion reply. Watch telemetry and
        any ``ERR <joint> <code>`` lines (the cal sequence halts + holds all motors on the
        first cal failure, Task 3 §3f; validation emits a line per failing joint), then read
        results back via ``get_calibration``.

        Hip-yaw joints are SKIPPED by default: they have no end-stops, so the end-stop
        cal sweep would drive them until the leg jams (yaw cal is out of M17 pending a
        homing scheme). ``include_yaw=True`` (``CALL yaw``) opts in on hardware that can
        take it. ``validate=False`` (``CALL skipval``) stops after the neutral pose,
        skipping the chassis-required lifts (Task 4 §7).
        """
        if not self.ser or not self.ser.is_open:
            return
        # Send "noyaw" explicitly rather than relying on the firmware default: a board
        # still running pre-opt-in firmware treats a bare CALL as include-yaw, so the
        # explicit token keeps the skip safe across firmware version skew.
        args = (" yaw" if include_yaw else " noyaw") + ("" if validate else " skipval")
        wire = f"CALL{args}\n"
        self.ser.write(wire.encode('utf-8'))
        self.ser.flush()
        logger.info("CMD -> %s (calibrate all)", wire.strip())

    def validate_current_sense(self):
        """Run the current-sense validation standalone (wire ``CALL valonly``, Task 4 §6),
        assuming per-joint cal already ran. Drives every joint to the neutral pose, then
        lifts each leg in turn and emits ``ERR <joint> current_sense_no_signal/no_spike``
        for any actuator whose current doesn't read sanely under load. Chassis-required."""
        if not self.ser or not self.ser.is_open:
            return
        self.ser.write(b"CALL valonly\n")
        self.ser.flush()
        logger.info("CMD -> CALL valonly (validate current sense)")

    @staticmethod
    def _validate_cal_direction(name: str, direction: Optional[str]) -> Optional[str]:
        """Check a cal direction against the joint type, returning the normalized token
        (or None for a full sweep). Valid tokens come from the joint registry: linear
        joints take extend/retract, yaw joints left/right. Raises ValueError for an
        unknown joint name (previously it fell through as linear)."""
        if direction is None:
            return None
        direction = direction.lower()
        js = joints.spec(name)
        if direction not in js.cal_directions:
            kind = "yaw" if js.is_yaw else "linear"
            raise ValueError(
                f"direction {direction!r} invalid for {kind} joint {name}; "
                f"valid: {', '.join(sorted(js.cal_directions))}")
        return direction

    def get_calibration(self, name: str, timeout: float = 2.0) -> Optional[dict]:
        """Read back the stored calibration for one joint (wire command ``Q<name>``).

        Returns {type, rev, min, max, cal} (all strings, as they arrive on the wire),
        or None on timeout. ``cal == "0"`` means the firmware recorded values but did
        not trust them (e.g. the sweep range failed the sanity check). Same
        request/reply + tagged-line pattern as read_version / send_get.
        """
        if name not in ALL_JOINT_NAMES:
            raise ValueError(
                f"unknown joint {name!r}; valid joints: {', '.join(sorted(ALL_JOINT_NAMES))}")
        if not self.ser or not self.ser.is_open:
            return None
        self._last_cal_line = None
        self.ser.write(f"Q{name}\n".encode('utf-8'))
        self.ser.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._last_cal_line
            if line is not None:
                parsed = parse_cal_reply(line)
                if parsed and parsed[0] == name:
                    return parsed[1]
                self._last_cal_line = None  # a CAL for another joint; keep waiting
            time.sleep(0.02)
        return None

    def get_all_calibration(self, board: Optional[str] = None,
                            timeout: float = 2.0, expect: int = 6) -> Dict[str, dict]:
        """Whole-board calibration dump (wire ``GET[_LEFT|_RIGHT] calibration``, Task 4 §4).

        Returns ``{joint: {type, min, max, reversed}}`` for the addressed board's joints
        (``board`` = None/"front"/"left"/"right"). The firmware streams one positional
        ``CAL <joint> <type> <min> <max> <reversed>`` line per joint; the reader collects
        them by joint name (followers' lines relay up and self-attribute by name). Leaner
        than ``get_calibration`` per joint — the operator-facing after-the-fact dump.
        """
        if not self.ser or not self.ser.is_open:
            return {}
        self._board_cals = {}
        prefix = "GET" if board in (None, "front") else f"GET_{board.upper()}"
        self.ser.write(f"{prefix} calibration\n".encode('utf-8'))
        self.ser.flush()
        deadline = time.time() + timeout
        while time.time() < deadline and len(self._board_cals) < expect:
            time.sleep(0.05)
        return dict(self._board_cals)

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
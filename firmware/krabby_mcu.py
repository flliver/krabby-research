import math
import os
import sys
import serial
import time
import threading
import logging
from collections import deque
from numbers import Integral
from typing import Dict, Iterable, List, Optional, Sequence, Union
from firmware.interfaces.battery_telemetry import BatteryTelemetry
from firmware.interfaces.imu_telemetry import ImuTelemetry
from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.mcu_port import McuSerialTarget, default_target

# Must match the firmware BAUD_RATE used for the telemetry link.
DEFAULT_BAUD = 250000

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
    commits = parts[2].split("|") if len(parts) > 2 else []
    result = []
    for i, v in enumerate(versions):
        b = branches[i] if i < len(branches) else "-"
        c = commits[i] if i < len(commits) else "-"
        result.append((v, b, c))
    return result


def _raw_rx_to_stderr() -> bool:
    """When True, every non-empty decoded line is printed to stderr (see __main__.py --debug)."""
    v = os.environ.get("KRABBY_MCU_RAW_RX", "").strip().lower()
    return v in ("1", "true", "yes", "on")

# Joint names by board for readable debug output (FRONT / LEFT / RIGHT)
JOINT_GROUP_NAMES = (
    ("FRONT", ["FLHY", "FLHL", "FLKL", "FRHY", "FRHL", "FRKL"]),
    ("LEFT", ["RLHY", "RLHL", "RLKL", "MLHY", "MLHL", "MLKL"]),
    ("RIGHT", ["RRHY", "RRHL", "RRKL", "MRHY", "MRHL", "MRKL"]),
)

# --- Auto-calibration -------------------------------------------------------
# These mirror firmware/arduino/src/actuator/calibration_constants.h. A unit test
# (tests/unit/firmware/test_krabby_mcu_calibration.py) parses that header and fails
# if any value below drifts from it.
# firmware CAL_PWM_MIN: slowest accepted calibration speed. Owner decision F4
# (2026-11-24): PWM 80 stalls mid-stroke with no current signature (FLHL stopped 229
# edges short of its curl switch), so nothing below 110 is accepted.
CAL_PWM_MIN = 110
CAL_PWM_MAX = 150  # firmware CAL_PWM_MAX: fastest accepted calibration speed
CAL_DEFAULT_PWM = 128  # firmware CAL_DEFAULT_PWM: speed used when none is given
CAL_PASS_SEEK = "SEEK"  # curl every leg (knee, hip, knee again); saves nothing
# Exercise each leg and save its limits. A hall leg runs the table-safe order: knee
# curl, hip curl, knee uncurl, knee park CAL_KNEE_PARK_EDGES out, hip sweep to
# CAL_HIP_SWEEP_EDGES with the knee parked, hip back to its curl switch, knee curl.
CAL_PASS_SWEEP = "SWEEP"
CAL_PASS_NAMES = (CAL_PASS_SEEK, CAL_PASS_SWEEP)  # run order when both are requested
# Table-safe sweep of a hall leg (PROVISIONAL: only FL was modelled, every leg uses
# FL's values). The knee parks CAL_KNEE_PARK_EDGES out of its curl stop (firmware
# position -264), inside CAL_KNEE_PARK_BAND_MIN_EDGES..CAL_KNEE_PARK_BAND_MAX_EDGES
# out. With the knee held there the hip sweeps by count to CAL_HIP_SWEEP_EDGES and
# returns to its curl switch. The hip's extend switch is never sought on a solid
# stand: at 1069 no knee setting keeps the leg clear of belly, table and floor.
# The hip's saved range is [0, CAL_HIP_STROKE_EDGES] (FL's switch-to-switch stroke,
# derived, not switch-verified). Pot hips still sweep end to end.
CAL_KNEE_PARK_EDGES = 264
CAL_KNEE_PARK_BAND_MIN_EDGES = 244
CAL_KNEE_PARK_BAND_MAX_EDGES = 280
# FL's knee stroke, curl stop to switch. Before its park a table-safe knee
# whose stroke from its home is more than
# CAL_KNEE_STROKE_SHORT_MAX_EDGES short of it fails: its home is short of its
# curl stop (an obstacle or a bind there).
CAL_KNEE_STROKE_EDGES = 785
CAL_KNEE_STROKE_SHORT_MAX_EDGES = 24
CAL_KNEE_STROKE_MIN_EDGES = CAL_KNEE_STROKE_EDGES - CAL_KNEE_STROKE_SHORT_MAX_EDGES
CAL_HIP_SWEEP_EDGES = 840
CAL_HIP_STROKE_EDGES = 1069
# Verification nudge bounds in hall edges. A curled knee springs out a few more
# edges after its nudge stops (5-9 measured), so a knee nudge ends about 18-21 out.
CAL_HIP_NUDGE_EDGES = 24
CAL_KNEE_NUDGE_EDGES = 12
# Bounds a measured hall knee stroke must fall inside (PROVISIONAL: one knee, FLKL,
# measured 776 edges).
CAL_HALL_MIN_STROKE_EDGES = 450
CAL_HALL_MAX_STROKE_EDGES = 1500

# Curl direction, confirmed on the robot and identical on all six legs: the HIP
# curls by RETRACTING (-PWM) and the KNEE curls by EXTENDING (+PWM). The firmware
# keeps +PWM = extend, -PWM = retract. Source of truth:
# CAL_HIP_CURL_PWM_SIGN / CAL_KNEE_CURL_PWM_SIGN in calibration_constants.h.
CAL_HIP_CURL_PWM_SIGN = -1
CAL_KNEE_CURL_PWM_SIGN = +1
# Joint-name suffix -> curl PWM sign. Names end HY (yaw), HL (hip), KL (knee). Yaw is
# deliberately absent: calibration never moves it, so it has no curl direction.
JOINT_CURL_PWM_SIGN = {
    "HL": CAL_HIP_CURL_PWM_SIGN,
    "KL": CAL_KNEE_CURL_PWM_SIGN,
}
# How many recent CAL lines the SDK keeps for UIs.
CAL_RECENT_LINES = 64
CAL_LINE_PREFIX = "CAL "
# EEPROM load/save messages printed by the firmware (no "CAL " prefix, so they never
# enter the CAL deque). Logged at info level so an ignored record is visible.
EEPROM_LINE_PREFIXES = ("Calibration ", "No calibration record", "Limits saved")

# --- Per-joint sensor state (SENS lines) ------------------------------------
# "SENS <ROLE> <NAME>:<c> ..." comes from quadrature-hall firmware (design 6.4):
#   ? unknown   P pot   e encoder, not homed   x encoder, home lost
#   s encoder homed, no stroke known (needs a full calibration)
#   E encoder homed with a stroke (ready for T)
# Old firmware and Rev 2 boards never print it; sensor_state() is then None.
SENS_LINE_PREFIX = "SENS "
SENSOR_STATE_CODES = "?PexsE"
# States in which a hip or knee has a usable position (pot, or encoder homed with a stroke).
SENSOR_READY_STATES = ("P", "E")
# States only a full calibration (SEEK + SWEEP) gives a position: "s" (homed, no saved
# stroke) and "?" (no sensor kind yet: a fresh flash, or an ignored, corrupt or invalid
# record). Curl only homes an encoder but never saves a stroke or a kind.
FULL_CALIBRATION_STATES = ("s", "?")
# Joint-name suffixes that Neutral may move: hips (HL) and knees (KL), never yaw (HY).
NEUTRAL_JOINT_SUFFIXES = ("HL", "KL")


def curl_pwm_sign(joint_name: str) -> Optional[int]:
    """-1 / +1 = sign of the PWM that curls this joint; None for yaw or unknown."""
    return JOINT_CURL_PWM_SIGN.get(str(joint_name).strip().upper()[-2:])


def is_hip_or_knee(joint_name: str) -> bool:
    """True for names ending HL (hip) or KL (knee); yaw (HY) and anything else are False."""
    return str(joint_name).strip().upper()[-2:] in NEUTRAL_JOINT_SUFFIXES


def curl_button(joint_name: str) -> Optional[str]:
    """Which jog button curls this joint: "retract" (hip), "extend" (knee) or None."""
    sign = curl_pwm_sign(joint_name)
    if sign is None:
        return None
    return "retract" if sign < 0 else "extend"


def validate_calibration_pwm(pwm: object) -> int:
    """Return ``pwm`` as an int in [CAL_PWM_MIN, CAL_PWM_MAX] or raise ValueError.

    Only real integers are accepted (no bool, no float, no string), so a bad value
    can never be silently turned into a different speed.
    """
    if isinstance(pwm, bool) or not isinstance(pwm, Integral):
        raise ValueError(
            f"calibration speed must be an integer {CAL_PWM_MIN}-{CAL_PWM_MAX}, got {pwm!r}"
        )
    value = int(pwm)
    if not CAL_PWM_MIN <= value <= CAL_PWM_MAX:
        raise ValueError(
            f"calibration speed must be {CAL_PWM_MIN}-{CAL_PWM_MAX}, got {value}"
        )
    return value


def parse_calibration_pwm(text: str) -> int:
    """Parse text typed by a person (GUI field). Digits only, then the same range check."""
    stripped = str(text).strip()
    if not stripped.isascii() or not stripped.isdigit():
        raise ValueError(
            f"calibration speed must be a whole number {CAL_PWM_MIN}-{CAL_PWM_MAX}"
        )
    return validate_calibration_pwm(int(stripped))


def _normalize_calibration_passes(
    passes: Union[None, str, Iterable[str]],
) -> tuple:
    if passes is None:
        return CAL_PASS_NAMES
    if isinstance(passes, str):
        items = [passes]
    else:
        try:
            items = list(passes)
        except TypeError:
            raise ValueError(
                f"calibration passes must be SEEK and/or SWEEP, got {passes!r}"
            ) from None
    chosen = set()
    for item in items:
        name = item.strip().upper() if isinstance(item, str) else None
        if name not in CAL_PASS_NAMES:
            raise ValueError(
                f"unknown calibration pass {item!r}; use {' or '.join(CAL_PASS_NAMES)}"
            )
        chosen.add(name)
    if not chosen:
        raise ValueError("calibration needs at least one pass (SEEK and/or SWEEP)")
    return tuple(name for name in CAL_PASS_NAMES if name in chosen)


def build_calibrate_command(
    pwm: Optional[int] = None, passes: Union[None, str, Iterable[str]] = None
) -> str:
    """Wire text (no newline) for the firmware ``C [SEEK|SWEEP] [pwm]`` command.

    ``C`` / ``C 128`` run SEEK then SWEEP; ``C SEEK 128`` / ``C SWEEP 128`` run one pass.
    Raises ValueError for a speed outside 110-150 or an unknown pass.
    """
    parts = ["C"]
    chosen = _normalize_calibration_passes(passes)
    if len(chosen) == 1:
        parts.append(chosen[0])
    if pwm is not None:
        parts.append(str(validate_calibration_pwm(pwm)))
    return " ".join(parts)


def cal_robot_finished(line: Optional[str]) -> bool:
    """True for the leader's end-of-run lines: CAL ROBOT DONE / FAIL / ABORTED / ERROR.

    BUSY is not an end: it means another run is still going. Per-board lines
    (``CAL FRONT PASS_DONE ...``) are never an end either.
    """
    words = line.split() if line else []
    return (
        len(words) >= 3
        and words[0] == "CAL"
        and words[1] == "ROBOT"
        and words[2] in ("DONE", "FAIL", "ABORTED", "ERROR")
    )


def cal_status_line(lines: Sequence[str]) -> Optional[str]:
    """The line a UI should show for a run: its robot end line once there is one, else the newest.

    The leader prints ``CAL ROBOT FAIL|ABORTED ...`` after sending H to the followers,
    so a follower's ``CAL <ROLE> ABORTED H`` can arrive after it; the robot line is
    still the run's result.
    """
    for line in reversed(lines):
        if cal_robot_finished(line):
            return line
    return lines[-1] if lines else None


def cal_line_state(line: Optional[str]) -> str:
    """Classify a CAL line for display: "fail", "done" or "running" ("" for None)."""
    if not line:
        return ""
    words = line.split()
    event = words[2] if len(words) > 2 else ""
    if event in ("FAIL", "ABORTED", "ERROR", "BUSY"):
        return "fail"
    if words[1:3] == ["ROBOT", "DONE"]:
        return "done"
    return "running"


class KrabbyMCUSDK:
    def __init__(self, target: McuSerialTarget | None = None, baud=DEFAULT_BAUD):
        self.target = target if target is not None else default_target()
        self.baud = baud
        self.ser = None
        self.running = False

        # Structured telemetry per joint
        self.joints: Dict[str, Optional[JointTelemetry]] = {}

        #   imu is None          — no IMU segment ever seen. Normal for
        #                          followers and for firmware that predates
        #                          the IMU segment
        # imu distinguishes unseen, invalid, and valid samples.
        self.imu: Optional[ImuTelemetry] = None
        # Latest BATT frame, emitted every leader telemetry tick. Each monitor
        # carries its own validity flag; None means no frame this session.
        self.battery: Optional[BatteryTelemetry] = None

        self.last_feedback_ts = None
        self.thread = None
        self._last_debug_log_ts = 0.0
        self.last_error = None
        self.last_cmd: Dict[str, Optional[float]] = {}
        self._last_ver_line: Optional[str] = None
        # Recent "CAL <ROLE> ..." status lines from the calibration routine. Written by
        # the reader thread, read by UIs, so every access holds _cal_lock. These lines
        # never reach the telemetry parser.
        self._cal_lock = threading.Lock()
        self._cal_lines: deque = deque(maxlen=CAL_RECENT_LINES)
        self._cal_line_count = 0
        # Latest SENS state code per joint name; empty until a SENS line arrives.
        # Written by the reader thread, read by UIs, so every access holds _sensor_lock.
        self._sensor_lock = threading.Lock()
        self.joint_sensor_state: Dict[str, str] = {}
        # Last time a telemetry line was seen per role prefix (FRONT/UNKWN/LEFT/RIGHT).
        self.role_last_seen: Dict[str, float] = {}

    def connect(self):
        try:
            # Open without toggling DTR so the leader board is not reset on connect.
            # A DTR reset would break three-board role election (followers stay in
            # their assigned roles and stop broadcasting SYNC, so a newly-reset
            # leader can never hear from them).
            self.ser = self.target.open(baud=self.baud)
            time.sleep(
                5.0
            )  # wait for boot + 3-board role election before starting reader
            self.running = True
            self.last_error = None
            # Drop any sample cached from a prior connection. This connect()
            # deliberately does not reset the board, so the hardware may well
            # still be fine - but a sample held across a disconnect is of
            # unknown age, and None has to keep meaning "nothing read this
            # session" rather than "something, once".
            self.imu = None
            self.battery = None
            # The board counters (and so every homed state) may have reset, so telemetry
            # and SENS states from a prior connection are not trusted either.
            self.joints.clear()
            with self._sensor_lock:
                self.joint_sensor_state.clear()
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info("Connected to %s", self.target)

            # On startup, immediately command the MCU to hold all joints
            # at their current positions so the legs don't drift or move
            # unexpectedly before the user issues a command.
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
                    line = raw.decode("utf-8").strip()
                except UnicodeDecodeError as e:
                    logger.warning(
                        "Decode error on serial line (target=%s, len=%d): %s raw=%s",
                        self.target,
                        len(raw),
                        e,
                        raw.hex(),
                    )
                    line = raw.decode("utf-8", errors="ignore").strip()
                except Exception:
                    logger.exception("Decode error")
                    continue
                if not line:
                    continue
                if _raw_rx_to_stderr():
                    print(f"[serial rx] {line}", file=sys.stderr, flush=True)
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug("serial rx: %s", line)
                if line.startswith(CAL_LINE_PREFIX):
                    # Calibration status, not telemetry: keep it away from the parser.
                    self._record_cal_line(line)
                    logger.info("[MCU] %s", line)
                    continue
                role = TelemetryFrame.role_from_line(line)
                if role is not None:
                    self._parse_telemetry_line(line)
                    self.last_feedback_ts = time.time()
                    self.role_last_seen[role] = self.last_feedback_ts
                elif line.startswith(SENS_LINE_PREFIX):
                    self._record_sensor_line(line)
                elif line.startswith("VER "):
                    self._last_ver_line = line
                elif line.startswith("ERR "):
                    # Firmware error lines (e.g. ERR FWD LEFT: line overflow…).
                    logger.error("[MCU] %s", line)
                elif (
                    "Krabby" in line
                    or "CAL" in line
                    or "Saved" in line
                    or line.startswith(EEPROM_LINE_PREFIXES)
                ):
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
                logger.exception("Reader loop error")
                self.last_error = exc
                self.running = False
                break

    def _record_cal_line(self, line: str) -> None:
        with self._cal_lock:
            self._cal_lines.append(line)
            self._cal_line_count += 1

    def recent_cal_lines(self, limit: Optional[int] = None) -> List[str]:
        """Newest-last copy of the recent ``CAL ...`` lines (at most ``limit`` of them)."""
        with self._cal_lock:
            lines = list(self._cal_lines)
        if limit is not None:
            lines = lines[-limit:] if limit > 0 else []
        return lines

    @property
    def latest_cal_line(self) -> Optional[str]:
        """The newest ``CAL ...`` line, or None if none arrived since the last clear."""
        with self._cal_lock:
            return self._cal_lines[-1] if self._cal_lines else None

    @property
    def cal_line_count(self) -> int:
        """Total CAL lines seen this session (keeps rising after the buffer wraps)."""
        with self._cal_lock:
            return self._cal_line_count

    def clear_cal_lines(self) -> None:
        with self._cal_lock:
            self._cal_lines.clear()

    def _record_sensor_line(self, line: str) -> None:
        """Store the state of every well-formed ``NAME:c`` token of a SENS line.

        Format: ``SENS <ROLE> <NAME>:<c> ...``. A token whose name is not alphanumeric
        or whose code is not one of ``?PexsE`` is skipped; the rest of the line is kept.
        """
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != SENS_LINE_PREFIX.strip():
            return
        parsed: Dict[str, str] = {}
        for token in tokens[2:]:  # tokens[1] is the board role
            name, sep, code = token.partition(":")
            if not sep or not name.isascii() or not name.isalnum():
                continue
            if len(code) != 1 or code not in SENSOR_STATE_CODES:
                continue
            parsed[name] = code
        if not parsed:
            return
        with self._sensor_lock:
            changed = {
                name: (self.joint_sensor_state.get(name), code)
                for name, code in parsed.items()
                if self.joint_sensor_state.get(name) != code
            }
            self.joint_sensor_state.update(parsed)
        for name, (old, new) in changed.items():
            logger.info("[MCU] sensor %s: %s -> %s", name, old or "-", new)

    def sensor_state(self, name: str) -> Optional[str]:
        """Latest SENS code for a joint, or None when no SENS line named it."""
        with self._sensor_lock:
            return self.joint_sensor_state.get(str(name).strip().upper())

    def sensor_states(self) -> Dict[str, str]:
        """Copy of every joint's latest SENS code (empty on firmware without SENS)."""
        with self._sensor_lock:
            return dict(self.joint_sensor_state)

    def position_ready(self, name: str) -> bool:
        """True when T may be sent to this joint: a hip or knee with a finite position.

        With no SENS line seen for the joint (old firmware, Rev 2 boards) the finite
        position decides. Otherwise the state must also be pot (P) or encoder ready (E).
        """
        name = str(name).strip().upper()
        if not is_hip_or_knee(name):
            return False
        jt = self.joints.get(name)
        if jt is None or not math.isfinite(jt.pos):
            return False
        state = self.sensor_state(name)
        return state is None or state in SENSOR_READY_STATES

    def neutral_targets(self, value: float = 0.5) -> tuple[Dict[str, float], List[str]]:
        """(targets, skipped): ``value`` for every ready hip and knee, and the hips and
        knees left out because they have no position yet. Yaws are in neither."""
        targets: Dict[str, float] = {}
        skipped: List[str] = []
        for _, names in JOINT_GROUP_NAMES:
            for name in names:
                if not is_hip_or_knee(name):
                    continue
                if self.position_ready(name):
                    targets[name] = value
                else:
                    skipped.append(name)
        return targets, skipped

    def _parse_telemetry_line(self, line: str):
        parsed = TelemetryFrame.parse_line(line)
        if parsed.battery is not None:
            self.battery = parsed.battery
        if parsed.imu is not None:
            self.imu = parsed.imu
        if not parsed.joints:
            return
        for jt in parsed.joints:
            self.joints[jt.name] = jt

        # Debug Log: FRONT / LEFT / RIGHT each on its own line
        now = time.time()
        if (
            logger.isEnabledFor(logging.DEBUG)
            and (now - self._last_debug_log_ts) >= 0.25
        ):
            for group_name, names in JOINT_GROUP_NAMES:
                parts = []
                for name in names:
                    jt = self.joints.get(name)
                    if jt:
                        parts.append(jt.format_compact(self.last_cmd.get(name)))
                if parts:
                    logger.debug("JOINTS %s %s", group_name, "; ".join(parts))
            if self.imu is not None:
                logger.debug("IMU %s", self.imu.format_compact())
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
        self.ser.write(cmd.encode("utf-8"))
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
        self.ser.write(cmd.encode("utf-8"))
        self.ser.flush()

    def send_command_jog(self, joint_name: str, pwm: int):
        """Send J<name> <pwm> (-255 to 255)"""
        if not self.ser or not self.ser.is_open:
            return
        pwm = max(-255, min(255, int(pwm)))
        cmd = f"J{joint_name} {pwm}\n"
        self.ser.write(cmd.encode("utf-8"))
        self.ser.flush()
        logger.debug("CMD -> %s", cmd.strip())

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

    def send_command_calibrate(
        self,
        pwm: Optional[int] = None,
        passes: Union[None, str, Iterable[str]] = None,
    ) -> Optional[str]:
        """Start robot calibration: ``C``, ``C 128``, ``C SEEK 128``, ``C SWEEP``, ...

        ``pwm`` is the calibration speed (integer 110-150; None = firmware default 128).
        ``passes`` is None (SEEK then SWEEP), "SEEK", "SWEEP", or an iterable of those.
        Invalid values raise ValueError before anything is written, even when the port
        is closed. Returns the command text sent, or None if there is no open port.
        """
        command = build_calibrate_command(pwm, passes)
        if not self.ser or not self.ser.is_open:
            return None
        self.clear_cal_lines()  # so a UI never shows a previous run's line as current
        self.ser.write((command + "\n").encode("utf-8"))
        self.ser.flush()
        logger.info("CMD -> AUTO-CALIBRATE (%s)", command)
        return command

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

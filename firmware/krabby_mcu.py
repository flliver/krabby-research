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

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            self.running = True
            self.last_error = None
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")

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
        cmd = " ".join(parts) + "\n"
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
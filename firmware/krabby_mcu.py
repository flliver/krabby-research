import os
import serial
import time
import threading
import logging
from serial.tools import list_ports

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("KrabbySDK")

def _default_port():
    """Resolve a sensible default serial port across platforms."""
    env_port = os.getenv("KRABBY_MCU_PORT")
    if env_port:
        return env_port

    for p in list_ports.comports():
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        if "arduino" in desc or "arduino" in manuf:
            return p.device

    return "COM4" if os.name == "nt" else "/dev/ttyACM0"


class KrabbyMCUSDK:
    def __init__(self, port=None, baud=115200):
        self.port = port or _default_port()
        self.baud = baud
        self.ser = None
        self.running = False

        # State for TWO motors
        self.latest_pos_left = None
        self.latest_pos_right = None
        self.latest_pwm_left = None
        self.latest_pwm_right = None
        self.latest_en = {"L_ENR": None, "L_ENL": None, "R_ENR": None, "R_ENL": None}

        # Safety Flags: (Triggered, Runaway)
        # Structure: {'L': {'safe': False, 'run': False}, 'R': ...}
        self.safety_status = {
            'L': {'safe': False, 'run': False},
            'R': {'safe': False, 'run': False}
        }
        self._last_safety_status = {
            'L': {'safe': False, 'run': False},
            'R': {'safe': False, 'run': False}
        }
        self.latest_is = {"L_ISR": 0, "L_ISL": 0, "R_ISR": 0, "R_ISL": 0}

        self.last_feedback_ts = None
        self.thread = None

    def connect(self, wait_feedback_timeout=2.0):
        """Establishes connection to the Arduino."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            self.running = True

            # Start background reader
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")

            if wait_feedback_timeout:
                if not self.wait_for_feedback(timeout=wait_feedback_timeout):
                    logger.error("No feedback detected from MCU.")
                    self.close()
                    return False

            return True
        except serial.SerialException as e:
            logger.error(f"Connection Failed: {e}")
            return False

    def _reader_loop(self):
        """Reads telemetry in background. Parses key/value fields in FB line; surfaces other lines."""
        while self.running and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                if line.startswith("FB:"):
                    self._parse_feedback_line(line)
                    self.last_feedback_ts = time.time()
                else:
                    # Only surface obvious MCU alerts; ignore malformed fragments
                    if "RUNAWAY" in line or "CRITICAL" in line:
                        logger.critical(f"MCU: {line}")
                    elif "Stall" in line or "Obstacle" in line:
                        logger.warning(f"MCU: {line}")
            except Exception:
                pass

    def _parse_feedback_line(self, line: str):
        """
        Parse FB line with flexible fields:
        FB:posL,posR,S:sL,sR,rL,rR,P:pwmL,pwmR,EN:LRLR,CTRL:...,IS:...
        Fields beyond positions are optional; order after positions is key-prefixed.
        """
        parts = line.split(',')
        if len(parts) >= 2:
            try:
                self.latest_pos_left = float(parts[0].split(':')[1])
                self.latest_pos_right = float(parts[1])
            except ValueError:
                pass

        # Temp copies to apply after parsing
        temp_safety = {
            'L': self.safety_status['L'].copy(),
            'R': self.safety_status['R'].copy()
        }

        i = 2
        while i < len(parts):
            token = parts[i]
            if token.startswith("S:"):
                vals = [token.split(':')[1]]
                j = i + 1
                while j < len(parts) and ':' not in parts[j] and len(vals) < 4:
                    vals.append(parts[j])
                    j += 1
                try:
                    temp_safety['L']['safe'] = bool(int(vals[0]))
                    temp_safety['R']['safe'] = bool(int(vals[1]))
                    temp_safety['L']['run'] = bool(int(vals[2]))
                    temp_safety['R']['run'] = bool(int(vals[3]))
                except (ValueError, IndexError):
                    pass
                i = j
                continue
            if token.startswith("P:"):
                vals = [token.split(':')[1]]
                j = i + 1
                if j < len(parts) and ':' not in parts[j]:
                    vals.append(parts[j])
                    j += 1
                try:
                    self.latest_pwm_left = int(vals[0])
                    if len(vals) > 1:
                        self.latest_pwm_right = int(vals[1])
                except ValueError:
                    self.latest_pwm_left = self.latest_pwm_right = None
                i = j
                continue
            if token.startswith("EN:"):
                bits = token.split(':')[1]
                if len(bits) >= 4:
                    self.latest_en["L_ENR"] = int(bits[0])
                    self.latest_en["L_ENL"] = int(bits[1])
                    self.latest_en["R_ENR"] = int(bits[2])
                    self.latest_en["R_ENL"] = int(bits[3])
                i += 1
                continue
            if token.startswith("IS:"):
                vals = token.split(':')[1].split(',')
                if len(vals) >= 4:
                    try:
                        self.latest_is["L_ISR"] = int(vals[0])
                        self.latest_is["L_ISL"] = int(vals[1])
                        self.latest_is["R_ISR"] = int(vals[2])
                        self.latest_is["R_ISL"] = int(vals[3])
                    except ValueError:
                        pass
                i += 1
                continue
            i += 1

        # Apply safety updates after full parse so IS is available for logs
        self.safety_status = temp_safety
        self._check_flag_changes()

        # Debug dump of current status (one line per FB when DEBUG enabled)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "FB L:%.3f R:%.3f | PWM L:%s R:%s | EN:%s | Safe L:%s R:%s Run L:%s R:%s",
                round(self.latest_pos_left or 0.0, 3),
                round(self.latest_pos_right or 0.0, 3),
                self.latest_pwm_left,
                self.latest_pwm_right,
                "".join(str(self.latest_en[k]) if self.latest_en[k] is not None else "?" for k in ["L_ENR", "L_ENL", "R_ENR", "R_ENL"]),
                int(self.safety_status['L']['safe']),
                int(self.safety_status['R']['safe']),
                int(self.safety_status['L']['run']),
                int(self.safety_status['R']['run']),
            )

    def _check_flag_changes(self):
        """Logs warnings/criticals when safety/runaway flags are set."""
        if self.safety_status['L']['run']:
            logger.critical("LEFT MOTOR RUNAWAY FLAG SET (encoder wiring/polarity/power).")
        if self.safety_status['R']['run']:
            logger.critical("RIGHT MOTOR RUNAWAY FLAG SET (encoder wiring/polarity/power).")
        if self.safety_status['L']['safe']:
            logger.warning(
                "Left Motor Obstacle Detected (Stall). IS(L_ISR,L_ISL,R_ISR,R_ISL)=%s",
                (self.latest_is["L_ISR"], self.latest_is["L_ISL"], self.latest_is["R_ISR"], self.latest_is["R_ISL"]),
            )
        if self.safety_status['R']['safe']:
            logger.warning(
                "Right Motor Obstacle Detected (Stall). IS(L_ISR,L_ISL,R_ISR,R_ISL)=%s",
                (self.latest_is["L_ISR"], self.latest_is["L_ISL"], self.latest_is["R_ISR"], self.latest_is["R_ISL"]),
            )

    def send_dual_yaw(self, left_val, right_val):
        """Sends target yaw [-1.0, 1.0]."""
        if not self.ser or not self.ser.is_open:
            return

        l_val = max(-1.0, min(1.0, left_val))
        r_val = max(-1.0, min(1.0, right_val))

        cmd = f"T {l_val:.4f} {r_val:.4f}\n"

        logger.info(f"Sending Command -> L: {l_val:.2f}, R: {r_val:.2f}")
        self.ser.write(cmd.encode('utf-8'))

    def get_positions(self):
        l = self.latest_pos_left if self.latest_pos_left is not None else 0.0
        r = self.latest_pos_right if self.latest_pos_right is not None else 0.0
        return (l, r)

    def wait_for_feedback(self, timeout=2.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.last_feedback_ts and (time.time() - self.last_feedback_ts) < 2.0:
                return True
            time.sleep(0.05)
        return False

    def wait_for_move(self, seconds):
        """Waits for a duration; verbose logging is handled in _parse_feedback_line."""
        start = time.time()
        while time.time() - start < seconds:
            time.sleep(0.1)

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()


# --- MAIN TEST BLOCK ---
if __name__ == "__main__":
    # Debug toggle: set env KRABBY_DEBUG=1 or pass --debug to see more logs
    import os
    import sys

    debug_enabled = os.getenv("KRABBY_DEBUG") == "1" or "--debug" in sys.argv
    if debug_enabled:
        logging.getLogger("KrabbySDK").setLevel(logging.DEBUG)
        # strip flag if present
        if "--debug" in sys.argv:
            sys.argv.remove("--debug")

    mcu = KrabbyMCUSDK()
    if mcu.connect():
        try:
            logger.info("--- TEST 1: Left Motor Left (-1.0) ---")
            mcu.send_dual_yaw(-1.0, 0.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Left Motor Right (1.0) ---")
            mcu.send_dual_yaw(1.00, 0.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Left Motor Center (0.0) ---")
            mcu.send_dual_yaw(0.0, 0.0)
            mcu.wait_for_move(2.0)

            logger.info("--- TEST 1: Right Motor Left (-1.0) ---")
            mcu.send_dual_yaw(0.0, -1.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Right Motor Right (1.0) ---")
            mcu.send_dual_yaw(0.0, 1.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Right Motor Center (0.0) ---")
            mcu.send_dual_yaw(0.0, 0.0)
            mcu.wait_for_move(2.0)

            logger.info("--- TEST 1: Motors Left/Right ---")
            mcu.send_dual_yaw(-1.0, 1.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Motors Right/Left (1.0) ---")
            mcu.send_dual_yaw(1.0, -1.0)
            mcu.wait_for_move(2.0)
            logger.info("--- TEST 1: Both Motor Center (0.0) ---")
            mcu.send_dual_yaw(0.0, 0.0)
            mcu.wait_for_move(2.0)


            logger.info("Test Complete.")

        except KeyboardInterrupt:
            logger.warning("Stopping by user request...")
            mcu.send_dual_yaw(0, 0)
        finally:
            mcu.close()

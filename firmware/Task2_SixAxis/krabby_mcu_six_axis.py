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
    env_port = os.getenv("KRABBY_MCU_PORT")
    if env_port:
        return env_port
    for p in list_ports.comports():
        if "arduino" in (p.description or "").lower() or "arduino" in (p.manufacturer or "").lower():
            return p.device
    return "COM5" if os.name == "nt" else "/dev/ttyACM0"


class KrabbyMCUSDK:
    def __init__(self, port=None, baud=115200):
        self.port = port or _default_port()
        self.baud = baud
        self.ser = None
        self.running = False

        # State for 6 Joints
        # Order: YawL, YawR, HipL, KneeL, HipR, KneeR
        self.latest_positions = [0.0] * 6

        # Calibration Data (Raw Analog 0-1023)
        self.latest_pots = [0, 0, 0, 0]  # HipL, KneeL, HipR, KneeR

        # Safety flags snapshot (order matches S: payload)
        # [yawL_safe, yawR_safe, yawL_run, yawR_run, hipL_safe, kneeL_safe, hipR_safe, kneeR_safe]
        self.flags = [0] * 8

        self.last_feedback_ts = None
        self.thread = None

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)
            self.running = True
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            logger.info(f"Connected to {self.port}")
            return True
        except Exception as e:
            logger.error(f"Connection Failed: {e}")
            return False

    def _reader_loop(self):
        while self.running and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                if line.startswith("FB:"):
                    self._parse_feedback(line)
                    self.last_feedback_ts = time.time()
                elif line.startswith("SAFETY:") or line.startswith("RUNAWAY:"):
                    logger.warning("MCU ALERT: %s", line)
                elif "POT:" in line:
                    # Sometimes POT data might come on a separate line in debug modes
                    pass
            except Exception:
                pass

    def _parse_feedback(self, line):
        # Format: FB:yL,yR,hL,kL,hR,kR,POT:p1,p2,p3,p4,AVG:a1...
        try:
            parts = line.split(',')

            # 1. Parse Positions (First 6 values)
            # parts[0] is "FB:0.123"
            self.latest_positions[0] = float(parts[0].split(':')[1])
            for i in range(1, 6):
                self.latest_positions[i] = float(parts[i])

            # 2. Parse Optional Tags (POT, AVG, etc.)
            for i in range(6, len(parts)):
                token = parts[i]

                # Raw Potentiometer Values (Crucial for Calibration)
                if token.startswith("POT:"):
                    # Expecting POT:p1,p2,p3,p4
                    # Current index i is POT:p1. Next 3 are p2, p3, p4
                    vals = [token.split(':')[1]]
                    # Grab next 3 parts if available
                    for k in range(1, 4):
                        if i+k < len(parts):
                            vals.append(parts[i+k])

                    self.latest_pots = [int(v) for v in vals]
                elif token.startswith("S:"):
                    vals = [token.split(':')[1]]
                    for k in range(1, 8):
                        if i+k < len(parts):
                            vals.append(parts[i+k])
                    try:
                        self.flags = [int(v) for v in vals]
                    except ValueError:
                        pass

        except (ValueError, IndexError):
            pass

        # Debug Log: Show Positions + Raw Pots for tuning
        if logger.isEnabledFor(logging.DEBUG):
            # Compact per-joint view: LHY/LHL/LKL/RHY/RHL/RKL
            flags = (self.flags + [0] * 8)[:8]  # pad if malformed
            # Yaw uses (pos, run, saf); linear uses (pos, pot, saf)
            joints = {
                "LHY": (round(self.latest_positions[0], 3), flags[2], flags[0]),
                "RHY": (round(self.latest_positions[1], 3), flags[3], flags[1]),
                "LHL": (round(self.latest_positions[2], 3), self.latest_pots[0], flags[4]),
                "LKL": (round(self.latest_positions[3], 3), self.latest_pots[1], flags[5]),
                "RHL": (round(self.latest_positions[4], 3), self.latest_pots[2], flags[6]),
                "RKL": (round(self.latest_positions[5], 3), self.latest_pots[3], flags[7]),
            }
            logger.debug("JOINTS (joint: {pos, pot/run, saf}): %s", joints)

    def send_command(self, yaw_l, yaw_r, hip_l, knee_l, hip_r, knee_r):
        """
        Sends 6-axis command vector.
        Yaw: [-1.0, 1.0]
        Linear: [0.0, 1.0]
        """
        if not self.ser or not self.ser.is_open:
            return

        # Clamp Yaw
        y_l = max(-1.0, min(1.0, yaw_l))
        y_r = max(-1.0, min(1.0, yaw_r))
        # Clamp Linear
        h_l = max(0.0, min(1.0, hip_l))
        k_l = max(0.0, min(1.0, knee_l))
        h_r = max(0.0, min(1.0, hip_r))
        k_r = max(0.0, min(1.0, knee_r))

        cmd = f"T {y_l:.3f} {y_r:.3f} {h_l:.3f} {k_l:.3f} {h_r:.3f} {k_r:.3f}\n"
        self.ser.write(cmd.encode('utf-8'))

        logger.info(
            f"CMD -> Yaw:{y_l:.2f}/{y_r:.2f} Hip:{h_l:.2f}/{h_r:.2f} Knee:{k_l:.2f}/{k_r:.2f}")

    def wait_for_move(self, seconds):
        time.sleep(seconds)

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()


# --- MAIN TEST ---
if __name__ == "__main__":
    import sys
    # Enable Debug to see POT values
    if "--debug" in sys.argv or True:  # Force debug for calibration visibility
        logger.setLevel(logging.DEBUG)

    mcu = KrabbyMCUSDK()
    if mcu.connect():
        try:
            logger.info("--- CALIBRATION MODE: Reading Potentiometers ---")
            logger.info(
                "Move the linear actuators by hand (if possible) to see ranges.")

            # Send neutral command (Yaw Center, Linear 50%)
            mcu.send_command(0, 0, 0.5, 0.5, 0.5, 0.5)

            # Loop to print values for client calibration
            for _ in range(20):
                # Just hold position and log values
                mcu.wait_for_move(0.5)

            logger.info("--- TEST: Gentle Sweep ---")
            # Extend Hips slightly
            mcu.send_command(0, 0, 0.7, 0.5, 0.7, 0.5)
            mcu.wait_for_move(2.0)

            # Retract Hips slightly
            mcu.send_command(0, 0, 0.3, 0.5, 0.3, 0.5)
            mcu.wait_for_move(2.0)

            logger.info("Test Complete. Check logs for min/max POT values.")

        except KeyboardInterrupt:
            mcu.send_command(0, 0, 0.5, 0.5, 0.5, 0.5)
        finally:
            mcu.close()

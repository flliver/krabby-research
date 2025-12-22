import os
import serial
import time
import threading
from serial.tools import list_ports


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
        self.latest_position = None
        self.latest_pwm = None
        self.last_feedback_ts = None
        self.last_command_ts = None
        self.position_at_command = None
        self.command_value = None
        self.thread = None

    def connect(self, wait_feedback_timeout=2.0):
        """Establishes connection to the Arduino and waits for initial feedback."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            self.running = True

            # Start background reader thread
            self.thread = threading.Thread(
                target=self._reader_loop, daemon=True)
            self.thread.start()
            print(f"[SDK] Connected to {self.port}")

            if wait_feedback_timeout is not None:
                if not self.wait_for_feedback(timeout=wait_feedback_timeout):
                    print(
                        "[SDK] No feedback detected (motor/MCU not sending data?).")
                    self.close()
                    return False

            return True
        except serial.SerialException as e:
            print(f"[SDK] Connection Failed: {e}")
            return False

    def _reader_loop(self):
        """Reads telemetry from Arduino in background."""
        while self.running and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith("FB:"):
                    # Parse "FB:0.123,PWM:50"
                    parts = line.split(',')
                    pos_part = parts[0].split(':')[1]
                    self.latest_position = float(pos_part)
                    # Parse PWM if present
                    if len(parts) > 1 and "PWM:" in parts[1]:
                        try:
                            self.latest_pwm = float(parts[1].split(':')[1])
                        except ValueError:
                            self.latest_pwm = None
                    self.last_feedback_ts = time.time()
            except Exception:
                pass

    def send_yaw(self, normalized_val):
        """Sends a target yaw [-1.0, 1.0] to the MCU."""
        if not self.ser or not self.ser.is_open:
            return

        # Clamp value
        val = max(-1.0, min(1.0, normalized_val))
        cmd = f"T {val:.4f}\n"
        # Track command issuance for movement detection
        self.last_command_ts = time.time()
        self.position_at_command = self.latest_position
        self.command_value = val
        print(f"[SDK] Sweeping to {val:.1f}...")
        self.ser.write(cmd.encode('utf-8'))

    def get_position(self):
        """Returns the latest normalized position (0.0 if none received yet)."""
        return self.latest_position if self.latest_position is not None else 0.0

        # Optional: callers can also check has_feedback() to see if data arrived.

    def has_feedback(self, max_age=2.0):
        """
        Returns True if we've seen feedback within the last `max_age` seconds.
        Use this to detect missing MCU/motor feedback.
        """
        if self.last_feedback_ts is None:
            return False
        return (time.time() - self.last_feedback_ts) <= max_age

    def wait_for_feedback(self, timeout=2.0, poll_interval=0.05):
        """
        Blocks until feedback is seen or timeout elapses. Returns True if feedback arrived.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.has_feedback(max_age=timeout):
                return True
            time.sleep(poll_interval)
        return False

    def _movement_unresponsive(self, min_delta=0.01, timeout=1.0, pwm_threshold=10.0):
        """
        Detects if a command was issued but no movement is observed.
        """
        if self.last_command_ts is None:
            return False  # no command sent yet

        if (time.time() - self.last_command_ts) < timeout:
            return False  # allow time to move

        if self.latest_position is None or self.position_at_command is None:
            return True  # no position feedback at all

        delta = abs(self.latest_position - self.position_at_command)
        if delta >= min_delta:
            return False

        # If PWM is small, controller may be idle; don't flag as unresponsive
        if self.latest_pwm is not None and abs(self.latest_pwm) < pwm_threshold:
            return False

        return True

    def wait(self, seconds, poll_interval=0.05, min_delta=0.01, pwm_threshold=10.0):
        """
        Waits for `seconds` while monitoring feedback/movement. Raises RuntimeError if
        no movement is detected after a command within the wait window.
        """
        start = time.time()
        while time.time() - start < seconds:
            if self.position_at_command is not None and self.latest_position is not None:
                if abs(self.latest_position - self.position_at_command) >= min_delta:
                    return

            if self._movement_unresponsive(min_delta=min_delta, timeout=seconds, pwm_threshold=pwm_threshold):
                raise RuntimeError(
                    "[SDK] Command sent but no movement detected (check motor/encoder power/wiring).")

            time.sleep(poll_interval)

        if self._movement_unresponsive(min_delta=min_delta, timeout=seconds, pwm_threshold=pwm_threshold):
            raise RuntimeError(
                "[SDK] Command sent but no movement detected (check motor/encoder power/wiring).")

    def close(self):
        self.running = False
        if self.ser:
            self.ser.close()


# --- Example Usage (can run this directly) ---
if __name__ == "__main__":
    mcu = KrabbyMCUSDK()
    if mcu.connect():
        try:
            mcu.send_yaw(-1.0)
            mcu.wait(2)

            mcu.send_yaw(1.0)
            mcu.wait(2)

            mcu.send_yaw(0.0)
            mcu.wait(2)

        except KeyboardInterrupt:
            print("Stopping...")
        except RuntimeError as e:
            print(str(e))
        finally:
            mcu.close()

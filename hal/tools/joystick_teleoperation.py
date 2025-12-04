"""Joystick teleoperation tool for controlling robot via navigation commands.

This tool provides keyboard and gamepad interfaces for sending navigation commands
to control any robot running with a ParkourInferenceClient.

Usage with keyboard:
    python -m hal.tools.joystick_teleoperation --keyboard

Usage with gamepad:
    python -m hal.tools.joystick_teleoperation --gamepad
"""

import argparse
import logging
import sys
import time
from typing import Optional

from hal.client.observation.types import NavigationCommand

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JoystickTeleoperation:
    """Joystick teleoperation for sending navigation commands.

    Can be used with keyboard or gamepad to control a robot via navigation commands.
    """

    def __init__(self, inference_client=None):
        """Initialize joystick teleoperation.

        Args:
            inference_client: ParkourInferenceClient instance to send commands to.
                If None, navigation commands will be logged but not sent.
        """
        self.inference_client = inference_client
        self.running = False
        self.vx = 0.0
        self.vy = 0.0
        self.yaw_rate = 0.0

    def run_keyboard(self) -> None:
        """Run keyboard input loop.

        Controls:
        - W/S: Forward/backward (vx)
        - A/D: Left/right (vy)
        - Q/E: Rotate left/right (yaw_rate)
        - Space: Stop
        - Esc: Exit
        """
        try:
            import termios
            import tty
        except ImportError:
            logger.error("termios not available (Windows?). Use --gamepad instead.")
            return

        logger.info("Keyboard teleoperation controls:")
        logger.info("  W/S: Forward/backward (vx)")
        logger.info("  A/D: Left/right (vy)")
        logger.info("  Q/E: Rotate left/right (yaw_rate)")
        logger.info("  Space: Stop")
        logger.info("  Esc: Exit")
        logger.info("")
        logger.info("Press any key to start...")

        # Set terminal to raw mode
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)

            self.running = True
            step = 0.1  # Velocity step size

            while self.running:
                # Read single character
                char = sys.stdin.read(1)

                if char == "\x1b":  # ESC
                    logger.info("Exiting...")
                    self.running = False
                    break
                elif char == "w" or char == "W":
                    self.vx = min(self.vx + step, 1.0)
                elif char == "s" or char == "S":
                    self.vx = max(self.vx - step, -1.0)
                elif char == "a" or char == "A":
                    self.vy = min(self.vy + step, 1.0)
                elif char == "d" or char == "D":
                    self.vy = max(self.vy - step, -1.0)
                elif char == "q" or char == "Q":
                    self.yaw_rate = min(self.yaw_rate + step, 1.0)
                elif char == "e" or char == "E":
                    self.yaw_rate = max(self.yaw_rate - step, -1.0)
                elif char == " ":  # Space
                    self.vx = 0.0
                    self.vy = 0.0
                    self.yaw_rate = 0.0

                # Send navigation command
                self._send_command()

                time.sleep(0.05)  # Small delay to avoid flooding

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run_gamepad(self) -> None:
        """Run gamepad/joystick input loop.

        Requires pygame library.

        Controls:
        - Left stick Y: Forward/backward (vx)
        - Left stick X: Left/right (vy)
        - Right stick X: Rotate (yaw_rate)
        - Start button: Exit
        """
        try:
            import pygame
        except ImportError:
            logger.error("pygame not available. Install with: pip install pygame")
            return

        logger.info("Initializing gamepad...")
        pygame.init()
        pygame.joystick.init()

        # Check for joysticks
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            logger.error("No joystick/gamepad found")
            return

        logger.info(f"Found {joystick_count} joystick(s)")
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        logger.info(f"Using joystick: {joystick.get_name()}")

        logger.info("Gamepad teleoperation controls:")
        logger.info("  Left stick Y: Forward/backward (vx)")
        logger.info("  Left stick X: Left/right (vy)")
        logger.info("  Right stick X: Rotate (yaw_rate)")
        logger.info("  Start button: Exit")

        self.running = True
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 7:  # Start button (may vary by controller)
                        logger.info("Exiting...")
                        self.running = False
                        break

            if not self.running:
                break

            # Read joystick axes
            # Left stick (axes 0, 1): vy, vx
            # Right stick (axes 2, 3): yaw_rate (use axis 2)
            self.vx = -joystick.get_axis(1)  # Invert Y axis (up = forward)
            self.vy = joystick.get_axis(0)  # X axis (left/right)
            self.yaw_rate = joystick.get_axis(2)  # Right stick X (rotation)

            # Apply dead zone
            dead_zone = 0.1
            if abs(self.vx) < dead_zone:
                self.vx = 0.0
            if abs(self.vy) < dead_zone:
                self.vy = 0.0
            if abs(self.yaw_rate) < dead_zone:
                self.yaw_rate = 0.0

            # Send navigation command
            self._send_command()

            # Limit update rate to 20 Hz (joystick polling)
            clock.tick(20)

        pygame.quit()

    def _send_command(self) -> None:
        """Send current navigation command to inference client."""
        nav_cmd = NavigationCommand.create_now(
            vx=self.vx,
            vy=self.vy,
            yaw_rate=self.yaw_rate,
        )

        if self.inference_client:
            self.inference_client.set_navigation_command(nav_cmd)

        logger.info(
            f"Nav command: vx={self.vx:+.2f}, vy={self.vy:+.2f}, "
            f"yaw_rate={self.yaw_rate:+.2f}"
        )

    def stop(self) -> None:
        """Stop the teleoperation interface."""
        self.running = False


def main():
    """Main entry point for joystick teleoperation tool.

    Note: This tool requires a ParkourInferenceClient instance to send commands.
    For standalone testing, commands are logged but not sent to any robot.

    To integrate with a running system, import this class and pass your
    ParkourInferenceClient instance to the constructor.
    """
    parser = argparse.ArgumentParser(
        description="Joystick teleoperation for robot navigation commands"
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Use keyboard input (default)",
    )
    parser.add_argument(
        "--gamepad",
        action="store_true",
        help="Use gamepad/joystick input (requires pygame)",
    )

    args = parser.parse_args()

    # Default to keyboard if no input method specified
    use_keyboard = args.keyboard or (not args.gamepad)

    logger.info("Starting joystick teleoperation in standalone mode")
    logger.info("Note: No inference client connected - commands will be logged only")
    logger.info("To control a robot, integrate this tool with ParkourInferenceClient")
    logger.info("")

    # Create teleoperation interface without inference client (standalone mode)
    teleop = JoystickTeleoperation(inference_client=None)

    try:
        # Run input loop
        if use_keyboard:
            teleop.run_keyboard()
        else:
            teleop.run_gamepad()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        logger.info("Joystick teleoperation closed")


if __name__ == "__main__":
    main()

"""Simple joystick client for sending navigation commands to hal.

This script provides a simple interface for joystick input to control the robot
via navigation commands (vx, vy, yaw_rate) sent to the HAL client.

Usage:
    python locomotion/jetson/joystick_client.py \
        --observation_endpoint tcp://localhost:6001 \
        --command_endpoint tcp://localhost:6002

For keyboard input (development/testing):
    python locomotion/jetson/joystick_client.py --keyboard

For gamepad/joystick input (requires pygame):
    python locomotion/jetson/joystick_client.py --gamepad
"""

import argparse
import logging
import sys
import time
from typing import Optional

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.observation.types import NavigationCommand

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class JoystickClient:
    """Simple joystick client for sending navigation commands."""

    def __init__(self, hal_client: HalClient):
        """Initialize joystick client.

        Args:
            hal_client: HAL client for sending navigation commands
        """
        self.hal_client = hal_client
        self.running = False

    def run_keyboard(self) -> None:
        """Run keyboard input loop (for development/testing).

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

        logger.info("Keyboard controls:")
        logger.info("  W/S: Forward/backward (vx)")
        logger.info("  A/D: Left/right (vy)")
        logger.info("  Q/E: Rotate left/right (yaw_rate)")
        logger.info("  Space: Stop")
        logger.info("  Esc: Exit")
        logger.info("Press any key to start...")

        # Set terminal to raw mode
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)

            self.running = True
            vx = 0.0
            vy = 0.0
            yaw_rate = 0.0
            step = 0.1  # Velocity step size

            while self.running:
                # Read single character
                char = sys.stdin.read(1)

                if char == "\x1b":  # ESC
                    logger.info("Exiting...")
                    self.running = False
                    break
                elif char == "w" or char == "W":
                    vx = min(vx + step, 1.0)
                elif char == "s" or char == "S":
                    vx = max(vx - step, -1.0)
                elif char == "a" or char == "A":
                    vy = min(vy + step, 1.0)
                elif char == "d" or char == "D":
                    vy = max(vy - step, -1.0)
                elif char == "q" or char == "Q":
                    yaw_rate = min(yaw_rate + step, 1.0)
                elif char == "e" or char == "E":
                    yaw_rate = max(yaw_rate - step, -1.0)
                elif char == " ":  # Space
                    vx = 0.0
                    vy = 0.0
                    yaw_rate = 0.0

                # Send navigation command
                nav_cmd = NavigationCommand.create_now(vx=vx, vy=vy, yaw_rate=yaw_rate)
                self.hal_client.set_navigation_command(nav_cmd)
                logger.info(f"Command: vx={vx:.2f}, vy={vy:.2f}, yaw_rate={yaw_rate:.2f}")

                time.sleep(0.05)  # Small delay to avoid flooding

        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def run_gamepad(self) -> None:
        """Run gamepad/joystick input loop.

        Requires pygame library.
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

        logger.info("Gamepad controls:")
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
            # Left stick (axes 0, 1): vx, vy
            # Right stick (axes 2, 3): yaw_rate (use axis 2)
            vx = -joystick.get_axis(1)  # Invert Y axis (up = forward)
            vy = joystick.get_axis(0)  # X axis (left/right)
            yaw_rate = joystick.get_axis(2)  # Right stick X (rotation)

            # Apply dead zone
            dead_zone = 0.1
            if abs(vx) < dead_zone:
                vx = 0.0
            if abs(vy) < dead_zone:
                vy = 0.0
            if abs(yaw_rate) < dead_zone:
                yaw_rate = 0.0

            # Send navigation command
            nav_cmd = NavigationCommand.create_now(vx=vx, vy=vy, yaw_rate=yaw_rate)
            self.hal_client.set_navigation_command(nav_cmd)

            # Limit update rate to 20 Hz (joystick polling)
            clock.tick(20)

        pygame.quit()

    def stop(self) -> None:
        """Stop the joystick client."""
        self.running = False


def main():
    """Main entry point for joystick client."""
    parser = argparse.ArgumentParser(description="Joystick client for HAL navigation commands")
    parser.add_argument(
        "--observation_endpoint",
        type=str,
        default="tcp://localhost:6001",
        help="Observation endpoint (default: tcp://localhost:6001)",
    )
    parser.add_argument(
        "--command_endpoint",
        type=str,
        default="tcp://localhost:6002",
        help="Command endpoint (default: tcp://localhost:6002)",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Use keyboard input (for development/testing)",
    )
    parser.add_argument(
        "--gamepad",
        action="store_true",
        help="Use gamepad/joystick input (requires pygame)",
    )

    args = parser.parse_args()

    # Default to keyboard if no input method specified
    use_keyboard = args.keyboard or (not args.gamepad)

    # Create HAL client config
    config = HalClientConfig.from_endpoints(
        observation_endpoint=args.observation_endpoint,
        command_endpoint=args.command_endpoint,
    )

    # Initialize HAL client
    hal_client = HalClient(config)
    try:
        hal_client.initialize()
        logger.info("HAL client initialized")

        # Create joystick client
        joystick_client = JoystickClient(hal_client)

        # Run input loop
        if use_keyboard:
            joystick_client.run_keyboard()
        else:
            joystick_client.run_gamepad()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        hal_client.close()
        logger.info("Joystick client closed")


if __name__ == "__main__":
    main()


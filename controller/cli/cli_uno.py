#!/usr/bin/env python3
"""CLI entry point for krabby-uno: gamepad → Krabby HAL (real hardware path).

Connects to the Krabby HAL server via ZMQ TCP and sends joint commands from
the gamepad. Run the HAL server first in another terminal.

Usage:
  krabby-uno
  krabby-uno --observation_endpoint tcp://localhost:6001 --command_endpoint tcp://localhost:6002
  krabby-uno --InputController 0 --rate 50
"""

import argparse
import logging
import signal
import sys
import time

from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from hal.client.config import HalClientConfig
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _initialize_pygame() -> None:
    """Initialize Pygame in main thread for InputController (e.g. macOS)."""
    try:
        import pygame
        if not pygame.get_init():
            pygame.init()
            logger.info("Pygame initialized in main thread")
    except Exception:
        pass


def _pump_pygame_events(last_pump_time: list) -> None:
    """Pump Pygame events so joystick state updates (main thread)."""
    try:
        import pygame
        if pygame.get_init():
            t = time.time()
            if t - last_pump_time[0] > 0.016:
                pygame.event.pump()
                last_pump_time[0] = t
    except Exception:
        pass


def main() -> int:
    """Run control loop client until Ctrl+C."""
    parser = argparse.ArgumentParser(
        description="krabby-uno: gamepad → Krabby HAL (ZMQ TCP). Start HAL server first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--observation_endpoint",
        type=str,
        default="tcp://localhost:6001",
        help="HAL observation endpoint (client connects)",
    )
    parser.add_argument(
        "--command_endpoint",
        type=str,
        default="tcp://localhost:6002",
        help="HAL command endpoint (client connects)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        metavar="ID",
        dest="device_id",
        help="Gamepad device ID (from python -m controller.input --list). Default: first device.",
    )
    parser.add_argument(
        "--InputController",
        type=int,
        default=None,
        metavar="ID",
        dest="input_controller_id",
        help="Alias for --device-id (gamepad device ID).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Input controller update rate (Hz)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=("hex", "go2"),
        default="hex",
        help="HAL joint topology for gamepad commands (must match hal.server.jetson --robot)",
    )
    args = parser.parse_args()

    device_id = args.device_id if args.device_id is not None else args.input_controller_id

    _initialize_pygame()

    hal_client_config = HalClientConfig(
        observation_endpoint=args.observation_endpoint,
        command_endpoint=args.command_endpoint,
    )
    robot_definition = UNITREE_GO2_DEFINITION if args.robot == "go2" else KRABBY_HEX_DEFINITION
    control_loop_config = ControlLoopConfig(
        mode=ControlMode.INPUT_CONTROLLER_KRABBY,
        hal_client_config=hal_client_config,
        input_controller_device_id=device_id,
        input_controller_update_rate_hz=args.rate,
        krabby_gamepad_robot_definition=robot_definition,
    )

    control_loop = ControlLoop(control_loop_config)
    control_loop.start()

    logger.info(
        "Control loop client started (INPUT_CONTROLLER_KRABBY). "
        "observation=%s, command=%s. Press Ctrl+C to stop.",
        args.observation_endpoint,
        args.command_endpoint,
    )
    logger.info(
        "Gamepad: LT/LB/LS/RS/RT/RB = legs; left stick Y/X = hip/knee; right stick Y = hip yaw"
    )

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_pump = [0.0]
    try:
        while running:
            _pump_pygame_events(last_pump)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        control_loop.stop()
        logger.info("Control loop client stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI entry point for krabby-uno-sim: gamepad → IsaacSim HAL (simulation path).

Connects to the IsaacSim HAL server via ZMQ TCP and sends joint commands from
the gamepad. Start the IsaacSim HAL server first; see controller/scripts/isaac/isaacsim_demo_runbook.md.

Usage:
  krabby-uno-sim --quad   # 12-joint quad/Go2 sim
  krabby-uno-sim --hex    # 18-joint hexapod (crab hex)
  krabby-uno-sim          # same as --hex (default)
  krabby-uno-sim --observation_endpoint tcp://127.0.0.1:5555 --command_endpoint tcp://127.0.0.1:5556
  krabby-uno-sim --InputController 0 --rate 50
  krabby-uno-sim --gamepad-wait 600   # wait up to 10 min for Pro Controller (default)
  krabby-uno-sim --gamepad-wait 0     # exit immediately if no gamepad (legacy behavior)
"""

import argparse
import logging
import signal
import sys
import time

from controller.control_loop import ControlLoop, ControlLoopConfig, ControlMode
from controller.input import InputController
from controller.robot_definition_quad import KRABBY_QUAD_DEFINITION
from hal.client.config import HalClientConfig
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION

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
    """Run control loop client (IsaacSim HAL) until Ctrl+C."""
    parser = argparse.ArgumentParser(
        description="krabby-uno-sim: gamepad → IsaacSim HAL (ZMQ TCP). Start IsaacSim HAL server first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--observation_endpoint",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="HAL observation endpoint (client connects)",
    )
    parser.add_argument(
        "--command_endpoint",
        type=str,
        default="tcp://127.0.0.1:5556",
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
        "--debug",
        action="store_true",
        help="Enable debug logging (gamepad mapper and HAL client)",
    )
    robot_group = parser.add_mutually_exclusive_group()
    robot_group.add_argument(
        "--quad",
        action="store_true",
        help="Use 12-joint quad (for Isaac Sim demo with Go2/quad task)",
    )
    robot_group.add_argument(
        "--hex",
        action="store_true",
        help="Use 18-joint hexapod (crab hex; default)",
    )
    parser.add_argument(
        "--mapper-hip-scale",
        type=float,
        default=None,
        metavar="RAD_PER_UNIT",
        help="Override hip up/down scale (radians per unit LY) for gamepad mapper",
    )
    parser.add_argument(
        "--mapper-knee-scale",
        type=float,
        default=None,
        metavar="RAD_PER_UNIT",
        help="Override knee out/in scale (radians per unit LX) for gamepad mapper",
    )
    parser.add_argument(
        "--mapper-hip-yaw-scale",
        type=float,
        default=None,
        metavar="RAD_PER_UNIT",
        help="Override hip yaw scale (radians per unit RY) for gamepad mapper",
    )
    parser.add_argument(
        "--connection-timeout",
        type=float,
        default=15.0,
        metavar="SECONDS",
        help="Seconds to wait for HAL server connection before exiting (0 = do not wait)",
    )
    parser.add_argument(
        "--gamepad-wait",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="Seconds to wait for a gamepad/Pro Controller to appear before exiting (0 = exit immediately)",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("controller").setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled for gamepad and mapper")

    device_id = args.device_id if args.device_id is not None else args.input_controller_id

    # Mapper scaling: same defaults as Jetson (0.3, 0.3, 0.2) for parity; overrides via --mapper-*.
    hip_scale = 0.3
    knee_scale = 0.3
    hip_yaw_scale = 0.2
    if args.mapper_hip_scale is not None:
        hip_scale = args.mapper_hip_scale
    if args.mapper_knee_scale is not None:
        knee_scale = args.mapper_knee_scale
    if args.mapper_hip_yaw_scale is not None:
        hip_yaw_scale = args.mapper_hip_yaw_scale

    _initialize_pygame()

    devices = InputController.list_devices()
    if not devices:
        if args.gamepad_wait <= 0:
            logger.error(
                "No gamepad/joystick detected. Connect a controller and try again. "
                "Run 'python -m controller.input --list' to verify."
            )
            return 1
        poll_interval_s = 2.5
        deadline = time.time() + args.gamepad_wait
        logger.info(
            "No gamepad/joystick detected. Connect Pro Controller (or other gamepad). "
            "Waiting up to %.0fs (--gamepad-wait).",
            args.gamepad_wait,
        )
        while time.time() < deadline:
            time.sleep(poll_interval_s)
            devices = InputController.list_devices()
            if devices:
                logger.info("Gamepad detected: %d device(s) available.", len(devices))
                break
        if not devices:
            logger.error(
                "No gamepad/joystick detected after waiting %.0fs. "
                "Connect a controller and try again, or increase --gamepad-wait.",
                args.gamepad_wait,
            )
            return 1
    if device_id is not None and (device_id < 0 or device_id >= len(devices)):
        logger.error(
            "Device ID %s is not available. Found %d controller(s). "
            "Run 'python -m controller.input --list' to see devices.",
            device_id,
            len(devices),
        )
        return 1

    hal_client_config = HalClientConfig(
        observation_endpoint=args.observation_endpoint,
        command_endpoint=args.command_endpoint,
    )
    control_loop_config = ControlLoopConfig(
        mode=ControlMode.INPUT_CONTROLLER_ISAACSIM,
        hal_client_config=hal_client_config,
        input_controller_device_id=device_id,
        input_controller_update_rate_hz=args.rate,
         mapper_hip_up_down_scale=hip_scale,
         mapper_knee_out_in_scale=knee_scale,
         mapper_hip_yaw_scale=hip_yaw_scale,
        isaacsim_robot_definition=KRABBY_QUAD_DEFINITION if args.quad else KRABBY_HEX_DEFINITION,
    )

    control_loop = ControlLoop(control_loop_config)
    control_loop.start()

    if args.connection_timeout > 0:
        logger.info("Waiting for HAL server (timeout=%.0fs)...", args.connection_timeout)
        if not control_loop.wait_for_hal_server(timeout_s=args.connection_timeout):
            logger.error(
                "Could not connect to HAL server: no observation received within %.0fs. "
                "Is the Isaac Sim HAL server running (e.g. ./scripts/run_isaac_hal_server.sh)?",
                args.connection_timeout,
            )
            control_loop.stop()
            return 1

    logger.info(
        "Control loop client started (INPUT_CONTROLLER_ISAACSIM). "
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

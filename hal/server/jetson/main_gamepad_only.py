"""HAL server entry point for gamepad-only operation.

Runs the Jetson HAL server with TCP ZMQ endpoints so that a separate
krabby-uno container (or any HAL client) can connect and send joint
commands without requiring a model checkpoint.

Usage:
    krabby-hal-server-gamepad-only [--control_rate HZ]
                                   [--observation_bind ENDPOINT]
                                   [--command_bind ENDPOINT]
"""

import argparse
import logging
import signal
import sys
import time

from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jetson HAL server for gamepad-only operation (no inference checkpoint required)"
    )
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument(
        "--observation_bind",
        type=str,
        default="tcp://*:6001",
        help="HAL observation bind endpoint (default: tcp://*:6001)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default="tcp://*:6002",
        help="HAL command bind endpoint (default: tcp://*:6002)",
    )
    args = parser.parse_args()

    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    robot_definition = KRABBY_HEX_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    try:
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        hal_server = JetsonHalServer(
            hal_server_config,
            observation_dimensions=observation_dimensions,
            action_dim=model_definition.action_dim,
            robot_definition=robot_definition,
        )
        hal_server.initialize()
        hal_server.initialize_sensors()
        hal_server.initialize_actuators()

        logger.info(
            "HAL server started in gamepad-only mode "
            "(observation=%s, command=%s). Run `krabby uno` to connect.",
            args.observation_bind,
            args.command_bind,
        )

        period_s = 1.0 / args.control_rate
        lag_warning_count = 0

        try:
            while running:
                loop_start_ns = time.time_ns()

                hal_server.set_observation()
                command = hal_server.get_joint_command(timeout_ms=1)
                if command is not None:
                    hal_server.apply_command(command)

                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    lag_warning_count = 0
                elif loop_duration_s > period_s * 1.1:
                    lag_warning_count += 1
                    if lag_warning_count == 1 or lag_warning_count % 100 == 0:
                        logger.warning(
                            "Loop unable to keep up! %.2fms > %.2fms target (count=%d)",
                            loop_duration_s * 1000.0,
                            period_s * 1000.0,
                            lag_warning_count,
                        )
                else:
                    lag_warning_count = 0

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    except Exception as e:
        logger.error("Failed to run gamepad-only HAL server: %s", e, exc_info=True)
        sys.exit(1)

    finally:
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

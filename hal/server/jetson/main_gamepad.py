"""Jetson HAL server, gamepad mode.

Binds the HAL over TCP so a separate ``krabby-uno`` process can send gamepad joint
commands. Cameras are skipped (no observation consumer, and bench rigs often have
no ZED) and neither the policy (torch) nor the teleop stack is imported.
"""

import argparse
import logging
import signal
import sys

from hal.server.jetson.runtime import (
    add_common_args,
    configure_logging,
    create_hal_server,
    run_control_loop,
    select_robot,
    start_data_collector,
    stop_thread,
    warn_if_mcu_missing,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson HAL server for gamepad control (krabby-uno client)")
    add_common_args(parser)
    parser.add_argument("--control-source", choices=["gamepad"], default="gamepad")
    # Fleet launch configs pass these for every control source; they have no effect here.
    parser.add_argument("--teleop-ip", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--teleop-control-echo", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.teleop_ip:
        logger.info("control-source=gamepad: ignoring --teleop-ip (no in-process teleop client)")

    robot_definition = select_robot(args.robot)
    if not robot_definition.get_mcu_joints():
        parser.error(
            f"--control-source gamepad requires a robot with MCU joints; "
            f"'{args.robot}' has none. Use --robot hex for the Krabby hexapod."
        )

    observation_bind = args.observation_bind or "tcp://*:6001"
    command_bind = args.command_bind or "tcp://*:6002"

    running = True

    def signal_handler(_signum, _frame):
        nonlocal running
        running = False  # no logging — logger is not async-signal-safe

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    collector_stop = collector_thread = None

    try:
        hal_server = create_hal_server(robot_definition, observation_bind, command_bind)
        warn_if_mcu_missing(hal_server)
        logger.info("HAL server initialized")
        logger.info(
            "Gamepad mode active: HAL bound at observation=%s, command=%s. "
            "`krabby run` starts the krabby-uno client automatically; to connect a "
            "separate client, run `krabby-uno`.",
            observation_bind,
            command_bind,
        )

        collector_stop, collector_thread = start_data_collector(
            args, observation_bind, command_bind, hal_server.get_transport_context()
        )

        run_control_loop(hal_server, lambda: running)

    except Exception as e:
        logger.error(f"Failed to run Jetson HAL server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        stop_thread(collector_stop, collector_thread, "HalDataCollector")
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

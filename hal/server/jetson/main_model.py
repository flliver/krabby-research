"""Jetson HAL server, model mode (control sources ``inference`` and ``portal``).

- inference: an in-process ParkourInferenceClient drives the joints from a policy
  checkpoint; with --teleop-ip, portal video runs alongside it.
- portal: joint commands arrive from the teleop portal's WebRTC data channel.

Both use inproc HAL endpoints shared with their in-process clients and initialize
the RGB-D cameras for observations and teleop video.
"""

import argparse
import logging
import signal
import sys
import threading

from hal.client.config import HalClientConfig
from hal.server.jetson.runtime import (
    CONTROL_RATE_HZ,
    add_common_args,
    configure_logging,
    create_hal_server,
    run_control_loop,
    select_robot,
    start_data_collector,
    stop_thread,
    warn_if_mcu_missing,
)
from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.policy_interface import ModelWeights
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

logger = logging.getLogger(__name__)

INPROC_OBSERVATION_ENDPOINT = "inproc://hal_observation"
INPROC_COMMAND_ENDPOINT = "inproc://hal_commands"


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson HAL server with policy inference or portal teleop")
    add_common_args(parser)
    parser.add_argument(
        "--control-source",
        type=str,
        default="portal",
        choices=["portal", "inference"],
        help=(
            "'portal' uses WebRTC data-channel commands; 'inference' uses the policy inference client. "
            "Use --control-source gamepad for krabby-uno gamepad control."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required when --control-source inference)",
    )
    parser.add_argument(
        "--teleop-ip",
        type=str,
        default=None,
        metavar="HOST",
        help=(
            "Teleop portal host (IP or hostname). Enables outbound WebRTC signaling to "
            "ws://HOST:9000/ws/robot using HAL RGB-D cameras. Required for --control-source portal. "
            "Optional for inference (video alongside policy)."
        ),
    )
    parser.add_argument(
        "--teleop-control-echo",
        action="store_true",
        help=(
            "Echo the input controller's current state onto the telemetry channel as "
            "`last_control`, so a test harness can confirm a control message was actually "
            "applied. Off by default -- no operator-facing feature reads it; only set this "
            "on a bench used for automated teleop control round-trip verification."
        ),
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.control_source == "inference" and not args.checkpoint:
        parser.error("--checkpoint is required when --control-source inference")
    if args.control_source == "portal" and not args.teleop_ip:
        parser.error("--control-source portal requires --teleop-ip")

    teleop_enabled = args.teleop_ip is not None
    robot_definition = select_robot(args.robot)
    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)

    observation_bind = args.observation_bind or INPROC_OBSERVATION_ENDPOINT
    command_bind = args.command_bind or INPROC_COMMAND_ENDPOINT

    running = True

    def signal_handler(_signum, _frame):
        nonlocal running
        running = False  # no logging — logger is not async-signal-safe

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    parkour_client = None
    collector_stop = collector_thread = None
    teleop_stop: threading.Event | None = None
    teleop_thread: threading.Thread | None = None

    try:
        hal_server = create_hal_server(robot_definition, observation_bind, command_bind)
        # Cameras feed RGB-D observations (inference) and teleop video.
        hal_server.initialize_cameras()
        warn_if_mcu_missing(hal_server)

        teleop_settings = None
        teleop_sensor_ids = None
        if teleop_enabled:
            # teleop.edge is not installed in the production locomotion image, so import lazily.
            from teleop.edge.robot_settings import build_teleop_edge_settings

            # Bootstrap HAL poll until the browser sends ``catalog_ids`` on hello/offer (portal viewer).
            teleop_sensor_ids = [hal_server._primary_catalog_id]
            teleop_settings = build_teleop_edge_settings(
                host_or_url=args.teleop_ip, control_echo_enabled=args.teleop_control_echo
            )
            if not hal_server._hal_rgbd_cameras:
                logger.warning(
                    "--teleop-ip: no HAL RGB-D cameras opened after initialize_cameras(); "
                    "teleop signaling still starts (video will be black until cameras work)",
                )

        logger.info("HAL server initialized")

        transport_context = hal_server.get_transport_context()
        teleop_send_commands = args.control_source == "portal"

        if teleop_settings is not None:
            from hal.server.teleop_portal_signaling import start_hal_teleop_signaling_thread

            teleop_stop = threading.Event()
            teleop_thread = start_hal_teleop_signaling_thread(
                HalClientConfig(
                    observation_endpoint=observation_bind,
                    command_endpoint=command_bind if teleop_send_commands else None,
                ),
                transport_context,
                hal_server.get_sensor_interface(),
                stop_event=teleop_stop,
                bootstrap_sensor_catalog_ids=teleop_sensor_ids,
                teleop_edge_settings=teleop_settings,
                robot_definition=robot_definition,
                send_hal_commands=teleop_send_commands,
            )
            logger.info(
                "Teleop outbound signaling started: mode=%s url=%s reconnect_s=%.1f "
                "(bootstrap catalog ids=%s; viewer may override via signaling ``catalog_ids``); "
                "webrtc_hal_commands=%s",
                teleop_settings.mode,
                teleop_settings.server_signaling_ws_url,
                teleop_settings.server_reconnect_s,
                teleop_sensor_ids,
                teleop_send_commands,
            )

        if args.control_source == "inference":
            parkour_client = ParkourInferenceClient(
                hal_client_config=HalClientConfig(
                    observation_endpoint=observation_bind,
                    command_endpoint=command_bind,
                ),
                model_weights=ModelWeights(
                    checkpoint_path=args.checkpoint,
                    observation_dimensions=observation_dimensions,
                    action_dim=model_definition.action_dim,
                ),
                observation_dimensions=observation_dimensions,
                robot_definition=robot_definition,
                control_rate=CONTROL_RATE_HZ,
                device="cuda",
                transport_context=transport_context,
            )
            parkour_client.initialize()
            logger.info("Parkour inference client initialized")
            parkour_client.start_thread(running_flag=lambda: running)
            if teleop_enabled:
                logger.info(
                    "Teleop video active; inference commands use source=inference; operator overrides when portal sends",
                )
        else:
            logger.info(
                "Portal controller mode active: waiting for teleop control data-channel "
                "commands on HAL command socket (%s)",
                command_bind,
            )

        collector_stop, collector_thread = start_data_collector(
            args, observation_bind, command_bind, transport_context
        )

        run_control_loop(hal_server, lambda: running)

    except Exception as e:
        logger.error(f"Failed to run Jetson HAL server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        stop_thread(teleop_stop, teleop_thread, "Teleop HTTP")
        stop_thread(collector_stop, collector_thread, "HalDataCollector")
        if parkour_client:
            parkour_client.close()
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

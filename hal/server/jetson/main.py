"""Entry point for Jetson HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together on the robot.

For standalone server mode (client runs separately), use TCP endpoints instead.

Gathers observations from real hardware (camera, sensors), runs inference,
and applies commands to control the robot actuators.
"""

import argparse
import logging
from pathlib import Path
import signal
import sys
import threading
import time

from data_collection.collector import start_collector_thread
from data_collection.collector_settings import build_data_collector_config
from data_collection.config import load_config
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.teleop_portal_signaling import start_hal_teleop_signaling_thread
from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.policy_interface import ModelWeights
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
from teleop.edge.robot_settings import build_teleop_edge_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
CONTROL_RATE_HZ = 100.0
OBSERVATION_ENDPOINT = "inproc://hal_observation"
COMMAND_ENDPOINT = "inproc://hal_commands"


def main():
    """Main entry point for Jetson production deployment."""
    parser = argparse.ArgumentParser(description="Jetson production deployment with HAL server and inference")

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required when --control-source inference)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level for this process (default: INFO)",
    )

    parser.add_argument(
        "--data-collector-output-dir",
        type=str,
        default=None,
        help=(
            "Enable second HalClient + rosbag2 (mcap) recording and write bags to this directory. "
            "Mount this path to host storage for persistence."
        ),
    )
    parser.add_argument(
        "--data-collector-config",
        type=str,
        default=None,
        help=(
            "Optional YAML config for collector settings (rates/topics/rotation/quota/output_dir). "
            "HAL inproc endpoints are always enforced by this entrypoint."
        ),
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help=(
            "Run teleop WebRTC in-process: RGB from HAL RGB-D cameras. "
            "Configure signaling URL, mode, and optional sensor ids in teleop/edge/robot_settings.py."
        ),
    )
    parser.add_argument(
        "--control-source",
        type=str,
        default="portal",
        choices=["portal", "inference"],
        help=(
            "Primary command source for actuator control. "
            "'portal' uses WebRTC data-channel commands; 'inference' uses policy inference client."
        ),
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="hex",
        choices=["hex", "go2"],
        help="Robot definition to use (default: hex).",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.setLevel(getattr(logging, args.log_level))
    # aioice (aiortc ICE) logs every candidate-pair transition at INFO; keep operator logs readable.
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)

    if args.control_source == "inference" and not args.checkpoint:
        parser.error("--checkpoint is required when --control-source inference")
    if args.control_source == "portal" and not args.teleop:
        logger.info("control-source=portal selected: enabling --teleop automatically")
        args.teleop = True

    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    if args.robot == "hex":
        robot_definition = KRABBY_HEX_DEFINITION
        logger.info("Using Krabby Hex robot definition (default)")
    else:
        # Keep explicit option for Unitree-Go2 checkpoints.
        robot_definition = UNITREE_GO2_DEFINITION
        logger.info("Using Unitree Go2 robot definition")
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)

    # Running flag for graceful shutdown
    running = True

    def signal_handler(sig, frame):
        """Handle interrupt signals."""
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    parkour_client = None
    collector_stop: threading.Event | None = None
    collector_thread: threading.Thread | None = None
    teleop_stop: threading.Event | None = None
    teleop_thread: threading.Thread | None = None
    teleop_sensor_ids: list[str] | None = None

    try:
        # Create HAL server config (inproc for production)
        hal_server_config = HalServerConfig(
            observation_bind=OBSERVATION_ENDPOINT,
            command_bind=COMMAND_ENDPOINT,
        )

        # Create and initialize HAL server
        hal_server = JetsonHalServer(
            hal_server_config,
            observation_dimensions=observation_dimensions,
            action_dim=model_definition.action_dim,
            robot_definition=robot_definition,
        )
        hal_server.initialize()

        # Initialize hardware (camera, sensors, actuators)
        hal_server.initialize_sensors()
        hal_server.initialize_actuators()
        hal_server.initialize_cameras()
        if args.teleop:
            # Bootstrap HAL poll until the browser sends ``catalog_ids`` on hello/offer (portal viewer).
            teleop_sensor_ids = [hal_server._primary_catalog_id]
            _teleop_st = build_teleop_edge_settings()
            if not _teleop_st.agent_enabled:
                logger.error(
                    "--teleop requires agent mode with a non-empty signaling URL: set "
                    "TELEOP_EDGE_MODE=\"agent\" and SERVER_SIGNALING_WS_URL in "
                    "teleop/edge/robot_settings.py; teleop signaling not started",
                )
                teleop_sensor_ids = None
            elif not hal_server._hal_rgbd_cameras:
                logger.warning(
                    "--teleop: no HAL RGB-D cameras opened after initialize_cameras(); "
                    "teleop signaling still starts (video will be black until cameras work)",
                )

        logger.info("HAL server initialized")

        # Get transport context for inproc connections
        transport_context = hal_server.get_transport_context()

        # Shared ZMQ context for inproc clients below. Inference and portal teleop both PUSH to COMMAND_ENDPOINT.
        teleop_send_commands = args.control_source == "portal"

        # Inference/ParkourInferenceClient: SUB observations; PUSH joint commands (source=inference).
        inference_hal_client_config = HalClientConfig(
            observation_endpoint=OBSERVATION_ENDPOINT,
            command_endpoint=COMMAND_ENDPOINT,
        )
        # Teleop thread: SUB observations for WebRTC frames; PUSH to operator bind only when portal drives joints.
        teleop_hal_client_config = HalClientConfig(
            observation_endpoint=OBSERVATION_ENDPOINT,
            command_endpoint=COMMAND_ENDPOINT if teleop_send_commands else None,
        )

        if teleop_sensor_ids is not None:
            teleop_stop = threading.Event()
            teleop_thread = start_hal_teleop_signaling_thread(
                teleop_hal_client_config,
                transport_context,
                hal_server.get_sensor_interface(),
                stop_event=teleop_stop,
                bootstrap_sensor_catalog_ids=teleop_sensor_ids,
                teleop_edge_settings=_teleop_st,
                robot_definition=robot_definition,
                send_hal_commands=teleop_send_commands,
            )
            logger.info(
                "Teleop outbound signaling started: mode=%s url=%s reconnect_s=%.1f "
                "(bootstrap catalog ids=%s; viewer may override via signaling ``catalog_ids``); "
                "webrtc_hal_commands=%s",
                _teleop_st.mode,
                _teleop_st.server_signaling_ws_url,
                _teleop_st.server_reconnect_s,
                teleop_sensor_ids,
                teleop_send_commands,
            )

        if args.control_source == "inference":
            model_weights = ModelWeights(
                checkpoint_path=args.checkpoint,
                observation_dimensions=observation_dimensions,
                action_dim=model_definition.action_dim,
            )

            parkour_client = ParkourInferenceClient(
                hal_client_config=inference_hal_client_config,
                model_weights=model_weights,
                observation_dimensions=observation_dimensions,
                robot_definition=robot_definition,
                control_rate=CONTROL_RATE_HZ,
                device="cuda",
                transport_context=transport_context,
            )
            parkour_client.initialize()
            logger.info("Parkour inference client initialized")

            # Start inference client in separate thread
            parkour_client.start_thread(running_flag=lambda: running)
            if args.teleop:
                logger.info(
                    "Teleop video active; inference commands use source=inference; operator overrides when portal sends",
                )
        else:
            logger.info(
                f"Portal controller mode active: waiting for teleop control data-channel commands "
                f"on HAL command socket ({COMMAND_ENDPOINT})",
            )

        if args.data_collector_output_dir is not None or args.data_collector_config is not None:
            if args.data_collector_config is not None:
                dc_cfg = load_config(args.data_collector_config)
                # Entry-point transport wiring is authoritative for inproc deployment.
                dc_cfg.hal.observation_endpoint = OBSERVATION_ENDPOINT
                dc_cfg.hal.command_endpoint = COMMAND_ENDPOINT
                if args.data_collector_output_dir is not None:
                    dc_cfg.output_dir = Path(args.data_collector_output_dir).expanduser()
            else:
                dc_cfg = build_data_collector_config(
                    observation_endpoint=OBSERVATION_ENDPOINT,
                    command_endpoint=COMMAND_ENDPOINT,
                    output_dir=args.data_collector_output_dir,
                )
            collector_stop = threading.Event()
            _collector, collector_thread = start_collector_thread(
                dc_cfg,
                transport_context,
                collector_stop,
            )
            _collector.initialize()
            collector_thread.start()
            logger.info("HalDataCollector thread started (output_dir=%s)", dc_cfg.output_dir)

        logger.info(f"Starting production loop at {CONTROL_RATE_HZ} Hz")
        period_s = 1.0 / CONTROL_RATE_HZ
        lag_warning_count = 0

        # Main loop: HAL server operations
        try:
            while running:
                loop_start_ns = time.time_ns()

                # Publish observations from real sensors
                hal_server.set_observation()

                # Try to get joint command from the selected command source (non-blocking)
                # This command was generated from observations we published in a PREVIOUS iteration
                # We'll apply it in THIS iteration
                command = hal_server.get_joint_command(timeout_ms=1)  # 1ms timeout for non-blocking check
                if command is not None:
                    # Apply the command to actuators
                    hal_server.apply_command(command)
                # If no new command available, reuse the last command to maintain current pose
                # This allows the robot to continue while inference processes observations
                # The robot will continue moving based on the last applied command

                # Timing control
                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                    lag_warning_count = 0
                else:
                    if loop_duration_s > period_s * 1.1:
                        lag_warning_count += 1
                        if lag_warning_count == 1 or lag_warning_count % 100 == 0:
                            logger.warning(
                                "Loop unable to keep up! Frame time: %.2fms exceeds target: %.2fms (count=%d)",
                                loop_duration_s * 1000.0,
                                period_s * 1000.0,
                                lag_warning_count,
                            )
                    else:
                        lag_warning_count = 0

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Failed to run Jetson HAL server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        if teleop_stop is not None:
            teleop_stop.set()
        if teleop_thread is not None and teleop_thread.is_alive():
            teleop_thread.join(timeout=8.0)
            if teleop_thread.is_alive():
                logger.warning("Teleop HTTP thread did not exit within timeout")
        if collector_stop is not None:
            collector_stop.set()
        if collector_thread is not None and collector_thread.is_alive():
            collector_thread.join(timeout=8.0)
            if collector_thread.is_alive():
                logger.warning("HalDataCollector thread did not exit within timeout")
        # Clean up in reverse order of creation
        if parkour_client:
            parkour_client.close()
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

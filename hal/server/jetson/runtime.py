"""Pieces shared by the gamepad and model entry points of the Jetson HAL server.

Nothing here imports the policy (torch) or teleop stacks, so the gamepad entry
point can run on hosts that only have the HAL and firmware SDK installed.
"""

import argparse
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Optional

from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from hal.server.robot_definition import RobotDefinition
from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONTROL_RATE_HZ = 100.0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments accepted by every control source."""
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level for this process (default: INFO)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="hex",
        choices=["hex", "go2"],
        help="Robot definition to use (default: hex).",
    )
    parser.add_argument("--observation-bind", type=str, default=None, help="ZMQ observation bind endpoint.")
    parser.add_argument("--command-bind", type=str, default=None, help="ZMQ command bind endpoint.")
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
            "HAL endpoints from --observation-bind/--command-bind are enforced by this entrypoint."
        ),
    )


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name)
    logging.getLogger().setLevel(level)
    # aioice (aiortc ICE) logs every candidate-pair transition at INFO; keep operator logs readable.
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("aiortc").setLevel(logging.WARNING)


def select_robot(name: str) -> RobotDefinition:
    if name == "hex":
        logger.info("Using Krabby Hex robot definition (default)")
        return KRABBY_HEX_DEFINITION
    # Keep explicit option for Unitree-Go2 checkpoints.
    logger.info("Using Unitree Go2 robot definition")
    return UNITREE_GO2_DEFINITION


def create_hal_server(
    robot_definition: RobotDefinition, observation_bind: str, command_bind: str
) -> JetsonHalServer:
    """Build and initialize the HAL server with sensors and actuators (cameras are left to the caller)."""
    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    hal_server = JetsonHalServer(
        HalServerConfig(observation_bind=observation_bind, command_bind=command_bind),
        observation_dimensions=model_definition.get_observation_dimensions(robot_definition),
        action_dim=model_definition.action_dim,
        robot_definition=robot_definition,
    )
    hal_server.initialize()
    hal_server.initialize_sensors()
    hal_server.initialize_actuators()
    return hal_server


def warn_if_mcu_missing(hal_server: JetsonHalServer) -> None:
    # Missing MCU is non-fatal: the stack still starts so fleet telemetry can
    # report mcu_present=false / mcu_missing. Joint commands are no-ops until
    # the board is connected (apply_command logs when the SDK is absent).
    if hal_server._mcusdk is None or not hal_server._mcusdk.is_connected():
        logger.warning(
            "MCU not available — check firmware and wiring. "
            "Continuing without MCU (joint commands will not be sent)."
        )


def start_data_collector(
    args: argparse.Namespace,
    observation_bind: str,
    command_bind: str,
    transport_context,
) -> tuple[Optional[threading.Event], Optional[threading.Thread]]:
    """Start the optional rosbag collector; returns (None, None) when not requested."""
    if args.data_collector_output_dir is None and args.data_collector_config is None:
        return None, None

    # data_collection is not installed in the production locomotion image, so import lazily.
    from data_collection.collector import start_collector_thread
    from data_collection.collector_settings import build_data_collector_config
    from data_collection.config import load_config

    if args.data_collector_config is not None:
        dc_cfg = load_config(args.data_collector_config)
        # Entry-point transport wiring is authoritative.
        dc_cfg.hal.observation_endpoint = observation_bind
        dc_cfg.hal.command_endpoint = command_bind
        if args.data_collector_output_dir is not None:
            dc_cfg.output_dir = Path(args.data_collector_output_dir).expanduser()
    else:
        dc_cfg = build_data_collector_config(
            observation_endpoint=observation_bind,
            command_endpoint=command_bind,
            output_dir=args.data_collector_output_dir,
        )
    stop_event = threading.Event()
    collector, thread = start_collector_thread(dc_cfg, transport_context, stop_event)
    collector.initialize()
    thread.start()
    logger.info("HalDataCollector thread started (output_dir=%s)", dc_cfg.output_dir)
    return stop_event, thread


def run_control_loop(hal_server: JetsonHalServer, is_running: Callable[[], bool]) -> None:
    """Publish observations and apply the latest joint command at CONTROL_RATE_HZ until stopped."""
    logger.info(f"Starting production loop at {CONTROL_RATE_HZ} Hz")
    period_s = 1.0 / CONTROL_RATE_HZ
    lag_warning_count = 0

    try:
        while is_running():
            loop_start_ns = time.time_ns()

            hal_server.set_observation()

            # Non-blocking: the loop sleeps out the period below, and a timed poll rounds
            # up to the ~15 ms OS timer tick on Windows, overrunning the 10 ms period.
            command = hal_server.get_joint_command(timeout_ms=0)
            if command is not None:
                hal_server.apply_command(command)
            # If no new command available, the last command remains in effect.

            loop_duration_s = (time.time_ns() - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)

            if sleep_time > 0:
                time.sleep(sleep_time)
                lag_warning_count = 0
            elif loop_duration_s > period_s * 1.1:
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

        logger.info("Received interrupt signal, stopping...")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")


def stop_thread(stop_event: Optional[threading.Event], thread: Optional[threading.Thread], name: str) -> None:
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=8.0)
        if thread.is_alive():
            logger.warning("%s thread did not exit within timeout", name)

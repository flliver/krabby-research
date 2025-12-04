"""Production entry point for Jetson robot deployment.

This combines HAL server and parkour inference in the same process,
using inproc ZMQ for communication.

NOTE: This is the PRODUCTION entry point that runs on the robot.
"""

import argparse
import logging
import signal
import sys
import time

from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.jetson import JetsonHalServer
from compute.parkour.inference_client import ParkourInferenceClient
from compute.parkour.policy_interface import ModelWeights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Jetson production deployment."""
    parser = argparse.ArgumentParser(description="Jetson production deployment with HAL server and inference")

    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")

    # HAL endpoints (inproc for same-process communication)
    parser.add_argument(
        "--observation_bind",
        type=str,
        default="inproc://hal_observation",
        help="Observation endpoint (inproc for same-process)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default="inproc://hal_commands",
        help="Command endpoint (inproc for same-process)",
    )

    args = parser.parse_args()

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

    try:
        # Create HAL server config (inproc for production)
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        # Create and initialize HAL server
        hal_server = JetsonHalServer(hal_server_config)
        hal_server.initialize()

        # Initialize hardware (camera, sensors, actuators)
        hal_server.initialize_camera()
        hal_server.initialize_sensors()
        hal_server.initialize_actuators()

        logger.info("HAL server initialized")

        # Get transport context for inproc connections
        transport_context = hal_server.get_transport_context()

        # Create HAL client config
        hal_client_config = HalClientConfig(
            observation_endpoint=args.observation_bind,
            command_endpoint=args.command_bind,
        )

        # Create model weights configuration
        model_weights = ModelWeights(
            checkpoint_path=args.checkpoint,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
        )

        # Create parkour inference client
        parkour_client = ParkourInferenceClient(
            hal_client_config=hal_client_config,
            model_weights=model_weights,
            control_rate=args.control_rate,
            device=args.device,
            transport_context=transport_context,
        )
        parkour_client.initialize()
        logger.info("Parkour inference client initialized")

        # Start inference client in separate thread
        parkour_client.start_thread(running_flag=lambda: running)

        logger.info(f"Starting production loop at {args.control_rate} Hz")
        period_s = 1.0 / args.control_rate

        # Main loop: HAL server operations
        try:
            while running:
                loop_start_ns = time.time_ns()

                # Publish observations from real sensors
                hal_server.set_observation()

                # Apply joint commands from inference client
                hal_server.apply_command()

                # Timing control
                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_duration_s > period_s * 1.1:
                        logger.warning(
                            f"Loop unable to keep up! "
                            f"Frame time: {loop_duration_s*1000:.2f}ms "
                            f"exceeds target: {period_s*1000:.2f}ms"
                        )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Jetson deployment failed: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up in reverse order of creation
        if parkour_client:
            parkour_client.close()
        if hal_server:
            hal_server.close()


if __name__ == "__main__":
    main()

"""Production entry point for Jetson robot deployment.

NOTE: This is the PRODUCTION entry point that runs on the robot.
"""

import argparse
import logging
import signal
import sys

from hal.server import HalServerConfig
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel
from locomotion.jetson.inference_runner import InferenceRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for production deployment."""
    parser = argparse.ArgumentParser(description="Production inference runner for Jetson")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    parser.add_argument(
        "--observation_bind",
        type=str,
        default="inproc://hal_observation",
        help="Observation endpoint (use inproc for production)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default="inproc://hal_commands",
        help="Command endpoint (use inproc for production)",
    )

    args = parser.parse_args()

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        weights = ModelWeights(
            checkpoint_path=args.checkpoint,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
        )
        model = ParkourPolicyModel(weights, device=args.device)

        # Create HAL server config (inproc for production)
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        # Create inference runner
        runner = InferenceRunner(model, hal_server_config)
        runner.initialize()

        # Set up signal handlers
        def signal_handler(sig, frame):
            logger.info("Received interrupt signal, stopping inference runner...")
            runner.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run production loop
        try:
            runner.run(control_rate_hz=args.control_rate)
        finally:
            runner.close()

    except Exception as e:
        logger.error(f"Production inference runner failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""Entry point for inference test runner script.

This script runs the inference test runner which simulates the game loop
(inference logic) for testing purposes.

NOTE: This is for testing/development only. Production uses locomotion/jetson/main.py.
"""

import argparse
import logging
import sys

from compute.testing.inference_test_runner import run_inference_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inference test runner for policy inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--observation_endpoint", type=str, default="inproc://hal_observation", help="Observation endpoint")
    parser.add_argument("--command_endpoint", type=str, default="inproc://hal_commands", help="Command endpoint")
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")

    args = parser.parse_args()

    hal_endpoints = {
        "observation": args.observation_endpoint,
        "command": args.command_endpoint,
    }

    try:
        run_inference_test(
            checkpoint_path=args.checkpoint,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
            hal_endpoints=hal_endpoints,
            control_rate_hz=args.control_rate,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Inference test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


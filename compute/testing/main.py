"""Entry point for inference test runner script.

This script runs the inference test runner which simulates the game loop
(inference logic) for testing purposes.

It connects to a HAL server via the specified endpoints and runs inference
on observations received from the server.

It automatically starts a mock HAL server that publishes synthetic observations
(useful for testing inference without Isaac Sim or real hardware).

NOTE: This is for testing/development only. Production uses locomotion/jetson/main.py.
"""

import argparse
import logging
import signal
import sys
import threading
import time

from compute.testing.inference_test_runner import run_inference_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running
    logger.info("Received interrupt signal, stopping...")
    running = False


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
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds (None = run indefinitely)")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Always start mock HAL server
    logger.info("Starting mock HAL server with synthetic observations...")
    from compute.testing.mock_hal_server import MockHalServer
    from hal.server import HalServerConfig
    
    config = HalServerConfig(
        observation_bind=args.observation_endpoint,
        command_bind=args.command_endpoint,
    )
    mock_server = MockHalServer(config)
    mock_server.initialize()
    mock_server.start(rate_hz=args.control_rate)
    logger.info("Mock HAL server started")
    
    # Give server time to start
    time.sleep(0.5)

    # Set up timeout if specified
    timeout_triggered = threading.Event()
    
    def timeout_handler():
        """Handle timeout by setting the event."""
        if args.timeout is not None:
            time.sleep(args.timeout)
            if not timeout_triggered.is_set():
                logger.info(f"Timeout of {args.timeout} seconds reached, stopping...")
                timeout_triggered.set()
                # Send SIGTERM to ourselves to trigger graceful shutdown
                import os
                os.kill(os.getpid(), signal.SIGTERM)
    
    timeout_thread = None
    if args.timeout is not None:
        timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
        timeout_thread.start()
        logger.info(f"Timeout set to {args.timeout} seconds")

    try:
        hal_endpoints = {
            "observation": args.observation_endpoint,
            "command": args.command_endpoint,
        }

        # For inproc connections, pass the server's transport context to the client
        # This ensures they share the same ZMQ context
        transport_context = None
        if args.observation_endpoint.startswith("inproc://") or args.command_endpoint.startswith("inproc://"):
            transport_context = mock_server.get_transport_context()
            logger.info("Using shared ZMQ context for inproc connections")

        # Run inference test
        run_inference_test(
            checkpoint_path=args.checkpoint,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
            hal_endpoints=hal_endpoints,
            control_rate_hz=args.control_rate,
            device=args.device,
            transport_context=transport_context,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Inference test failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up
        global running
        running = False
        mock_server.stop()
        mock_server.close()
        logger.info("Mock HAL server stopped")


if __name__ == "__main__":
    main()

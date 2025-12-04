"""Entry point for IsaacSim HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together.

For standalone server mode (client runs separately), use TCP endpoints instead.
"""

import argparse
import logging
import signal
import sys
import time

from isaaclab.app import AppLauncher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for IsaacSim HAL server with integrated inference."""
    parser = argparse.ArgumentParser(
        description="IsaacSim HAL server with integrated inference client"
    )

    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        required=True,
        help="Action dimension",
    )
    parser.add_argument(
        "--obs_dim",
        type=int,
        required=True,
        help="Observation dimension",
    )
    parser.add_argument(
        "--control_rate",
        type=float,
        default=100.0,
        help="Control loop rate in Hz",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )

    # IsaacSim arguments
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name (e.g., Isaac-Anymal-D-v0)",
    )

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

    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import after AppLauncher to avoid conflicts
    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

    from hal.client.config import HalClientConfig
    from hal.server import HalServerConfig
    from hal.server.isaac import IsaacSimHalServer
    from compute.parkour.inference_client import ParkourInferenceClient
    from compute.parkour.policy_interface import ModelWeights

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
    env = None

    try:
        # Parse environment configuration
        # Note: parse_env_cfg() will import parkour_tasks internally, which triggers
        # gym registration, but we bypass gym.make() and use direct instantiation instead
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=1,
            use_fabric=True,
        )

        # Create environment using IsaacLab native API (bypassing deprecated gym.make())
        # The environment class still implements gym.Env interface for compatibility
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        logger.info(f"Created IsaacSim environment: {args.task} (using direct instantiation)")

        # Create HAL server config
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        # Create and initialize HAL server
        hal_server = IsaacSimHalServer(hal_server_config, env=env)
        hal_server.initialize()
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

        logger.info(f"Starting integrated loop at {args.control_rate} Hz")
        period_s = 1.0 / args.control_rate

        # Main loop: step simulation and publish observations
        try:
            while running and simulation_app.is_running():
                loop_start_ns = time.time_ns()

                # Step IsaacSim environment
                env.step(None)

                # Publish telemetry from simulation
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
                            f"Simulation unable to keep up! "
                            f"Frame time: {loop_duration_s*1000:.2f}ms "
                            f"exceeds target: {period_s*1000:.2f}ms"
                        )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Failed to run IsaacSim HAL server: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up in reverse order of creation
        if parkour_client:
            parkour_client.close()
        if hal_server:
            hal_server.close()
        if env:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()


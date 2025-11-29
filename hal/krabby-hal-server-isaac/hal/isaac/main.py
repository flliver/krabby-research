"""Entry point for IsaacSim HAL server."""

import argparse
import logging

from isaaclab.app import AppLauncher

from hal.server.config import HalServerConfig
from hal.isaac.hal_server import IsaacSimHalServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for IsaacSim HAL server."""
    parser = argparse.ArgumentParser(description="IsaacSim HAL server")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--base_port", type=int, default=6000, help="Base port for HAL endpoints")
    parser.add_argument("--observation_bind", type=str, default=None, help="Observation endpoint (overrides base_port)")
    parser.add_argument("--command_bind", type=str, default=None, help="Command endpoint (overrides base_port)")

    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import after app launch
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    try:
        # Parse environment configuration
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)

        # Create environment
        env = gym.make(args.task, cfg=env_cfg)

        # Create HAL server config
        if args.observation_bind and args.command_bind:
            config = HalServerConfig.from_endpoints(
                observation_bind=args.observation_bind,
                command_bind=args.command_bind,
            )
        else:
            config = HalServerConfig(base_port=args.base_port)

        # Create and initialize HAL server
        hal_server = IsaacSimHalServer(config, env=env)
        hal_server.initialize()

        logger.info("IsaacSim HAL server started")

        # Main loop
        try:
            while simulation_app.is_running():
                # Step environment
                env.step(None)  # Action will come from hal

                # Set observation
                hal_server.set_observation()

                # Move robot (get command from HAL and apply)
                hal_server.move()

        except KeyboardInterrupt:
            logger.info("HAL server interrupted by user")
        finally:
            hal_server.close()
            env.close()

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()


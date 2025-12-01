"""Entry point for Isaac Sim container that combines inference and HAL server.

This combines policy inference and Isaac Sim HAL server in the same process,
using inproc ZMQ for communication.
"""

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from isaaclab.app import AppLauncher
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg

from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server import HalServerConfig
from hal.server.isaac import IsaacSimHalServer
from compute.parkour.policy_interface import ModelWeights, ParkourPolicyModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for Isaac Sim container."""
    parser = argparse.ArgumentParser(description="Isaac Sim container with inference and HAL server")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--action_dim", type=int, required=True, help="Action dimension")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension")
    parser.add_argument("--control_rate", type=float, default=100.0, help="Control loop rate in Hz")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    
    # Isaac Sim arguments
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., Isaac-Anymal-D-v0)")
    
    # HAL server arguments
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
    
    # Add AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

        hal_server: Optional[IsaacSimHalServer] = None
        hal_client: Optional[HalClient] = None
        env = None
        model: Optional[ParkourPolicyModel] = None
        nav_cmd: Optional[NavigationCommand] = None
        running = True

    def signal_handler(sig, frame):
        """Handle interrupt signals."""
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        weights = ModelWeights(
            checkpoint_path=args.checkpoint,
            action_dim=args.action_dim,
            obs_dim=args.obs_dim,
        )
        model = ParkourPolicyModel(weights, device=args.device)

        # Parse environment configuration
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1, use_fabric=True)

        # Create environment
        env = gym.make(args.task, cfg=env_cfg)

        # Create HAL server config (inproc for production)
        hal_server_config = HalServerConfig(
            observation_bind=args.observation_bind,
            command_bind=args.command_bind,
        )

        # Create and initialize HAL server
        hal_server = IsaacSimHalServer(hal_server_config, env=env)
        hal_server.initialize()

        # Create HAL client config with inproc endpoints
        hal_client_config = HalClientConfig(
            observation_endpoint=args.observation_bind,
            command_endpoint=args.command_bind,
        )
        # Get transport context from server for inproc connections
        transport_context = hal_server.get_transport_context() if "inproc://" in args.observation_bind else None
        hal_client = HalClient(hal_client_config, transport_context=transport_context)
        hal_client.initialize()

        logger.info("Isaac Sim container initialized")
        logger.info(f"Starting combined loop at {args.control_rate} Hz")

        period_s = 1.0 / args.control_rate

        # Main loop: combines Isaac Sim stepping, HAL server, and inference
        try:
            while running and simulation_app.is_running():
                loop_start_ns = time.time_ns()

                # Step Isaac Sim environment
                env.step(None)  # Action will come from hal/inference

                # Publish telemetry from simulation
                hal_server.set_observation()

                # Poll HAL for latest hardware observation
                hw_obs = hal_client.poll(timeout_ms=1)
                
                if hw_obs is None:
                    # No new observation available, skip inference this frame
                    logger.debug("No new observation available")
                    hal_server.move()  # Still apply any pending commands
                    continue

                # Map hardware observation to model observation format using mapper
                from compute.parkour.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
                from compute.parkour.types import ParkourModelIO
                from hal.client.observation.types import NavigationCommand
                
                mapper = KrabbyHWObservationsToParkourMapper()
                model_obs = mapper.map(hw_obs)
                
                # Check if navigation command is set (initialize with default if not)
                if nav_cmd is None:
                    nav_cmd = NavigationCommand.create_now(vx=0.0, vy=0.0, yaw_rate=0.0)
                
                # Build model IO (preserve timestamp from observation)
                model_io = ParkourModelIO(
                    timestamp_ns=model_obs.timestamp_ns,
                    schema_version=model_obs.schema_version,
                    nav_cmd=nav_cmd,
                    observation=model_obs,
                )

                # Run inference
                try:
                    inference_result = model.inference(model_io)

                    if not inference_result.success:
                        raise RuntimeError(f"Inference failed: {inference_result.error_message}")

                    # Map inference response to hardware joint positions
                    from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper
                    mapper = ParkourLocomotionToKrabbyHWMapper(model_action_dim=model.action_dim)
                    joint_positions = mapper.map(inference_result)

                    # Send command back to HAL server
                    if not hal_client.put_joint_command(joint_positions):
                        raise RuntimeError("Failed to put joint command to HAL")
                except Exception as e:
                    logger.error(f"Critical inference failure: {e}", exc_info=True)
                    # Fail fast - re-raise to stop the loop
                    raise

                # Apply joint command to simulation
                hal_server.move()

                # Timing control
                loop_end_ns = time.time_ns()
                loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                sleep_time = max(0.0, period_s - loop_duration_s)

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    if loop_duration_s > period_s * 1.1:
                        logger.warning(
                            f"Simulation is unable to keep up! "
                            f"Frame time: {loop_duration_s*1000:.2f}ms exceeds target: {period_s*1000:.2f}ms. "
                            f"This may cause control instability."
                        )

        except KeyboardInterrupt:
            logger.info("Container interrupted by user")
        finally:
            if hal_client:
                hal_client.close()
            if hal_server:
                hal_server.close()
            if env:
                env.close()

    except Exception as e:
        logger.error(f"Isaac Sim container failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()


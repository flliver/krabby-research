"""Entry point for IsaacSim HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together.

For standalone server mode (client runs separately), use TCP endpoints instead.

Simulates a robot environment (default: single robot), gathers observations,
runs inference, and applies commands to control the robot. Supports visual display
and video recording like play.py.
"""

import argparse
import logging
import os
import signal
import sys
import time

import torch
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
        "--inference_device",
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
        help="Task name (e.g., Isaac-Anymal-D-v0 or Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0)",
    )

    # Environment arguments
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel environments to simulate (default: 16, matching play.py)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record videos during execution",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=500,
        help="Length of the recorded video in steps",
    )
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible",
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
    
    # Enable cameras for visual display (needed for rendering)
    # This ensures the window and rendering pipeline are set up correctly
    if not hasattr(args, 'enable_cameras') or args.enable_cameras is None:
        args.enable_cameras = True

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Wait for app window to be created (needed for camera controller)
    # The window may not exist immediately after AppLauncher starts
    import omni.appwindow
    import time
    logger.info("Waiting for app window to be created...")
    max_wait_time = 5.0
    wait_interval = 0.1
    elapsed = 0.0
    while elapsed < max_wait_time:
        app_window = omni.appwindow.get_default_app_window()
        if app_window is not None:
            logger.info("App window created successfully")
            break
        time.sleep(wait_interval)
        elapsed += wait_interval
    
    if omni.appwindow.get_default_app_window() is None:
        logger.warning("App window not available after waiting. Camera controller may fail.")

    # Import after AppLauncher to avoid conflicts
    import sys
    from isaaclab_tasks.utils import parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
    
    # Import parkour_tasks to register gym environments
    # Add parkour_tasks to sys.path to ensure it's found
    parkour_tasks_path = "/workspace/parkour/parkour_tasks"
    if parkour_tasks_path not in sys.path:
        sys.path.insert(0, parkour_tasks_path)
    
    # Import packages to register gym environments
    # This must happen before parse_env_cfg is called
    logger.info("Importing packages to register gym environments...")
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401
    logger.info("Packages imported successfully")

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
            device=args.device,  # Use AppLauncher's device for environment
            num_envs=args.num_envs,
            use_fabric=not args.disable_fabric,
        )

        # Determine render mode based on video flag
        # For visual display, use None (default window rendering)
        # For video recording, use "rgb_array" 
        render_mode = "rgb_array" if args.video else None
        
        # Create environment using IsaacLab native API (bypassing deprecated gym.make())
        # The environment class still implements gym.Env interface for compatibility
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=render_mode)
        
        logger.info(f"Created IsaacSim environment: {args.task} with {env.num_envs} parallel environments (render_mode={render_mode})")

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
            device=args.inference_device,
            transport_context=transport_context,
        )
        parkour_client.initialize()
        logger.info("Parkour inference client initialized")

        # Start inference client in separate thread
        parkour_client.start_thread(running_flag=lambda: running)

        logger.info(f"Starting integrated loop at {args.control_rate} Hz")
        period_s = 1.0 / args.control_rate
        
        # Get environment step dt for real-time mode
        dt = env.unwrapped.step_dt
        timestep = 0

        # Main loop: step simulation and publish observations
        try:
            while running and simulation_app.is_running():
                loop_start_ns = time.time_ns()

                # Get joint command from inference client and convert to action tensor
                # This must happen before env.step() which requires an action
                command = hal_server.get_joint_command(timeout_ms=10)
                if command is None:
                    logger.warning("No command received from inference client, using zero actions")
                    # Create zero action tensor as fallback
                    action = torch.zeros((args.num_envs, args.action_dim), device=env.device, dtype=torch.float32)
                else:
                    # Convert command to action tensor
                    # JointCommand has 18 joints, but environment expects action_dim (12) actions
                    # The mapper puts the first action_dim actions into the first action_dim joint positions
                    command_array = command.joint_positions[:args.action_dim]  # Take first action_dim joints
                    action = torch.from_numpy(command_array).to(device=env.device, dtype=torch.float32)
                    # Add batch dimension if needed (env.step expects (num_envs, action_dim))
                    if action.ndim == 1:
                        action = action.unsqueeze(0)  # Shape: (1, ACTION_DIM)
                    # Expand to match num_envs if needed
                    if action.shape[0] == 1 and args.num_envs > 1:
                        action = action.expand(args.num_envs, -1)

                # Step IsaacSim environment with action
                env.step(action)

                # Publish telemetry from simulation
                hal_server.set_observation()

                # Handle video recording limit
                if args.video:
                    timestep += 1
                    if timestep >= args.video_length:
                        logger.info(f"Reached video length limit ({args.video_length} steps), stopping...")
                        break

                # Timing control
                if args.real_time:
                    # Real-time mode: sleep based on environment step dt
                    loop_end_ns = time.time_ns()
                    loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
                    sleep_time = dt - loop_duration_s
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                else:
                    # Fixed rate mode: sleep based on control rate
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


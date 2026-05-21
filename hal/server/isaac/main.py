"""Entry point for IsaacSim HAL server with integrated inference client.

This entry point runs both the HAL server and inference client in the same process
using inproc ZMQ for zero-copy communication. This is the recommended deployment
for production use where server and client run together.

For standalone server mode (client runs separately), use TCP endpoints instead.

Simulates a robot environment (default: single robot), gathers observations,
runs inference, and applies commands to control the robot. Supports visual display
and video recording. 
"""

import argparse
import logging
import os
import signal
from dataclasses import replace
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

from data_collection.collector import start_collector_thread
from data_collection.collector_settings import build_data_collector_config
from data_collection.config import load_config

logger = logging.getLogger(__name__)
CONTROL_RATE_HZ = 100.0
# Sensor catalog ids passed to teleop signaling when HAL introspects RGB-D streams (--teleop).
TELEOP_BOOTSTRAP_SENSOR_CATALOG_IDS: tuple[str, ...] = ("front_rgbd", "side_rgbd")

# Outbound WebSocket to krabby-teleop-portal ``/ws/robot``. Default for Isaac sim when the portal
# runs alongside on the same machine (see Docker note in docs / reply when using bridge networks).
ISAAC_TELEOP_SIGNALING_WS_URL: str = "ws://127.0.0.1:9000/ws/robot"


def _sim_rgbd_mount_link(robot: str) -> str | None:
    """USD link under ``Robot/<link>/`` for HAL sim RGB-D prims; ``None`` if ``robot`` is ``auto``."""

    if robot == "hex":
        return "crab_hex"
    if robot in ("go2", "quad"):
        return "base"
    return None


def _zmq_client_connect_url(bind: str) -> str:
    """HAL server ``bind`` address → ZMQ ``connect`` URL for in-process or localhost clients."""
    if bind.startswith("tcp://*:"):
        return "tcp://127.0.0.1:" + bind[len("tcp://*:") :]
    if bind.startswith("tcp://0.0.0.0:"):
        return "tcp://127.0.0.1:" + bind[len("tcp://0.0.0.0:") :]
    return bind


def _ensure_server_log_visible(debug: bool = False) -> None:
    """Attach stderr handler to HAL server loggers so messages appear when Isaac Sim overrides root logging."""
    level = logging.DEBUG if debug else logging.INFO
    for name in ("hal.server.isaac.main", "hal.server.isaac.hal_server", "hal.server.server"):
        log = logging.getLogger(name)
        has_stderr = any(getattr(h, "stream", None) is sys.stderr for h in log.handlers)
        if not has_stderr:
            h = logging.StreamHandler(sys.stderr)
            h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
            log.addHandler(h)
        log.setLevel(level)


def main():
    """Main entry point for IsaacSim HAL server with integrated inference."""
    parser = argparse.ArgumentParser(
        description="IsaacSim HAL server with integrated inference client"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging (HAL and controller)",
    )
    parser.add_argument(
        "--joystick",
        action="store_true",
        help="Joystick: no checkpoint, TCP bind; use krabby-uno-sim as client",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (required unless --joystick)",
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
        default=None,
        help="Task name (e.g., Isaac-Extreme-Parkour-Teacher-Unitree-Go2-Play-v0). Omit when using --usd.",
    )
    parser.add_argument(
        "--usd",
        type=str,
        default=None,
        help="Path to robot/scene USD (e.g. assets/crab_hex_ref.usd). Uses Isaac-CrabHex-Joystick-v0 and --robot hex.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="auto",
        choices=["auto", "quad", "go2", "hex"],
        help="Robot definition: auto (from task name), quad, go2, or hex (18 joints)",
    )

    # Environment arguments
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
        "--seed",
        type=int,
        default=42,
        help="Seed for environment randomization (default: 42)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible",
    )

    parser.add_argument(
        "--observation_bind",
        type=str,
        default=None,
        help="Observation endpoint (default: inproc or tcp://*:5555 if --joystick)",
    )
    parser.add_argument(
        "--command_bind",
        type=str,
        default=None,
        help="Command endpoint (default: inproc or tcp://*:5556 if --joystick)",
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
            "HAL endpoints are always enforced from --observation_bind/--command_bind."
        ),
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help=(
            "Outbound WebRTC teleop to the portal: HAL observation poll + signaling. When the viewer enables "
            "operator_override and sends controller state, the teleop HalClient PUSHes OPERATOR JointCommands "
            "to command_bind (same socket as inference; operator wins precedence). Signaling ws://127.0.0.1:9000/ws/robot. "
            "ICE/STUN from teleop/edge/robot_settings.py. Combine with --joystick for krabby-uno-sim on TCP."
        ),
    )

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if not args.joystick and not args.checkpoint:
        parser.error("--checkpoint required unless --joystick")
    if args.usd:
        if args.task is not None:
            parser.error("Do not pass --task when using --usd")
        os.environ["KRABBY_HEX_USD_PATH"] = args.usd
        args.task = "Isaac-CrabHex-Joystick-v0"
        args.robot = "hex"
        logger.info("Using --usd: task=%s, robot=hex, KRABBY_HEX_USD_PATH=%s", args.task, args.usd)
    elif args.task is None:
        parser.error("--task required unless --usd is given")
    if args.teleop and args.robot == "auto":
        parser.error("--teleop requires --robot go2, quad, or hex (not auto)")
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Ensure our server logs appear even when Isaac Sim overrides root logging (e.g. in Docker)
    _ensure_server_log_visible(debug=args.debug)
    if args.joystick and args.checkpoint:
        args.checkpoint = None
    if args.observation_bind is None:
        args.observation_bind = "tcp://*:5555" if args.joystick else "inproc://hal_observation"
    if args.command_bind is None:
        args.command_bind = "tcp://*:5556" if args.joystick else "inproc://hal_commands"

    args.enable_cameras = bool(args.teleop)
    logger.info("enable_cameras=%s (sim RGB-D sensors only with --teleop)", args.enable_cameras)

    # Launch IsaacLab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Wait for app window when not headless (needed for camera controller).
    # In joystick mode use a short wait so startup is not blocked.
    if not getattr(args, "headless", False):
        import omni.appwindow
        # The window may not exist immediately after AppLauncher starts.
        max_wait_time = 0.5 if args.joystick else 5.0
        wait_interval = 0.1
        elapsed = 0.0
        while elapsed < max_wait_time:
            app_window = omni.appwindow.get_default_app_window()
            if app_window is not None:
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
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401

    from hal.client.config import HalClientConfig
    from hal.server import HalServerConfig
    from hal.server.isaac import IsaacSimHalServer
    from hal.server.isaac.sim_rgbd_camera_cfgs import attach_sim_rgbd_sensors_to_scene_cfg
    from hal.server.robot_definition_krabby_hex import KRABBY_HEX_DEFINITION
    from hal.server.robot_definition_krabby_quad import KRABBY_QUAD_DEFINITION
    from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
    from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
    from compute.parkour.inference_client import ParkourInferenceClient
    from compute.parkour.policy_interface import ModelWeights
    from hal.server.teleop_portal_signaling import start_hal_teleop_signaling_thread
    from teleop.edge.robot_settings import build_teleop_edge_settings

    # Running flag for graceful shutdown
    running = True
    teleop_stop = threading.Event()

    def signal_handler(sig, frame):
        """Handle interrupt signals."""
        nonlocal running
        logger.info("Received interrupt signal, stopping...")
        teleop_stop.set()
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hal_server = None
    parkour_client = None
    env = None
    collector_stop: threading.Event | None = None
    collector_thread: threading.Thread | None = None
    teleop_thread: threading.Thread | None = None

    # Parse environment configuration
    # Note: parse_env_cfg() will import parkour_tasks internally, which triggers
    # gym registration, but we bypass gym.make() and use direct instantiation instead
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,  # Use AppLauncher's device for environment
        use_fabric=not args.disable_fabric,
    )
    
    # Set seed for environment randomization
    env_cfg.seed = args.seed
    if args.joystick:
        # One env is enough for joystick control and significantly speeds up startup and step time.
        env_cfg.scene.num_envs = 1

    if args.teleop:
        sim_rgbd_link = _sim_rgbd_mount_link(args.robot)
        if sim_rgbd_link is None:
            raise ValueError(
                f"Cannot resolve sim RGB-D mount link for --robot {args.robot!r} with --teleop"
            )
        attach_sim_rgbd_sensors_to_scene_cfg(env_cfg.scene, sim_rgbd_link)
        _sensor_dt = env_cfg.sim.dt * env_cfg.decimation
        env_cfg.scene.front_camera.update_period = _sensor_dt
        env_cfg.scene.front_rgb.update_period = _sensor_dt
        env_cfg.scene.side_camera.update_period = _sensor_dt
        env_cfg.scene.side_rgb.update_period = _sensor_dt
        logger.info(
            "HAL sim RGB-D sensors on Robot/%s (front_rgb, front_camera, side_rgb, side_camera)",
            sim_rgbd_link,
        )

    # Determine render mode based on video flag
    # For visual display, use None (default window rendering)
    # For video recording, use "rgb_array" 
    render_mode = "rgb_array" if args.video else None
    
    # Create environment using gym.make() to ensure proper configuration
    # This ensures all gym environment registration and configuration is properly applied
    import gymnasium as gym
    env = gym.make(args.task, cfg=env_cfg, render_mode=render_mode)
    
    logger.info(f"Created IsaacSim environment: {args.task} with {env.unwrapped.num_envs} parallel environments (render_mode={render_mode})")

    # Reset environment to initialize state
    # env.reset() calls observation_manager.compute() internally (line 178 in parkour_manager_based_env.py)
    # This is the first compute() call. We need a second compute() call after reset to match
    # the environment's initialization pattern, which we do by calling set_observation() after reset.
    reset_obs_dict, reset_extras = env.reset()
    logger.info("Environment reset complete")

    # Create HAL server config
    hal_server_config = HalServerConfig(
        observation_bind=args.observation_bind,
        command_bind=args.command_bind,
    )

    if args.robot == "hex":
        robot_definition = KRABBY_HEX_DEFINITION
        logger.info("Using Krabby Hex robot definition (18 joints)")
    elif args.robot == "go2":
        robot_definition = UNITREE_GO2_DEFINITION
        logger.info(f"Using Unitree Go2 robot definition for task: {args.task}")
    elif args.robot == "quad":
        robot_definition = KRABBY_QUAD_DEFINITION
        logger.info(f"Using Krabby Quad robot definition for task: {args.task}")
    else:
        if "Unitree-Go2" in args.task or "unitree_go2" in args.task.lower():
            robot_definition = UNITREE_GO2_DEFINITION
            logger.info(f"Using Unitree Go2 robot definition (num_prop=53) for task: {args.task}")
        else:
            robot_definition = KRABBY_QUAD_DEFINITION
            logger.info(f"Using Krabby Quad robot definition for task: {args.task}")
    env_action_dim = env.unwrapped.action_manager.total_action_dim
    robot_joint_count = robot_definition.get_total_joint_count()
    if robot_joint_count != env_action_dim:
        raise ValueError(
            f"Robot joint count ({robot_joint_count} for --robot {args.robot}) does not match "
            f"the task's action dimension ({env_action_dim}). "
            f"The task '{args.task}' is configured for a different robot. "
            "Use --robot quad or --robot go2 for Go2/quad tasks; hexapod (18-joint) requires a hexapod task (not yet available)."
        )

    model_definition = PARKOUR_MODEL_OBSERVATION_DEFINITION
    observation_dimensions = model_definition.get_observation_dimensions(robot_definition)
    hal_server = IsaacSimHalServer(
        hal_server_config,
        robot_definition,
        env=env,
        observation_dimensions=observation_dimensions,
        command_timeout_s=None if args.joystick else 1.0,
    )
    hal_server.initialize()
    if args.joystick and getattr(args, "debug", False):
        hal_server.set_debug(True)
        logger.debug("HAL server ZMQ debug enabled (will log received commands)")
    logger.info("HAL server initialized")

    # Get transport context for inproc connections
    transport_context = hal_server.get_transport_context()
    # Same ZMQ connect URL for every HalClient that SUBscribes to HAL observations (inproc unchanged).
    observation_client_endpoint = _zmq_client_connect_url(args.observation_bind)
    command_client_endpoint = _zmq_client_connect_url(args.command_bind)

    if args.teleop:
        base_teleop = build_teleop_edge_settings()
        teleop_settings = replace(
            base_teleop,
            mode="agent",
            server_signaling_ws_url=ISAAC_TELEOP_SIGNALING_WS_URL,
        )
        bootstrap_ids = list(TELEOP_BOOTSTRAP_SENSOR_CATALOG_IDS)
        teleop_hal_cfg = HalClientConfig(
            observation_endpoint=observation_client_endpoint,
            command_endpoint=command_client_endpoint,
        )
        teleop_thread = start_hal_teleop_signaling_thread(
            teleop_hal_cfg,
            transport_context,
            hal_server.get_sensor_interface(),
            stop_event=teleop_stop,
            bootstrap_sensor_catalog_ids=bootstrap_ids,
            teleop_edge_settings=teleop_settings,
            robot_definition=robot_definition,
            send_hal_commands=True,
        )
        logger.info(
            "Teleop signaling thread started (url=%s, observation=%s, command_connect=%s, "
            "operator_override→OPERATOR PUSH, bootstrap=%s)",
            teleop_settings.server_signaling_ws_url,
            observation_client_endpoint,
            command_client_endpoint,
            bootstrap_ids,
        )

    if not args.joystick:
        # Create HAL client config
        hal_client_config = HalClientConfig(
            observation_endpoint=observation_client_endpoint,
            command_endpoint=args.command_bind,
        )
        model_weights = ModelWeights(
            checkpoint_path=args.checkpoint,
            observation_dimensions=observation_dimensions,
            action_dim=model_definition.action_dim,
        )
        parkour_client = ParkourInferenceClient(
            hal_client_config=hal_client_config,
            model_weights=model_weights,
            observation_dimensions=observation_dimensions,
            robot_definition=robot_definition,
            control_rate=CONTROL_RATE_HZ,
            device=args.inference_device,
            transport_context=transport_context,
        )
        # Initialize inference client first (creates model)
        parkour_client.initialize()
        logger.info("Parkour inference client initialized")
        # Start inference client in separate thread
        parkour_client.start_thread(running_flag=lambda: running)
    else:
        parkour_client = None
        logger.info("Joystick mode: waiting for krabby-uno-sim on %s / %s", args.observation_bind, args.command_bind)

    if args.data_collector_output_dir is not None or args.data_collector_config is not None:
        if args.data_collector_config is not None:
            dc_cfg = load_config(args.data_collector_config)
            # Entry-point transport wiring is authoritative for this process.
            dc_cfg.hal.observation_endpoint = observation_client_endpoint
            dc_cfg.hal.command_endpoint = args.command_bind
            if args.data_collector_output_dir is not None:
                dc_cfg.output_dir = Path(args.data_collector_output_dir).expanduser()
        else:
            dc_cfg = build_data_collector_config(
                observation_endpoint=observation_client_endpoint,
                command_endpoint=args.command_bind,
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

    logger.info(f"Starting loop at {CONTROL_RATE_HZ} Hz")
    period_s = 1.0 / CONTROL_RATE_HZ
    
    # Get environment step dt for real-time mode
    dt = env.unwrapped.step_dt
    timestep = 0

    # Publish initial observation through HAL
    # This is the second compute() call after reset. We need this because gym.make() + reset()
    # requires a second observation computation to match the environment's initialization pattern.
    # set_observation() will compute the observation (second call) and cache it
    hal_server.set_observation()

    if args.joystick:
        logger.info(
            "Joystick: main loop ready, waiting for first command on %s (start krabby-uno-sim on host if not already)",
            args.command_bind,
        )
        if args.teleop:
            logger.info(
                "Joystick+teleop: enable operator_override in the portal to drive from the browser controller; "
                "otherwise krabby-uno-sim (TCP) pushes commands.",
            )

    timestep = 0
    loop_start_t = time.time()
    telemetry_last_t = loop_start_t
    # Main loop: step simulation and publish observations
    # Pattern: wait for action (for observation we published) -> step -> publish new observation
    # ALL communication must go through HAL - no direct inference calls
    # Note: Initial observation was already published before the loop
    while running and simulation_app.is_running():
        loop_start_ns = time.time_ns()

        # Wait for action corresponding to the observation we published (synchronous matching)
        # For timestep 0, this is the initial observation published before the loop
        # For subsequent timesteps, this is the observation published at the end of the previous iteration
        # apply_command() will loop internally until command received or timeout (throws if timeout)
        # apply_command() automatically expands single action to all environments if num_envs > 1
        action = hal_server.apply_command()

        # Ensure action has correct shape [num_envs, action_dim]
        # apply_command() should already expand to [num_envs, action_dim] if needed
        num_envs = env.unwrapped.num_envs
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] != num_envs:
            # If we got a single action, expand it to all environments
            if action.shape[0] == 1:
                action = action.expand(num_envs, -1)
            else:
                raise ValueError(f"Action tensor shape {action.shape} != [{num_envs}, action_dim]")
        action = action.to(device=env.unwrapped.device)

        # Track last applied action BEFORE stepping, so set_observation() can use it for previous_action
        # Extract first 12 joints (action_dim) from action tensor for first environment
        action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
        n = len(hal_server._last_applied_action)
        prev = hal_server._last_applied_action.copy()
        joint_delta_max = float(np.max(np.abs(action_np[:n] - prev))) if n else 0.0
        hal_server._last_applied_action[:] = action_np[:n].astype(np.float32)

        # Step environment with action for all environments
        # Note: env.step() increments episode_length_buf BEFORE computing observations
        obs_dict, _, _, _, extras = env.step(action)
        
        # Cache the observation from env.step() for set_observation() to use
        # We use the observation directly from env.step() to avoid recomputing it, which ensures
        # we use the exact observation that the environment computed with the correct previous_action
        # from action_history_buf (updated during env.step() via action_manager.process_action()).
        hal_server._latest_obs_dict = obs_dict
        hal_server._latest_obs_tensor = obs_dict["policy"]
        
        # Publish new observation through HAL for next iteration
        # set_observation() will use the cached observation from env.step()
        # The inference client running in background thread will process it and send back action
        hal_server.set_observation()

        timestep += 1
        if args.joystick:
            t = time.time()
            if timestep == 1:
                logger.info("Joystick: first command received, sim stepping. Telemetry every 5s.")
            # Periodic telemetry every 5s so Pro Controller activity is visible in logs.
            if t - telemetry_last_t >= 5.0:
                elapsed = t - loop_start_t
                rate = timestep / elapsed if elapsed > 0 else 0
                logger.info("Joystick: %d steps in %.1fs (~%.0f Hz)", timestep, elapsed, rate)
                telemetry_last_t = t

        # Handle video recording limit
        if args.video:
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

    # Clean up in reverse order of creation
    teleop_stop.set()
    if teleop_thread is not None and teleop_thread.is_alive():
        teleop_thread.join(timeout=12.0)
        if teleop_thread.is_alive():
            logger.warning("Teleop signaling thread did not exit within timeout")
    if collector_stop is not None:
        collector_stop.set()
    if collector_thread is not None and collector_thread.is_alive():
        collector_thread.join(timeout=8.0)
        if collector_thread.is_alive():
            logger.warning("HalDataCollector thread did not exit within timeout")
    if parkour_client:
        parkour_client.close()
    if hal_server:
        hal_server.close()
    if env:
        env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


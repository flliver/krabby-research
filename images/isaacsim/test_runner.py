#!/usr/bin/env python3
"""Standalone test runner that initializes AppLauncher and runs test functions directly.

This script initializes Isaac Sim via AppLauncher, then runs test functions directly
without pytest. This allows tests to create real Isaac Lab environments since AppLauncher
is properly initialized before any omni modules are imported.

Usage:
    /workspace/testenv/bin/python /workspace/test_runner.py [test_name]

Running Tests:
    PYTHONUNBUFFERED=1 timeout 300 docker run --rm --gpus all \
        --entrypoint /workspace/run_test_runner.sh \
        krabby-isaacsim:latest <test_name>
    
    Omit <test_name> to run all tests. Exit code 0 = PASS, 1 = FAIL.

Adding New Tests:
    1. Add function run_test_your_test_name() to this file
    2. Register in tests dictionary in main()
    3. Rebuild: make build-isaacsim-image

Troubleshooting Stalls:
    Run test_isaacsim_create_environment_only to isolate:
    - If it stalls: Issue in environment creation (check /isaac-sim/kit/logs/)
    - If it passes but test_isaacsim_hal_server_with_real_isaaclab stalls: Issue in HAL server initialization
    
    See images/isaacsim/README.md for details.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Import Isaac Sim modules (must be before AppLauncher initialization)
from isaacsim.simulation_app import SimulationApp
from isaaclab.app import AppLauncher

# Set up file logging
log_dir = Path("/workspace/test_logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to both file and console
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Test runner logging to: {log_file}")

# Ensure logs are flushed
def flush_logs():
    """Flush all log handlers to ensure logs are written."""
    for handler in logger.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()

# Set up signal handler to print current step on timeout/interrupt
current_step = "[INIT]"

def signal_handler(sig, frame):
    logger.warning(f"\n[INTERRUPT] Test interrupted at: {current_step}")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set up PYTHONPATH to include Isaac Lab source
isaaclab_source = "/workspace/isaaclab/source"
isaaclab_tasks_source = "/workspace/isaaclab/source/isaaclab_tasks"
parkour_source = "/workspace/parkour"
workspace = "/workspace"

for path in [isaaclab_tasks_source, isaaclab_source, parkour_source, workspace]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Don't set CUDA_VISIBLE_DEVICES - it can cause conflicts with Isaac Sim
# Let AppLauncher handle CUDA device selection

# Check if Isaac Sim is already initialized (when using /isaac-sim/python.sh)
# /isaac-sim/python.sh partially initializes Isaac Sim, so we should use existing instance
# to avoid CUDA context conflicts
current_step = "[STEP] Checking for existing Isaac Sim instance..."
logger.info(current_step)
simulation_app = None
app_launcher = None

# When using /isaac-sim/python.sh, Isaac Sim is partially initialized
# Check if there's an existing SimulationApp instance we can reuse
# This avoids creating a new CUDA context which causes conflicts
try:
    simulation_app = SimulationApp.get_instance()
    if simulation_app is not None:
        current_step = "[STEP] Found existing SimulationApp instance, reusing it"
        logger.info(current_step)
        # Verify it's running
        if not simulation_app.is_running():
            current_step = "[WARNING] SimulationApp exists but not running, waiting..."
            logger.warning(current_step)
            max_wait = 10
            waited = 0
            while not simulation_app.is_running() and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
        app_launcher = None  # Don't create a new one
    else:
        simulation_app = None
except (AttributeError, TypeError):
    # get_instance() method not available or no existing instance
    simulation_app = None

# If no existing instance, initialize via AppLauncher
# AppLauncher will handle CUDA context properly
if simulation_app is None:
    current_step = "[STEP] Initializing AppLauncher (no existing instance found)..."
    logger.info(current_step)
    try:
        # Create a namespace with headless=True for AppLauncher
        applauncher_ns = argparse.Namespace(headless=True)
        app_launcher = AppLauncher(applauncher_ns)
        simulation_app = app_launcher.app
        
        # Wait for simulation app to be fully ready
        max_wait = 30
        waited = 0
        while not simulation_app.is_running() and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1
        
        if not simulation_app.is_running():
            raise RuntimeError("SimulationApp failed to start within timeout")
        
        current_step = "[STEP] Isaac Sim initialized successfully via AppLauncher"
        logger.info(current_step)
    except Exception as e:
        current_step = f"[ERROR] Failed to initialize AppLauncher: {e}"
        logger.error(current_step)
        traceback.print_exc()
        sys.exit(1)

# Isaac Lab / parkour imports (after AppLauncher). Deferred so test_isaacsim_noop does not
# need parkour_tasks; register gym envs before parkour_isaaclab.envs (avoids circular import).
_env_stack_loaded = False
parse_env_cfg = None
ParkourManagerBasedRLEnv = None
IsaacSimHalServer = None
HalServerConfig = None
HalClientConfig = None
ModelWeights = None
ParkourInferenceClient = None
torch = None


def _import_env_stack() -> None:
    global _env_stack_loaded
    global parse_env_cfg, ParkourManagerBasedRLEnv, IsaacSimHalServer, HalServerConfig
    global HalClientConfig, ModelWeights, ParkourInferenceClient, torch
    if _env_stack_loaded:
        return
    parkour_tasks_path = "/workspace/parkour/parkour_tasks"
    if parkour_tasks_path not in sys.path:
        sys.path.insert(0, parkour_tasks_path)
    import isaaclab_tasks  # noqa: F401
    import parkour_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg as _parse_env_cfg
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv as _ParkourManagerBasedRLEnv
    from hal.server.isaac import IsaacSimHalServer as _IsaacSimHalServer
    from hal.server import HalServerConfig as _HalServerConfig
    from hal.client.config import HalClientConfig as _HalClientConfig
    from compute.parkour.policy_interface import ModelWeights as _ModelWeights
    from compute.parkour.inference_client import ParkourInferenceClient as _ParkourInferenceClient
    import torch as _torch

    parse_env_cfg = _parse_env_cfg
    ParkourManagerBasedRLEnv = _ParkourManagerBasedRLEnv
    IsaacSimHalServer = _IsaacSimHalServer
    HalServerConfig = _HalServerConfig
    HalClientConfig = _HalClientConfig
    ModelWeights = _ModelWeights
    ParkourInferenceClient = _ParkourInferenceClient
    torch = _torch
    _env_stack_loaded = True

def run_test_isaacsim_noop():
    """Simple no-op test to verify Isaac Sim test infrastructure.
    
    This test verifies that Isaac Sim is accessible and initialized when using
    /isaac-sim/python.sh. When using /isaac-sim/python.sh, Isaac Sim is already
    initialized by the Python interpreter, so we access the existing instance
    rather than trying to initialize it again (which would cause a segfault).
    
    NOTE: Attempting to initialize Isaac Sim again (via AppLauncher or SimulationApp)
    causes segfaults when using /isaac-sim/python.sh, as Isaac Sim is already
    partially initialized. Tests should use the existing instance.
    
    Run this test with:
        PYTHONUNBUFFERED=1 timeout 60 docker run --rm --gpus all \
            --entrypoint /workspace/run_test_runner.sh \
            krabby-isaacsim:latest \
            test_isaacsim_noop
    
    Test result: Exit code 0 = PASS, Exit code 1 = FAIL
    Note: Test output may be mixed with Isaac Sim initialization messages.
    Use exit code to determine pass/fail status.
    """
    global current_step
    
    current_step = "[TEST] test_isaacsim_noop"
    logger.info(f"{current_step}...")
    
    try:
        logger.info("[INFO] test: Verifying Isaac Sim is accessible...")
        
        # Use the global simulation_app that was initialized at module load
        # This is set by AppLauncher which uses the new API
        global simulation_app
        # Verify instance is valid and running
        is_running = simulation_app.is_running()
        logger.info(f"[INFO] test: SimulationApp is_running() = {is_running}")
        if not is_running:
            raise AssertionError("SimulationApp is not running")
        
        logger.info("[INFO] test: Isaac Sim is initialized and accessible")
        
        # Isaac Sim is initialized and accessible
        # We don't close it here since we didn't create it - /isaac-sim/python.sh manages it
        logger.info("[INFO] test: Isaac Sim initialization verified")
        logger.info("[PASS] test_isaacsim_noop passed")
        flush_logs()
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] test_isaacsim_noop failed: {e}")
        traceback.print_exc()
        flush_logs()
        return False

def run_test_isaacsim_create_environment_only():
    """Test that only creates an Isaac Lab environment (for troubleshooting stalls).
    
    This minimal test creates a real Isaac Lab environment without initializing
    the HAL server. Use this to isolate whether stalls occur during environment
    creation or during HAL server initialization.
    """
    logger.info("[TEST] test_isaacsim_create_environment_only...")
    _import_env_stack()
    
    env = None
    try:
        # Create environment configuration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this test but is not available")
        task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
        device = "cuda"
        logger.info(f"[STEP] Creating environment: {task_name}, Device: {device}")
        
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=1,
            use_fabric=True,
        )
        
        # Create environment
        logger.info("[STEP] Creating ParkourManagerBasedRLEnv (this may take 30-60 seconds)...")
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        
        # Verify environment was created successfully
        assert env.num_envs == 1, "Environment should have 1 env"
        
        # Clean up
        env.close()
        
        logger.info("[PASS] test_isaacsim_create_environment_only passed")
        flush_logs()
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] test_isaacsim_create_environment_only failed: {e}")
        traceback.print_exc()
        # Ensure cleanup on error
        if env is not None:
            try:
                env.close()
            except:
                pass
        flush_logs()
        return False

def run_test_isaacsim_hal_server_with_real_isaaclab():
    """Test with real IsaacLab environment.
    
    This test creates a real Isaac Lab environment and initializes the HAL server
    to verify end-to-end integration. Environment creation may take 30-60 seconds
    and can occasionally stall during sensor initialization.
    
    Run this test with:
        PYTHONUNBUFFERED=1 timeout 300 docker run --rm --gpus all \
            --entrypoint /workspace/run_test_runner.sh \
            krabby-isaacsim:latest \
            test_isaacsim_hal_server_with_real_isaaclab
    
    Test result: Exit code 0 = PASS, Exit code 1 = FAIL
    Note: Use a timeout (300 seconds = 5 minutes) as this test may stall during
    environment creation. If it stalls, it's likely waiting for sensor initialization.
    Test output may be mixed with Isaac Sim initialization messages - use exit code
    to determine pass/fail status.
    """
    global current_step
    
    current_step = "[TEST] test_isaacsim_hal_server_with_real_isaaclab"
    logger.info(f"{current_step}...")
    _import_env_stack()
    
    try:
        # Verify Isaac Sim is accessible
        current_step = "[STEP] Verifying Isaac Sim..."
        logger.info(current_step)
        assert simulation_app is not None and simulation_app != "available", "Isaac Sim should be accessible"
        if isinstance(simulation_app, str):
            # Old API - just verify modules are available
            logger.info("[STEP] Isaac Sim modules are available (old API)")
        else:
            logger.info("[STEP] Isaac Sim verified")
        
        # Create environment configuration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this test but is not available")
        task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
        device = "cuda"
        current_step = f"[STEP] Task: {task_name}, Device: {device}"
        logger.info(current_step)
        
        # parse_env_cfg will import parkour_tasks internally and register environments
        current_step = f"[STEP] Calling parse_env_cfg for {task_name}..."
        logger.info(current_step)
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=1,
            use_fabric=True,
        )
        current_step = "[STEP] parse_env_cfg completed"
        logger.info(current_step)
        
        # Create environment using direct instantiation
        # NOTE: Environment creation can take 30-60 seconds and may occasionally stall.
        # If it stalls, it's likely waiting for sensor initialization or scene setup.
        # The environment should print "[INFO]: Completed setting up the environment..." when done.
        current_step = "[STEP] Creating ParkourManagerBasedRLEnv (this may take 30-60 seconds, or longer if it stalls)..."
        logger.info(current_step)
        logger.info("[STEP] If this hangs, it may be waiting for sensor initialization or scene setup")
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        current_step = "[STEP] ParkourManagerBasedRLEnv constructor completed"
        logger.info(current_step)
        assert env.num_envs == 1
        current_step = "[STEP] Environment created successfully"
        logger.info(current_step)
        
        # Create HAL server config
        logger.info("[STEP] Creating HAL server config...")
        hal_server_config = HalServerConfig(
            observation_bind="inproc://test_obs",
            command_bind="inproc://test_cmd",
        )
        logger.info("[STEP] HAL server config created")
        
        from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION
        # Create and initialize HAL server with real environment
        logger.info("[STEP] Creating IsaacSimHalServer...")
        hal_server = IsaacSimHalServer(hal_server_config, UNITREE_GO2_DEFINITION, env=env)
        logger.info("[STEP] IsaacSimHalServer created, calling initialize()...")
        hal_server.initialize()
        logger.info("[STEP] HAL server initialized successfully")
        
        # Verify server can publish observation
        logger.info("[STEP] Calling hal_server.set_observation()...")
        hal_server.set_observation()
        logger.info("[STEP] Observation published successfully")
        
        # Clean up
        current_step = "[STEP] Cleaning up..."
        logger.info(current_step)
        hal_server.close()
        logger.info("[STEP] HAL server closed")
        env.close()
        logger.info("[STEP] Environment closed")
        
        logger.info("[PASS] test_isaacsim_hal_server_with_real_isaaclab passed")
        flush_logs()
        return True
        
    except Exception as e:
        logger.error(f"[FAIL] test_isaacsim_hal_server_with_real_isaaclab failed: {e}")
        traceback.print_exc()
        # Ensure cleanup even on failure
        try:
            if 'hal_server' in locals():
                hal_server.close()
                logger.info("[STEP] HAL server closed (cleanup on error)")
        except:
            pass
        try:
            if 'env' in locals():
                env.close()
                logger.info("[STEP] Environment closed (cleanup on error)")
        except:
            pass
        flush_logs()
        return False

def run_test_inference_latency_requirement():
    """Test that inference latency meets < 15ms requirement with real model and Isaac Sim.
    
    This test measures real inference latency over multiple runs with a real Isaac Sim
    environment and HAL server, verifying that average and p99 latency meet the < 15ms
    requirement. It uses an actual checkpoint and real Isaac Sim environment to test
    end-to-end inference performance including Isaac Sim overhead.
    
    The test uses GPU if available (CUDA), falling back to CPU. The < 15ms requirement
    is enforced when running on GPU. CPU inference may not meet this requirement, which is expected.
    
    Requires PARKOUR_CHECKPOINT_PATH environment variable to be set to the checkpoint file path.
    
    Run this test with:
        PARKOUR_CHECKPOINT_PATH=/workspace/test_assets/checkpoints \
        PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \
            --entrypoint /workspace/run_test_runner.sh \
            krabby-isaacsim:latest \
            test_inference_latency_requirement
    
    Test result: Exit code 0 = PASS, Exit code 1 = FAIL
    Note: Use a longer timeout (600 seconds = 10 minutes) as this test creates an environment
    and runs multiple inference iterations. Test output may be mixed with Isaac Sim
    initialization messages - use exit code to determine pass/fail status.
    """
    global current_step
    
    current_step = "[TEST] test_inference_latency_requirement"
    logger.info(f"{current_step}...")
    _import_env_stack()
    
    try:
        # Find checkpoint path
        current_step = "[STEP] Finding checkpoint path..."
        logger.info(current_step)
        checkpoint_name = "unitree_go2_parkour_teacher.pt"
        env_path = os.getenv("PARKOUR_CHECKPOINT_PATH")
        if not env_path:
            raise FileNotFoundError(
                "PARKOUR_CHECKPOINT_PATH environment variable is not set. "
                "Set it to the path of the checkpoint folder."
            )
        
        checkpoint_dir = Path(env_path)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}\n"
                f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
            )
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}\n"
                f"Expected: {checkpoint_dir / checkpoint_name}\n"
                f"PARKOUR_CHECKPOINT_PATH is set to: {env_path}"
            )
        
        logger.info(f"[STEP] Found checkpoint: {checkpoint_path}")
        
        # Create environment configuration
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this test but is not available")
        task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"[STEP] Task: {task_name}, Device: {device}")
        
        # Parse environment configuration
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=1,
            use_fabric=True,
        )
        
        # Create environment
        logger.info("[STEP] Creating ParkourManagerBasedRLEnv (this may take 30-60 seconds)...")
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        assert env.num_envs == 1
        
        # Create HAL server config (use inproc for same-process communication)
        hal_server_config = HalServerConfig(
            observation_bind="inproc://test_obs_latency",
            command_bind="inproc://test_cmd_latency",
        )
        
        # Reset environment before creating HAL server to ensure deterministic initial state
        logger.info("[STEP] Resetting environment...")
        env.reset()
        logger.info("[STEP] Environment reset complete")
        
        from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION
        from hal.server.robot_definition_unitree_go2 import UNITREE_GO2_DEFINITION

        robot_definition = UNITREE_GO2_DEFINITION
        observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
            robot_definition
        )
        action_dim = robot_definition.get_total_joint_count()
        # Create and initialize HAL server with real environment
        logger.info("[STEP] Creating and initializing HAL server...")
        server = IsaacSimHalServer(
            hal_server_config, robot_definition, env=env, observation_dimensions=observation_dimensions
        )
        server.initialize()
        logger.info("[STEP] HAL server initialized")
        
        current_step = "[STEP] Loading model checkpoint..."
        logger.info(current_step)
        weights = ModelWeights(
            checkpoint_path=str(checkpoint_path),
            observation_dimensions=observation_dimensions,
            action_dim=action_dim,
        )
        client_config = HalClientConfig(
            observation_endpoint="inproc://test_obs_latency",
            command_endpoint="inproc://test_cmd_latency",
        )
        logger.info("[STEP] Creating ParkourInferenceClient...")
        inference_client = ParkourInferenceClient(
            hal_client_config=client_config,
            model_weights=weights,
            observation_dimensions=observation_dimensions,
            robot_definition=robot_definition,
            control_rate=100.0,
            device=device,
            transport_context=server.get_transport_context(),
        )
        inference_client.initialize()
        logger.info("[STEP] ParkourInferenceClient initialized")
        
        # Start inference client thread (matching main.py)
        logger.info("[STEP] Starting inference client thread...")
        measurement_running = True
        inference_client.start_thread(running_flag=lambda: measurement_running)
        
        # Run main loop in main thread (matching main.py sequence)
        logger.info(f"[STEP] Starting main loop at 100 Hz (matching main.py sequence)...")
        period_s = 1.0 / 100.0
        timestep = 0
        
        # Latency tracking: measure time from set_observation() to apply_command() returning
        latencies: list[float] = []
        
        # Publish initial observation from environment (matching main.py)
        # First inference is for warmup only - don't capture latency
        logger.info("[STEP] Publishing initial observation (warmup inference)...")
        server.set_observation()
        
        # Wait for first action from inference client (matching main.py)
        # apply_command() will loop internally until command received or timeout (throws if timeout)
        logger.info("[STEP] Waiting for first action from inference client (warmup)...")
        first_action = server.apply_command()
        # Skip latency measurement for first inference (warmup)
        
        if first_action.shape[0] == 1 and env.unwrapped.num_envs > 1:
            first_action = first_action.expand(env.unwrapped.num_envs, -1)
        
        # Apply first action and step environment (matching main.py)
        obs_dict, _, _, _, extras = env.step(first_action)
        # Must match hal.server.isaac.main: cache env.step() policy obs for the next set_observation()
        server._latest_obs_dict = obs_dict
        server._latest_obs_tensor = obs_dict["policy"]

        # Track first applied action for next observation's previous_action
        action_np = first_action[0].cpu().numpy() if first_action.ndim == 2 else first_action.cpu().numpy()
        if len(action_np) >= action_dim:
            server._last_applied_action[:] = action_np[:action_dim].astype(np.float32)
        else:
            server._last_applied_action[: len(action_np)] = action_np.astype(np.float32)
        
        timestep += 1
        
        # Main loop: step simulation and publish observations (matching main.py)
        # Run for 0.5 seconds (50 cycles at 100 Hz) to collect latency samples
        logger.info("[STEP] Running main loop for 0.5 seconds to collect latency samples...")
        target_cycles = 50
        start_time = time.time()
        
        for cycle in range(target_cycles):
            loop_start_ns = time.time_ns()
            
            # Publish hardware observations via HAL (matching main.py)
            # Measure latency: time from set_observation() start to apply_command() returning
            obs_timestamp_ns = time.time_ns()
            server.set_observation()
            
            # Wait for action corresponding to the observation just published (matching main.py)
            action = server.apply_command()
            action_latency_ns = time.time_ns() - obs_timestamp_ns
            action_latency_ms = action_latency_ns / 1e6
            latencies.append(action_latency_ms)
            
            if action.shape[0] == 1 and env.unwrapped.num_envs > 1:
                action = action.expand(env.unwrapped.num_envs, -1)
            
            # Step environment (matching main.py)
            obs_dict, _, _, _, extras = env.step(action)
            server._latest_obs_dict = obs_dict
            server._latest_obs_tensor = obs_dict["policy"]

            # Track last applied action for next observation's previous_action
            action_np = action[0].cpu().numpy() if action.ndim == 2 else action.cpu().numpy()
            if len(action_np) >= action_dim:
                server._last_applied_action[:] = action_np[:action_dim].astype(np.float32)
            else:
                server._last_applied_action[: len(action_np)] = action_np.astype(np.float32)
            
            timestep += 1
            
            # Timing control (matching main.py)
            loop_end_ns = time.time_ns()
            loop_duration_s = (loop_end_ns - loop_start_ns) / 1e9
            sleep_time = max(0.0, period_s - loop_duration_s)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        elapsed_total = time.time() - start_time
        logger.info(f"[STEP] Completed {target_cycles} cycles in {elapsed_total:.2f} seconds")
        
        # Stop inference client thread
        logger.info("[STEP] Stopping inference client thread...")
        measurement_running = False
        inference_client.stop_thread()
        inference_client.close()
        
        # Clean up resources BEFORE assertions
        # This ensures cleanup happens even if latency requirements fail
        logger.info("[STEP] Cleaning up resources...")
        try:
            server.close()
            logger.info("[STEP] HAL server closed")
        except Exception as e:
            logger.warning(f"[WARNING] Error closing server: {e}")
        try:
            env.close()
            logger.info("[STEP] Environment closed")
        except Exception as e:
            logger.warning(f"[WARNING] Error closing environment: {e}")
        
        # Analyze latencies
        if len(latencies) == 0:
            raise AssertionError("No latency samples collected - inference may not have run")
        
        latencies_array = np.array(latencies)
        avg_latency = np.mean(latencies_array)
        p50_latency = np.percentile(latencies_array, 50)
        p95_latency = np.percentile(latencies_array, 95)
        p99_latency = np.percentile(latencies_array, 99)
        max_latency = np.max(latencies_array)
        
        # Log statistics
        logger.info(f"\n[STEP] Inference Latency Statistics ({device_name}):")
        logger.info(f"  Device: {device_name}")
        logger.info(f"  Checkpoint: {checkpoint_path}")
        logger.info(f"  Samples: {len(latencies_array)}")
        logger.info(f"  Average: {avg_latency:.2f}ms")
        logger.info(f"  P50: {p50_latency:.2f}ms")
        logger.info(f"  P95: {p95_latency:.2f}ms")
        logger.info(f"  P99: {p99_latency:.2f}ms")
        logger.info(f"  Max: {max_latency:.2f}ms")
        
        # Verify latency requirements
        # Note: Cleanup already happened above, so failure here won't cause resource leaks
        test_passed = True
        if avg_latency >= 15.0:
            logger.error(f"[FAIL] Average latency {avg_latency:.2f}ms exceeds 15ms requirement on GPU.")
            test_passed = False
        if p99_latency >= 15.0:
            logger.error(f"[FAIL] P99 latency {p99_latency:.2f}ms exceeds 15ms requirement on GPU.")
            test_passed = False
        if test_passed:
            logger.info("[STEP] Latency requirements met on GPU")
        
        if test_passed:
            logger.info("[PASS] test_inference_latency_requirement passed")
            flush_logs()
            return True
        else:
            logger.error("[FAIL] test_inference_latency_requirement failed: Latency requirements not met")
            flush_logs()
            return False
        
    except Exception as e:
        logger.error(f"[FAIL] test_inference_latency_requirement failed: {e}")
        traceback.print_exc()
        # Ensure cleanup even on failure - stop threads and close resources
        try:
            if 'measurement_running' in locals():
                measurement_running = False
                logger.info("[STEP] Inference client stopped (cleanup on error)")
        except:
            pass
        try:
            if 'inference_client' in locals():
                inference_client.stop_thread()
                inference_client.close()
                logger.info("[STEP] Inference client closed (cleanup on error)")
        except:
            pass
        try:
            if 'server' in locals():
                server.close()
                logger.info("[STEP] HAL server closed (cleanup on error)")
        except:
            pass
        try:
            if 'env' in locals():
                env.close()
                logger.info("[STEP] Environment closed (cleanup on error)")
        except:
            pass
        flush_logs()
        return False

def main():
    """Main entry point."""
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    tests = {
        "test_isaacsim_noop": run_test_isaacsim_noop,
        "test_isaacsim_create_environment_only": run_test_isaacsim_create_environment_only,
        "test_isaacsim_hal_server_with_real_isaaclab": run_test_isaacsim_hal_server_with_real_isaaclab,
        "test_inference_latency_requirement": run_test_inference_latency_requirement,
    }
    
    if test_name is None:
        # Run all tests
        logger.info("=" * 80)
        logger.info("Running Isaac Sim Tests")
        logger.info("=" * 80)
        all_passed = True
        for name, test_func in tests.items():
            if not test_func():
                all_passed = False
            # Add a small delay between tests to allow simulation context to clear
            # This prevents "Simulation context already exists" errors
            time.sleep(1.0)
        flush_logs()
        sys.exit(0 if all_passed else 1)
    elif test_name in tests:
        # Run specific test
        success = tests[test_name]()
        flush_logs()
        sys.exit(0 if success else 1)
    else:
        logger.error(f"[ERROR] Unknown test: {test_name}")
        logger.error(f"Available tests: {', '.join(tests.keys())}")
        flush_logs()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n[INTERRUPT] Test interrupted")
        flush_logs()
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        flush_logs()
        sys.exit(1)
    finally:
        # Clean up: close Isaac Sim only if we created it (not if using existing instance)
        if 'app_launcher' in globals() and app_launcher is not None:
            logger.info("[INFO] Closing Isaac Sim...")
            if 'simulation_app' in globals() and simulation_app is not None:
                simulation_app.close()
            logger.info("[INFO] Isaac Sim closed")
            flush_logs()
        # If we used existing instance from /isaac-sim/python.sh, don't close it


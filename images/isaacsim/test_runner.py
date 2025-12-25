#!/usr/bin/env python3
"""Standalone test runner that initializes AppLauncher and runs test functions directly.

This script initializes Isaac Sim via AppLauncher, then runs test functions directly
without pytest. This allows tests to create real Isaac Lab environments since AppLauncher
is properly initialized before any omni modules are imported.

Usage:
    /workspace/testenv/bin/python /workspace/test_runner.py [test_name]
"""

import sys
import os
import signal

# Set up signal handler to print current step on timeout/interrupt
current_step = "[INIT]"

def signal_handler(sig, frame):
    print(f"\n[INTERRUPT] Test interrupted at: {current_step}")
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
print(current_step)
simulation_app = None
app_launcher = None

# When using /isaac-sim/python.sh, Isaac Sim is partially initialized
# Check if there's an existing SimulationApp instance we can reuse
# This avoids creating a new CUDA context which causes conflicts
try:
    from isaacsim.simulation_app import SimulationApp as NewSimulationApp
    simulation_app = NewSimulationApp.get_instance()
    if simulation_app is not None:
        current_step = "[STEP] Found existing SimulationApp instance, reusing it"
        print(current_step)
        # Verify it's running
        if not simulation_app.is_running():
            current_step = "[WARNING] SimulationApp exists but not running, waiting..."
            print(current_step)
            import time
            max_wait = 10
            waited = 0
            while not simulation_app.is_running() and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
        app_launcher = None  # Don't create a new one
    else:
        simulation_app = None
except (ImportError, AttributeError):
    # New API not available or no existing instance
    simulation_app = None

# If no existing instance, initialize via AppLauncher
# AppLauncher will handle CUDA context properly
if simulation_app is None:
    current_step = "[STEP] Initializing AppLauncher (no existing instance found)..."
    print(current_step)
    try:
        from isaaclab.app import AppLauncher
        import argparse as argparse_module
        
        # Create a namespace with headless=True for AppLauncher
        applauncher_ns = argparse_module.Namespace(headless=True)
        app_launcher = AppLauncher(applauncher_ns)
        simulation_app = app_launcher.app
        
        # Wait for simulation app to be fully ready
        import time
        max_wait = 30
        waited = 0
        while not simulation_app.is_running() and waited < max_wait:
            time.sleep(0.1)
            waited += 0.1
        
        if not simulation_app.is_running():
            raise RuntimeError("SimulationApp failed to start within timeout")
        
        current_step = "[STEP] Isaac Sim initialized successfully via AppLauncher"
        print(current_step)
    except Exception as e:
        current_step = f"[ERROR] Failed to initialize AppLauncher: {e}"
        print(current_step)
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Now that Isaac Sim is fully initialized (including CUDA context), we can import Isaac Lab modules
# This is critical - imports must happen AFTER AppLauncher is ready to avoid Warp CUDA errors
def run_test_isaacsim_hal_server_with_real_isaaclab():
    """Test with real IsaacLab environment."""
    global current_step
    
    current_step = "[TEST] test_isaacsim_hal_server_with_real_isaaclab"
    print(f"{current_step}...")
    
    try:
        # Import required modules AFTER AppLauncher is fully initialized
        # This ensures CUDA context is ready before Warp tries to initialize
        # Import in the same order as hal/server/isaac/main.py
        from isaaclab_tasks.utils import parse_env_cfg
        from parkour_isaaclab.envs import ParkourManagerBasedRLEnv
        from hal.server.isaac import IsaacSimHalServer
        from hal.server import HalServerConfig
        import torch
        
        # Import parkour_tasks to register gym environments (after other imports)
        parkour_tasks_path = "/workspace/parkour/parkour_tasks"
        if parkour_tasks_path not in sys.path:
            sys.path.insert(0, parkour_tasks_path)
        import parkour_tasks  # noqa: F401
        
        # Verify Isaac Sim is accessible
        current_step = "[STEP] Verifying Isaac Sim..."
        print(current_step)
        assert simulation_app is not None and simulation_app != "available", "Isaac Sim should be accessible"
        if isinstance(simulation_app, str):
            # Old API - just verify modules are available
            print("[STEP] Isaac Sim modules are available (old API)")
        else:
            print("[STEP] Isaac Sim verified")
        
        # Create environment configuration
        task_name = "Isaac-Extreme-Parkour-Teacher-Unitree-Go2-v0"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        current_step = f"[STEP] Task: {task_name}, Device: {device}"
        print(current_step)
        
        # parse_env_cfg will import parkour_tasks internally and register environments
        current_step = f"[STEP] Calling parse_env_cfg for {task_name}..."
        print(current_step)
        env_cfg = parse_env_cfg(
            task_name,
            device=device,
            num_envs=1,
            use_fabric=True,
        )
        current_step = "[STEP] parse_env_cfg completed"
        print(current_step)
        
        # Create environment using direct instantiation
        current_step = "[STEP] Creating ParkourManagerBasedRLEnv (this may take 30-60 seconds)..."
        print(current_step)
        env = ParkourManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        current_step = "[STEP] ParkourManagerBasedRLEnv constructor completed"
        print(current_step)
        assert env is not None
        assert env.num_envs == 1
        current_step = "[STEP] Environment created successfully"
        print(current_step)
        
        # Create HAL server config
        print("[STEP] Creating HAL server config...")
        hal_server_config = HalServerConfig(
            observation_bind="inproc://test_obs",
            command_bind="inproc://test_cmd",
        )
        print("[STEP] HAL server config created")
        
        # Create and initialize HAL server with real environment
        print("[STEP] Creating IsaacSimHalServer...")
        hal_server = IsaacSimHalServer(hal_server_config, env=env)
        print("[STEP] IsaacSimHalServer created, calling initialize()...")
        hal_server.initialize()
        print("[STEP] HAL server initialized successfully")
        
        # Verify server can publish observation
        print("[STEP] Calling hal_server.set_observation()...")
        hal_server.set_observation()
        print("[STEP] Observation published successfully")
        
        # Clean up
        hal_server.close()
        print("[STEP] HAL server closed")
        
        print("[PASS] test_isaacsim_hal_server_with_real_isaaclab passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] test_isaacsim_hal_server_with_real_isaaclab failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    test_name = sys.argv[1] if len(sys.argv) > 1 else None
    
    tests = {
        "test_isaacsim_hal_server_with_real_isaaclab": run_test_isaacsim_hal_server_with_real_isaaclab,
    }
    
    if test_name is None:
        # Run all tests
        print("=" * 80)
        print("Running Isaac Sim Tests")
        print("=" * 80)
        all_passed = True
        for name, test_func in tests.items():
            if not test_func():
                all_passed = False
        sys.exit(0 if all_passed else 1)
    elif test_name in tests:
        # Run specific test
        success = tests[test_name]()
        sys.exit(0 if success else 1)
    else:
        print(f"[ERROR] Unknown test: {test_name}")
        print(f"Available tests: {', '.join(tests.keys())}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Test interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up: close Isaac Sim only if we created it (not if using existing instance)
        if 'app_launcher' in globals() and app_launcher is not None:
            print("[INFO] Closing Isaac Sim...")
            if 'simulation_app' in globals() and simulation_app is not None:
                simulation_app.close()
            print("[INFO] Isaac Sim closed")
        # If we used existing instance from /isaac-sim/python.sh, don't close it


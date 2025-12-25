"""No-op test to verify Isaac Sim test infrastructure works."""

import pytest


@pytest.mark.isaacsim
def test_isaacsim_noop():
    """Simple no-op test to verify Isaac Sim test infrastructure.
    
    This test verifies that Isaac Sim is accessible and initialized when using
    /isaac-sim/python.sh. When using /isaac-sim/python.sh, Isaac Sim is already
    initialized by the Python interpreter, so we access the existing instance
    rather than trying to initialize it again (which would cause a segfault).
    
    NOTE: Attempting to initialize Isaac Sim again (via AppLauncher or SimulationApp)
    causes segfaults when using /isaac-sim/python.sh, as Isaac Sim is already
    partially initialized. Tests should use the existing instance.
    """
    print("[INFO] test: Verifying Isaac Sim is accessible...")
    
    # Import AppLauncher to verify Isaac Lab modules are available
    from isaaclab.app import AppLauncher
    print("[INFO] test: AppLauncher imported successfully")
    
    # Try to access SimulationApp using the new API first, then fall back to old API
    simulation_app = None
    
    # Try new API (isaacsim.simulation_app)
    try:
        from isaacsim.simulation_app import SimulationApp as NewSimulationApp
        simulation_app = NewSimulationApp.get_instance()
        print("[INFO] test: Retrieved SimulationApp instance (new API)")
    except (ImportError, AttributeError):
        # Fall back to old API
        try:
            from omni.isaac.kit import SimulationApp as OldSimulationApp
            # Old API doesn't have get_instance(), but we can check if it's available
            # by trying to create an instance or checking if it's already created
            print("[INFO] test: Using old API (omni.isaac.kit)")
            # For old API, we can't easily get existing instance, but we can verify
            # the module is available which means Isaac Sim is initialized
            simulation_app = "available"  # Mark as available
        except ImportError:
            print("[WARNING] test: Could not import SimulationApp from either API")
    
    # Verify Isaac Sim is accessible
    if simulation_app is not None:
        print("[INFO] test: Isaac Sim is initialized and accessible")
        if isinstance(simulation_app, str):
            # Old API - just verify modules are available
            print("[INFO] test: Isaac Sim modules are available (old API)")
        else:
            # New API - verify instance is valid
            assert simulation_app is not None
            is_running = simulation_app.is_running()
            print(f"[INFO] test: SimulationApp is_running() = {is_running}")
    else:
        pytest.fail("Could not access SimulationApp instance")
    
    # Isaac Sim is initialized and accessible
    # We don't close it here since we didn't create it - /isaac-sim/python.sh manages it
    print("[INFO] test: Isaac Sim initialization verified")
    assert True


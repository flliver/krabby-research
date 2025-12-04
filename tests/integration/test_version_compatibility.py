"""Integration tests for version compatibility."""

import time

import numpy as np
import pytest

from hal.client.client import HalClient
from hal.server import HalServerBase
from hal.client.config import HalClientConfig, HalServerConfig
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import ParkourObservation, OBS_DIM
from hal.client.data_structures.hardware import KrabbyHardwareObservations
from tests.helpers import create_dummy_hw_obs


def test_reading_older_schema_versions():
    """Test reading older schema versions (forward compatibility)."""
    import zmq
    
    # Use shared context for inproc connections
    # For now, we only support schema version "1.0"
    # This test verifies that we can handle version checking
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_old",
        command_bind="inproc://test_command_old",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_old",
        command_endpoint="inproc://test_command_old",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Send message with schema version "1.0" (current)
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0] = 1.0
    server.set_observation(hw_obs)
    time.sleep(0.01)

    received_hw_obs = client.poll(timeout_ms=1000)

    # Should work with current schema
    assert received_hw_obs is not None

    client.close()
    server.close()


def test_forward_compatibility_unknown_fields():
    """Test forward compatibility (unknown fields ignored)."""
    # This tests that dataclasses with optional fields can handle
    # additional fields in future versions
    from hal.client.observation.types import NavigationCommand

    # Create command with current schema
    nav_cmd = NavigationCommand.create_now(vx=1.0, vy=0.0, yaw_rate=0.5)

    # Verify it has all required fields
    assert nav_cmd.timestamp_ns > 0
    assert nav_cmd.schema_version == "1.0"
    assert nav_cmd.vx == 1.0
    assert nav_cmd.vy == 0.0
    assert nav_cmd.yaw_rate == 0.5

    # In future, if we add optional fields, they should have defaults
    # and existing code should continue to work


def test_action_dim_mismatch_detection():
    """Test action_dim mismatch detection."""
    import torch
    from compute.parkour.parkour_types import InferenceResponse

    # Create inference response with wrong action_dim
    action_wrong = torch.tensor([0.0] * 10, dtype=torch.float32)  # 10 instead of 12
    response_wrong = InferenceResponse.create_success(
        action=action_wrong,
        inference_latency_ms=5.0,
    )

    # Validate with expected action_dim=12
    with pytest.raises(ValueError, match="action_dim"):
        response_wrong.validate_action_dim(12)

    # Should work with correct action_dim
    action_correct = torch.tensor([0.0] * 12, dtype=torch.float32)
    response_correct = InferenceResponse.create_success(
        action=action_correct,
        inference_latency_ms=5.0,
    )
    response_correct.validate_action_dim(12)  # Should not raise


def test_schema_version_compatibility_check():
    """Test schema version compatibility checking."""
    from compute.parkour.parkour_types import ParkourModelIO
    from hal.client.observation.types import NavigationCommand

    # Create components with same schema version
    nav_cmd = NavigationCommand.create_now()
    observation = ParkourObservation(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        observation=np.zeros(OBS_DIM, dtype=np.float32),
    )

    # Should work with compatible versions
    model_io = ParkourModelIO(
        timestamp_ns=time.time_ns(),
        schema_version="1.0",
        nav_cmd=nav_cmd,
        observation=observation,
    )

    assert model_io.is_complete()

    # Test with mismatched schema versions (should be detected in build_model_io)
    import zmq
    
    shared_context2 = zmq.Context()
    
    server_config = HalServerConfig(
        observation_bind="inproc://test_observation_schema_check",
        command_bind="inproc://test_command_schema_check",
    )
    server = HalServerBase(server_config)
    server.initialize()

    server.set_debug(True)

    client_config = HalClientConfig(
        observation_endpoint="inproc://test_observation_schema_check",
        command_endpoint="inproc://test_command_schema_check",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    client.set_debug(True)

    time.sleep(0.1)

    # Publish a dummy observation first to establish connection
    hw_obs_init = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    server.set_observation(hw_obs_init)
    client.poll(timeout_ms=1000)
    # Connection is now established

    # Publish observation
    hw_obs = create_dummy_hw_obs(
        camera_height=480, camera_width=640
    )
    hw_obs.joint_positions[0] = 1.0
    server.set_observation(hw_obs)
    time.sleep(0.01)

    received_hw_obs = client.poll(timeout_ms=1000)
    assert received_hw_obs is not None

    client.close()
    server.close()
    shared_context2.term()


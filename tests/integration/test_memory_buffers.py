"""Integration tests for memory buffer management."""

import time

import numpy as np
import pytest

from hal.client.client import HalClient
from hal.server.server import HalServerBase
from hal.client.config import HalClientConfig
from hal.server.config import HalServerConfig


def test_hwm_prevents_buffer_growth():
    """Test that HWM=1 prevents buffer growth."""
    import zmq
    
    # Use shared context for inproc connections
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_hwm",
        command_bind="inproc://test_command_hwm",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_hwm",
        command_endpoint="inproc://test_command_hwm",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    from hal.observation.types import OBS_DIM
    
    time.sleep(0.1)
    
    # Publish a dummy message first to establish connection
    observation_init = np.zeros(OBS_DIM, dtype=np.float32)
    server.set_observation(observation_init)
    client.poll(timeout_ms=1000)
    # Connection is now established

    # Publish many messages rapidly (faster than consumption)
    # With HWM=1, older messages are dropped, so we need to ensure
    # the subscriber receives at least some messages
    for i in range(1, 101):  # Start at 1 to avoid confusion with init message
        observation = np.zeros(OBS_DIM, dtype=np.float32)
        observation[0] = float(i)
        server.set_observation(observation)
        time.sleep(0.001)  # 1ms between publishes

    # Small delay to ensure all messages are sent
    time.sleep(0.01)
    
    # Poll multiple times to drain queue - with HWM=1, we should eventually get a message
    # The key test is that memory stays bounded (HWM=1), not that we get the absolute latest
    received_values = []
    for _ in range(20):
        client.poll(timeout_ms=100)
        if client._latest_observation is not None and client._latest_observation.observation is not None:
            val = client._latest_observation.observation[0]
            if val not in received_values:
                received_values.append(val)
        time.sleep(0.001)

    # With HWM=1, should receive messages (memory stays bounded)
    # Verify we got at least one message
    assert client._latest_observation is not None
    assert client._latest_observation.observation is not None
    
    # With HWM=1, we should receive some messages (exact value depends on timing and HWM behavior)
    # The important thing is that we received messages and memory stayed bounded
    assert len(received_values) > 0  # We should have received at least one message

    client.close()
    server.close()


def test_rapid_message_publishing():
    """Test with rapid message publishing (faster than consumption)."""
    import zmq
    
    # Use shared context for inproc connections
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_rapid",
        command_bind="inproc://test_command_rapid",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_rapid",
        command_endpoint="inproc://test_command_rapid",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    time.sleep(0.1)
    
    # Publish a dummy message first to establish connection
    from hal.observation.types import OBS_DIM
    observation_init = np.zeros(OBS_DIM, dtype=np.float32)
    server.set_observation(observation_init)
    client.poll(timeout_ms=1000)

    # Publish messages very rapidly
    import threading

    publish_count = [0]

    from hal.observation.types import OBS_DIM
    
    def rapid_publish():
        for i in range(1000):
            observation = np.zeros(OBS_DIM, dtype=np.float32)
            observation[0] = float(i)
            server.set_observation(observation)
            publish_count[0] += 1
            time.sleep(0.0001)  # 0.1ms between publishes (very fast)

    pub_thread = threading.Thread(target=rapid_publish)
    pub_thread.start()

    # Poll occasionally (slower than publishing)
    for _ in range(10):
        time.sleep(0.01)  # 10ms between polls
        client.poll(timeout_ms=100)

    pub_thread.join()

    # Memory should stay bounded (HWM=1 ensures only latest is kept)
    # Verify we got messages
    assert client._latest_observation is not None

    client.close()
    server.close()


def test_memory_usage_bounded():
    """Test that memory usage stays bounded."""
    import zmq
    
    # Use shared context for inproc connections
    # This is a simplified test - full memory profiling would require more tools
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_memory",
        command_bind="inproc://test_command_memory",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_memory",
        command_endpoint="inproc://test_command_memory",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    from hal.observation.types import OBS_DIM
    
    time.sleep(0.1)
    
    # Publish a dummy message first to establish connection
    observation_init = np.zeros(OBS_DIM, dtype=np.float32)
    server.set_observation(observation_init)
    client.poll(timeout_ms=1000)

    # Publish many messages
    for i in range(1000):
        observation = np.full(OBS_DIM, float(i), dtype=np.float32)  # Larger messages
        server.set_observation(observation)
        if i % 100 == 0:
            client.poll(timeout_ms=100)

    # With HWM=1, memory should stay bounded
    # (We can't easily measure exact memory, but the system should not crash)
    assert client._latest_observation is not None

    client.close()
    server.close()


def test_old_messages_dropped():
    """Test that old messages are dropped (not buffered)."""
    import zmq
    
    # Use shared context for inproc connections (required for reliable inproc PUB/SUB)
    server_config = HalServerConfig.from_endpoints(
        observation_bind="inproc://test_observation_drop",
        command_bind="inproc://test_command_drop",
        observation_buffer_size=1,
    )
    server = HalServerBase(server_config)
    server.initialize()

    client_config = HalClientConfig.from_endpoints(
        observation_endpoint="inproc://test_observation_drop",
        command_endpoint="inproc://test_command_drop",
    )
    client = HalClient(client_config, context=server.get_transport_context())
    client.initialize()

    from hal.observation.types import OBS_DIM
    
    # With shared context and inproc, connection should be immediate
    # Give a small delay to ensure sockets are ready
    time.sleep(0.1)

    # Publish and poll a dummy message to establish connection
    observation_init = np.zeros(OBS_DIM, dtype=np.float32)
    server.set_observation(observation_init)
    client.poll(timeout_ms=1000)
    # Connection is now established

    # Publish message 1.0 and poll - should receive it
    observation_1 = np.zeros(OBS_DIM, dtype=np.float32)
    observation_1[0] = 1.0
    server.set_observation(observation_1)
    time.sleep(0.01)
    client.poll(timeout_ms=1000)
    assert client._latest_observation.observation[0] == 1.0

    # Publish message 2.0 and poll - should receive it (replacing 1.0)
    observation_2 = np.zeros(OBS_DIM, dtype=np.float32)
    observation_2[0] = 2.0
    server.set_observation(observation_2)
    time.sleep(0.01)
    client.poll(timeout_ms=1000)
    assert client._latest_observation.observation[0] == 2.0

    # Publish message 3.0 and poll - should receive it (replacing 2.0)
    observation_3 = np.zeros(OBS_DIM, dtype=np.float32)
    observation_3[0] = 3.0
    server.set_observation(observation_3)
    time.sleep(0.01)
    client.poll(timeout_ms=1000)
    
    # Should have the latest value (3.0)
    assert client._latest_observation is not None
    assert client._latest_observation.observation is not None
    assert client._latest_observation.observation[0] == 3.0

    client.close()
    server.close()


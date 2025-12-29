"""Mock HAL server for inference testing.

This provides a simple HAL server that publishes synthetic observations
for testing inference without requiring Isaac Sim or real hardware.
"""

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from hal.client.data_structures.hardware import HardwareObservations
from hal.server import HalServerConfig
from hal.server.server import HalServerBase

logger = logging.getLogger(__name__)


class MockHalServer(HalServerBase):
    """Mock HAL server that publishes synthetic observations for inference testing.
    
    This server generates synthetic hardware observations and publishes them
    at a fixed rate. It also receives and logs joint commands from the inference client.
    """

    def __init__(self, config: HalServerConfig, camera_height: int = 240, camera_width: int = 320):
        """Initialize test HAL server.
        
        Args:
            config: HAL server configuration
            camera_height: Height of synthetic camera images
            camera_width: Width of synthetic camera images
        """
        super().__init__(config)
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.tick_count = 0
        self.observations_sent = 0
        self.commands_received = 0
        self._running = False
        self._publish_thread: Optional[threading.Thread] = None
        self._command_thread: Optional[threading.Thread] = None
        self._stats_thread: Optional[threading.Thread] = None

    def start(self, rate_hz: float = 100.0):
        """Start publishing observations at specified rate.
        
        Args:
            rate_hz: Publication rate in Hz
        """
        if self._running:
            return

        self._running = True
        period = 1.0 / rate_hz

        def publish_loop():
            """Background loop that publishes synthetic observations at a fixed rate."""
            logger.info(f"Starting observation publishing loop at {rate_hz} Hz")
            while self._running:
                try:
                    # Create synthetic observation
                    hw_obs = self._create_synthetic_observation()
                    self.set_observation(hw_obs)
                    self.tick_count += 1
                    self.observations_sent += 1
                    time.sleep(period)
                except Exception as e:
                    if not self._running:
                        break
                    logger.error(f"Error in publish loop: {e}", exc_info=True)

        def command_loop():
            """Handle incoming commands in a loop."""
            while self._running:
                try:
                    # Poll for commands (non-blocking with short timeout)
                    command = self.get_joint_command(timeout_ms=10)
                    if command is not None:
                        self.commands_received += 1
                except Exception as e:
                    if not self._running:
                        break
                    logger.debug(f"Exception in command loop (continuing): {e}")
        
        def stats_loop():
            """Periodically log statistics."""
            while self._running:
                time.sleep(5.0)  # Log every 5 seconds
                if self._running:
                    logger.info(
                        f"Mock HAL server stats: "
                        f"observations_sent={self.observations_sent}, "
                        f"commands_received={self.commands_received}, "
                        f"tick_count={self.tick_count}"
                    )

        self._publish_thread = threading.Thread(target=publish_loop, daemon=True)
        self._publish_thread.start()
        
        self._command_thread = threading.Thread(target=command_loop, daemon=True)
        self._command_thread.start()
        
        self._stats_thread = threading.Thread(target=stats_loop, daemon=True)
        self._stats_thread.start()

    def stop(self):
        """Stop publishing observations."""
        self._running = False
        if self._publish_thread:
            self._publish_thread.join(timeout=1.0)
        if self._command_thread:
            self._command_thread.join(timeout=1.0)
        if self._stats_thread:
            self._stats_thread.join(timeout=1.0)
        logger.info(
            f"Mock HAL server stopped. Final stats: "
            f"observations_sent={self.observations_sent}, "
            f"commands_received={self.commands_received}, "
            f"tick_count={self.tick_count}"
        )

    def _create_synthetic_observation(self) -> HardwareObservations:
        """Create a synthetic hardware observation.
        
        Returns:
            HardwareObservations with synthetic data
        """
        # Generate synthetic data
        # Joint positions: 18 DOF, simple sine wave pattern
        joint_positions = np.sin(np.arange(18) * 0.1 + self.tick_count * 0.01).astype(np.float32)
        
        # RGB camera images: simple gradient pattern
        rgb_camera_1 = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        rgb_camera_1[:, :, 0] = (np.arange(self.camera_width) * 255 // self.camera_width).astype(np.uint8)[None, :]
        rgb_camera_1[:, :, 1] = (np.arange(self.camera_height) * 255 // self.camera_height).astype(np.uint8)[:, None]
        rgb_camera_1[:, :, 2] = np.uint8((self.tick_count * 2) % 256)
        
        rgb_camera_2 = rgb_camera_1.copy()  # Same pattern for second camera
        
        # Depth map: simple gradient
        depth_map = np.linspace(0.5, 5.0, self.camera_height * self.camera_width, dtype=np.float32).reshape(
            (self.camera_height, self.camera_width)
        )
        
        # Confidence map: all ones (full confidence)
        confidence_map = np.ones((self.camera_height, self.camera_width), dtype=np.float32)
        
        # Timestamp
        timestamp_ns = int(time.time_ns())
        
        return HardwareObservations(
            joint_positions=joint_positions,
            rgb_camera_1=rgb_camera_1,
            rgb_camera_2=rgb_camera_2,
            depth_map=depth_map,
            confidence_map=confidence_map,
            camera_height=self.camera_height,
            camera_width=self.camera_width,
            timestamp_ns=timestamp_ns,
        )


def run_mock_hal_server(
    observation_bind: str = "inproc://hal_observation",
    command_bind: str = "inproc://hal_commands",
    rate_hz: float = 100.0,
):
    """Run mock HAL server in a blocking loop.
    
    Args:
        observation_bind: Observation endpoint
        command_bind: Command endpoint
        rate_hz: Publication rate in Hz
    """
    config = HalServerConfig(
        observation_bind=observation_bind,
        command_bind=command_bind,
    )
    
    server = MockHalServer(config)
    server.initialize()
    
    # Set up signal handlers
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Received interrupt signal, stopping mock HAL server...")
        running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        server.start(rate_hz=rate_hz)
        logger.info(f"Mock HAL server running at {rate_hz} Hz. Press Ctrl+C to stop.")
        while running:
            time.sleep(0.1)
    finally:
        server.stop()
        server.close()
        logger.info("Mock HAL server stopped")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Mock HAL server for inference testing")
    parser.add_argument("--observation_bind", type=str, default="inproc://hal_observation", help="Observation endpoint")
    parser.add_argument("--command_bind", type=str, default="inproc://hal_commands", help="Command endpoint")
    parser.add_argument("--rate", type=float, default=100.0, help="Publication rate in Hz")
    
    args = parser.parse_args()
    
    run_mock_hal_server(
        observation_bind=args.observation_bind,
        command_bind=args.command_bind,
        rate_hz=args.rate,
    )


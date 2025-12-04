"""ZED camera integration for Jetson robot deployment.

This module provides a wrapper for the ZED camera SDK to capture depth frames
and convert them to depth features matching the policy model's training format.

Production code: Requires ZED SDK (pyzed) and hardware to be available.
Fails fast if dependencies are missing.
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZedCamera:
    """ZED camera wrapper for depth frame capture and preprocessing.

    Handles ZED SDK initialization, frame capture, and conversion to depth features.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_mode: str = "PERFORMANCE",
        depth_feature_dim: int = 132,
    ):
        """Initialize ZED camera.

        Args:
            resolution: Camera resolution (width, height). Default (640, 480)
            fps: Frames per second. Default 30
            depth_mode: Depth mode ("PERFORMANCE", "QUALITY", "ULTRA"). Default "PERFORMANCE"
            depth_feature_dim: Number of depth features to extract. Default 132

        Raises:
            RuntimeError: If camera initialization fails
        """
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.depth_feature_dim = depth_feature_dim

        self.camera = None
        self.initialized = False
        self.last_frame_time_ns = 0
        self.frame_period_ns = 1_000_000_000 // fps  # Nanoseconds per frame

        # Pre-allocated buffers to avoid allocation in hot path
        self.depth_image = None
        self.feature_buffer = np.zeros(self.depth_feature_dim, dtype=np.float32)

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize ZED SDK and open camera.

        Raises:
            RuntimeError: If camera initialization fails or ZED SDK is not available
        """
        # Import ZED SDK - required for production
        try:
            import pyzed.sl as sl
            self._zed_module = sl
        except ImportError as e:
            raise RuntimeError(
                "ZED SDK (pyzed) not available. "
                "Install pyzed and ensure ZED SDK is installed on the system."
            ) from e

        try:
            # Create camera object
            self.camera = self._zed_module.Camera()

            # Create init parameters
            init_params = self._zed_module.InitParameters()
            init_params.camera_resolution = self._zed_module.RESOLUTION.VGA  # 640x480
            if self.resolution == (1280, 720):
                init_params.camera_resolution = self._zed_module.RESOLUTION.HD720
            elif self.resolution == (1920, 1080):
                init_params.camera_resolution = self._zed_module.RESOLUTION.HD1080

            init_params.camera_fps = self.fps
            init_params.depth_mode = getattr(
                self._zed_module.DEPTH_MODE, self.depth_mode, self._zed_module.DEPTH_MODE.PERFORMANCE
            )
            init_params.coordinate_units = self._zed_module.UNIT.METER
            init_params.coordinate_system = self._zed_module.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            # Open camera
            status = self.camera.open(init_params)
            if status != self._zed_module.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open ZED camera: {status}")

            # Get camera information
            camera_info = self.camera.get_camera_information()
            logger.info(f"ZED camera initialized: {camera_info.camera_model}")
            logger.info(f"Resolution: {self.resolution}, FPS: {self.fps}")

            # Create depth image mat
            self.depth_image = self._zed_module.Mat()

            self.initialized = True
            logger.info("ZED camera initialized successfully")

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"ZED camera initialization failed: {e}") from e

    def close(self) -> None:
        """Close camera and release resources."""
        if self.camera is not None:
            try:
                self.camera.close()
                logger.info("ZED camera closed")
            except Exception as e:
                logger.error(f"Error closing ZED camera: {e}")
            finally:
                self.camera = None
                self.initialized = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def capture_depth_frame(self) -> Optional[np.ndarray]:
        """Capture latest depth frame from ZED camera.

        Returns:
            Depth image as float32 numpy array (height, width) in meters, or None if capture fails

        Raises:
            RuntimeError: If camera not initialized
        """
        if not self.initialized:
            raise RuntimeError("Camera not initialized")

        try:
            # Grab frame
            if self.camera.grab() != self._zed_module.ERROR_CODE.SUCCESS:
                logger.warning("Failed to grab ZED frame")
                return None

            # Retrieve depth image
            self.camera.retrieve_measure(
                self.depth_image, self._zed_module.MEASURE.DEPTH, self._zed_module.MEM.CPU
            )

            # Convert to numpy array
            depth_array = self.depth_image.get_data()
            depth_array = np.asarray(depth_array, dtype=np.float32)

            # Update frame time
            self.last_frame_time_ns = time.time_ns()

            return depth_array

        except Exception as e:
            logger.error(f"Error capturing depth frame: {e}")
            return None

    def _validate_depth_frame(self, depth_frame: np.ndarray) -> bool:
        """Validate depth frame.

        Args:
            depth_frame: Depth frame to validate

        Returns:
            True if valid, False otherwise
        """
        if depth_frame is None:
            return False

        # Check dimensions
        if depth_frame.ndim != 2:
            logger.error(f"Invalid depth frame dimensions: {depth_frame.ndim}, expected 2")
            return False

        # Check for NaN and Inf
        if np.any(np.isnan(depth_frame)) or np.any(np.isinf(depth_frame)):
            logger.warning("Depth frame contains NaN or Inf values")
            return False

        # Check depth range (reasonable values: 0.1m to 10m)
        valid_mask = (depth_frame > 0.1) & (depth_frame < 10.0)
        valid_ratio = np.sum(valid_mask) / depth_frame.size

        if valid_ratio < 0.5:  # Less than 50% valid pixels
            logger.warning(f"Low valid depth ratio: {valid_ratio:.2%}")
            return False

        return True

    def _extract_depth_features(self, depth_frame: np.ndarray) -> np.ndarray:
        """Extract depth features from depth frame.

        Simulates ray-casting by:
        1. Sampling points from depth image (ray-like pattern)
        2. Converting depth to height measurements (relative to camera)
        3. Applying normalization: clip(height - 0.3, -1, 1)

        Args:
            depth_frame: Depth frame (height, width) in meters

        Returns:
            Depth features as float32 array of shape (depth_feature_dim,)
        """
        height, width = depth_frame.shape

        # Number of features to extract (configurable)
        num_features = self.depth_feature_dim

        # Sample points from depth image to simulate ray-casting
        # Use a pattern that covers the field of view (similar to ray scanner)
        # For a typical ray scanner, rays are cast in a fan pattern
        # We'll sample points in a grid pattern that covers the image

        # Calculate sampling pattern dynamically based on num_features
        # Find grid dimensions that approximate num_features
        # Use approximate square root to get roughly square grid
        grid_rows = int(np.sqrt(num_features))
        grid_cols = (num_features + grid_rows - 1) // grid_rows  # Ceiling division

        features = np.zeros(num_features, dtype=np.float32)

        row_indices = np.linspace(0, height - 1, grid_rows, dtype=np.int32)
        col_indices = np.linspace(0, width - 1, grid_cols, dtype=np.int32)

        idx = 0
        for row in row_indices:
            for col in col_indices:
                if idx < num_features:
                    # Get depth value at this point
                    depth_value = depth_frame[row, col]

                    # Convert depth to height measurement
                    # Training formula: camera_z - hit_z - 0.3
                    # For ZED camera, depth is distance from camera
                    # Height relative to camera: depth (in camera frame, Z is forward)
                    # We approximate: height = depth (simplified, assumes flat ground)
                    # More accurate would require camera pose and ground plane estimation
                    height_measurement = depth_value - 0.3  # Apply offset like training

                    # Apply same normalization as training: clip to [-1, 1]
                    features[idx] = np.clip(height_measurement, -1.0, 1.0)
                    idx += 1

        # Ensure we have exactly num_features
        if idx < num_features:
            # Pad with zeros if needed (shouldn't happen with correct grid calculation)
            features[idx:] = 0.0

        return features.astype(np.float32)

    def get_depth_features(self) -> Optional[np.ndarray]:
        """Get depth features with preprocessing.

        Captures depth frame, validates, extracts features, and returns feature array.

        Returns:
            Depth features as float32 array of shape (depth_feature_dim,), or None if capture/processing fails
        """
        # Capture depth frame
        depth_frame = self.capture_depth_frame()
        if depth_frame is None:
            return None

        # Validate frame
        if not self._validate_depth_frame(depth_frame):
            logger.warning("Depth frame validation failed, using previous frame or returning None")
            return None

        # Extract features
        try:
            features = self._extract_depth_features(depth_frame)
        except Exception as e:
            logger.error(f"Error extracting depth features: {e}")
            return None

        # Validate feature shape
        if len(features) != self.depth_feature_dim:
            logger.error(
                f"Feature dimension mismatch: {len(features)} != {self.depth_feature_dim}"
            )
            return None

        return features

    def is_ready(self) -> bool:
        """Check if camera is ready to capture frames.

        Returns:
            True if camera is initialized and ready
        """
        return self.initialized

    def get_frame_rate(self) -> float:
        """Get actual frame rate.

        Returns:
            Frame rate in Hz
        """
        if self.last_frame_time_ns == 0:
            return 0.0

        # Calculate from last frame time
        # This is a simplified version - could track multiple frame times for better accuracy
        return self.fps  # Return configured FPS for now


def create_zed_camera(
    resolution: tuple[int, int] = (640, 480),
    fps: int = 30,
    depth_mode: str = "PERFORMANCE",
    depth_feature_dim: int = 132,
) -> Optional[ZedCamera]:
    """Factory function to create ZED camera with error handling.

    Args:
        resolution: Camera resolution (width, height)
        fps: Frames per second
        depth_mode: Depth mode
        depth_feature_dim: Expected depth feature dimension. Default 132

    Returns:
        ZedCamera instance if successful, None if initialization fails
    """
    try:
        camera = ZedCamera(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            depth_feature_dim=depth_feature_dim,
        )
        return camera
    except RuntimeError as e:
        logger.error(f"Failed to create ZED camera: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating ZED camera: {e}", exc_info=True)
        return None


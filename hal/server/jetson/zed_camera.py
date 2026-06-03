"""ZED camera integration for Jetson robot deployment.

This module provides a wrapper for the ZED camera SDK to capture RGB and depth (meters).
ZedCamera implements the RgbDepthCamera protocol (see rgb_depth_camera.py).
Policy scan vectors are built in JetsonHalServer from the same depth frames.

Production code: Requires ZED SDK (pyzed) and hardware to be available.
Fails fast if dependencies are missing.
"""

import logging
import time
import ctypes
from pathlib import Path
from typing import Optional

import numpy as np

from hal.server.jetson.rgb_depth_camera import RgbDepthCamera
from hal.server.jetson.zed_imu import ZedImuSample, parse_zed_imu_data
from hal.server.jetson.zed_tracking import (
    parse_zed_tracking_lin_vel_sensor,
    positional_tracking_state_ok,
    tracking_reference_frame_camera,
)

logger = logging.getLogger(__name__)
_JETSON_NVJPEG_LIB = Path("/usr/lib/aarch64-linux-gnu/nvidia/libnvjpeg.so")


def _preload_jetson_nvjpeg() -> None:
    """Prefer Jetson camera-stack nvjpeg before importing pyzed.

    On Jetson containers that also carry CUDA, both provide `libnvjpeg.so`.
    ZED/Argus requires the Jetson camera-stack variant under
    `/usr/lib/aarch64-linux-gnu/nvidia`. If the CUDA one is resolved first,
    `import pyzed.sl` can fail with an undefined JPEG hardware symbol.
    """
    if not _JETSON_NVJPEG_LIB.exists():
        return
    try:
        ctypes.CDLL(str(_JETSON_NVJPEG_LIB), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        logger.debug("Could not preload Jetson nvjpeg library: %s", e)


class ZedCamera(RgbDepthCamera):
    """ZED camera wrapper implementing RgbDepthCamera.

    Handles ZED SDK initialization and frame capture.
    """

    def __init__(
        self,
        depth_mode: str,
        resolution: tuple[int, int],
        fps: int,
        serial_number: Optional[int] = None,
    ):
        """Initialize ZED camera.

        Args:
            resolution: Camera resolution (width, height). Default (640, 480)
            fps: Frames per second. Default 30
            depth_mode: Depth mode (e.g. "NEURAL", "NEURAL_LIGHT", "NEURAL_PLUS",
                "QUALITY", "ULTRA", "PERFORMANCE").
            serial_number: If set, open this USB ZED (Stereolabs serial). Default opens first device.

        Raises:
            RuntimeError: If camera initialization fails
        """
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.serial_number = serial_number

        self.camera = None
        self.initialized = False
        self.last_frame_time_ns = 0
        self.frame_period_ns = 1_000_000_000 // fps  # Nanoseconds per frame

        # Pre-allocated buffers to avoid allocation in hot path
        self.depth_image = None
        # Last captured frame (one grab fills both; used by get_rgb_image/get_depth_map)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_depth_np: Optional[np.ndarray] = None
        self._has_imu: bool = False
        self._sensors_data = None
        self._last_imu: Optional[ZedImuSample] = None
        self._has_tracking: bool = False
        self._tracking_pose = None
        self._last_tracking_lin_vel_sensor: Optional[np.ndarray] = None

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        """Initialize ZED SDK and open camera.

        Raises:
            RuntimeError: If camera initialization fails or ZED SDK is not available
        """
        _preload_jetson_nvjpeg()
        # Import ZED SDK - required for production
        try:
            import pyzed.sl as sl
            self._zed_module = sl
        except ImportError as e:
            raise RuntimeError(
                "ZED SDK (pyzed) import failed: "
                f"{e}. Install pyzed and ensure ZED SDK/runtime libraries are installed."
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
            depth_mode_map = {
                "NEURAL_LIGHT": getattr(
                    self._zed_module.DEPTH_MODE, "NEURAL_LIGHT", self._zed_module.DEPTH_MODE.NEURAL
                ),
                "NEURAL": self._zed_module.DEPTH_MODE.NEURAL,
                "NEURAL_PLUS": getattr(
                    self._zed_module.DEPTH_MODE, "NEURAL_PLUS", self._zed_module.DEPTH_MODE.NEURAL
                ),
                "QUALITY": self._zed_module.DEPTH_MODE.QUALITY,
                "ULTRA": self._zed_module.DEPTH_MODE.ULTRA,
                "PERFORMANCE": self._zed_module.DEPTH_MODE.PERFORMANCE,
            }
            requested_depth_mode = str(self.depth_mode).upper()
            if requested_depth_mode not in depth_mode_map:
                raise RuntimeError(
                    f"Unknown ZED depth_mode={self.depth_mode!r}. "
                    f"Expected one of: {sorted(depth_mode_map)}"
                )
            init_params.depth_mode = depth_mode_map[requested_depth_mode]
            init_params.coordinate_units = self._zed_module.UNIT.METER
            init_params.coordinate_system = self._zed_module.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            if hasattr(init_params, "sensors_required"):
                init_params.sensors_required = True

            if self.serial_number is not None:
                sn = int(self.serial_number)
                rc = init_params.set_from_serial_number(sn)
                if rc is not None and rc != self._zed_module.ERROR_CODE.SUCCESS:
                    raise RuntimeError(
                        f"ZED set_from_serial_number({sn}) failed: {rc}"
                    )

            # Open camera
            status = self.camera.open(init_params)
            if status != self._zed_module.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open ZED camera: {status}")

            # Get camera information
            camera_info = self.camera.get_camera_information()
            logger.info(f"ZED camera initialized: {camera_info.camera_model}")
            logger.info(f"Resolution: {self.resolution}, FPS: {self.fps}")

            zed_model = getattr(self._zed_module, "MODEL", None)
            if zed_model is not None and camera_info.camera_model == zed_model.ZED:
                self._has_imu = False
            else:
                self._has_imu = True
                self._sensors_data = self._zed_module.SensorsData()
                logger.info("ZED onboard IMU enabled")

            # Create depth and RGB image mats
            self.depth_image = self._zed_module.Mat()
            self._rgb_image = self._zed_module.Mat()
            self._runtime_params = self._zed_module.RuntimeParameters()
            # Most permissive confidence filtering: keep low-confidence points instead
            # of dropping them, to maximize depth completeness.
            if hasattr(self._runtime_params, "confidence_threshold"):
                self._runtime_params.confidence_threshold = 100
            if hasattr(self._runtime_params, "texture_confidence_threshold"):
                self._runtime_params.texture_confidence_threshold = 100
            logger.info(
                "ZED runtime confidence thresholds set to 100 (most permissive)"
            )

            tracking_params = self._zed_module.PositionalTrackingParameters()
            track_status = self.camera.enable_positional_tracking(tracking_params)
            if track_status == self._zed_module.ERROR_CODE.SUCCESS:
                self._has_tracking = True
                self._tracking_pose = self._zed_module.Pose()
                logger.info("ZED positional tracking enabled")
            else:
                logger.warning(
                    "ZED positional tracking failed to enable: %s (base_lin_vel_b unavailable)",
                    track_status,
                )

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
                if self._has_tracking and hasattr(self.camera, "disable_positional_tracking"):
                    try:
                        self.camera.disable_positional_tracking()
                    except Exception as e:
                        logger.debug("ZED disable_positional_tracking: %s", e)
                self.camera.close()
                logger.info("ZED camera closed")
            except Exception as e:
                logger.error(f"Error closing ZED camera: {e}")
            finally:
                self.camera = None
                self.initialized = False
                self._has_tracking = False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def _grab_frame(self) -> bool:
        """Grab one frame and fill _last_rgb and _last_depth_np. Returns True on success.

        Expected failure (no frame available, USB glitch, etc.) is reported by the SDK
        via grab() return value; we return False and callers get None. We do not catch
        exceptions: any raise from the SDK or numpy is a bug or serious error and
        should propagate.
        """
        if not self.initialized:
            return False
        if self.camera.grab(self._runtime_params) != self._zed_module.ERROR_CODE.SUCCESS:
            return False
        # Retrieve depth
        self.camera.retrieve_measure(
            self.depth_image, self._zed_module.MEASURE.DEPTH, self._zed_module.MEM.CPU
        )
        depth_array = self.depth_image.get_data()
        self._last_depth_np = np.asarray(depth_array, dtype=np.float32).copy()
        # Retrieve left RGB
        self.camera.retrieve_image(
            self._rgb_image, self._zed_module.VIEW.LEFT, self._zed_module.MEM.CPU
        )
        rgb_data = self._rgb_image.get_data()
        # ZED returns BGRA (4 channels); convert to RGB uint8 (H, W, 3)
        rgb_np = np.asarray(rgb_data)
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[:, :, :3]  # drop alpha
        elif rgb_np.shape[-1] != 3:
            rgb_np = rgb_np[:, :, :3]
        self._last_rgb = np.ascontiguousarray(rgb_np.astype(np.uint8))
        self.last_frame_time_ns = time.time_ns()
        self._update_imu()
        self._update_tracking()
        return True

    def _update_imu(self) -> None:
        """Refresh IMU sample synchronized to the last ``grab()`` (IMAGE time reference)."""
        if not self._has_imu or self._sensors_data is None:
            self._last_imu = None
            return
        time_ref = getattr(self._zed_module, "TIME_REFERENCE", None)
        image_ref = getattr(time_ref, "IMAGE", None) if time_ref is not None else None
        if image_ref is None:
            return
        if (
            self.camera.get_sensors_data(self._sensors_data, image_ref)
            != self._zed_module.ERROR_CODE.SUCCESS
        ):
            return
        sample = parse_zed_imu_data(self._sensors_data.get_imu_data())
        if sample is not None:
            self._last_imu = sample

    def _update_tracking(self) -> None:
        """Refresh linear velocity from positional tracking after the last ``grab()``."""
        if not self._has_tracking or self._tracking_pose is None:
            return
        ref = tracking_reference_frame_camera(self._zed_module)
        state = self.camera.get_position(self._tracking_pose, ref)
        if not positional_tracking_state_ok(state, self._zed_module):
            return
        lin_vel = parse_zed_tracking_lin_vel_sensor(self._tracking_pose)
        if lin_vel is not None:
            self._last_tracking_lin_vel_sensor = lin_vel

    def get_camera_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture one frame and return (RGB, depth) for HAL observations.

        One grab retrieves both left RGB and depth. Use this when populating
        camera_rgb and camera_depth to avoid double capture.

        Returns:
            (rgb, depth): rgb (H, W, 3) uint8, depth (H, W) float32 in meters; either may be None on failure.
        """
        if not self._grab_frame():
            return None, None
        return self._last_rgb, self._last_depth_np

    def get_rgb_image(self) -> Optional[np.ndarray]:
        """Capture and return left RGB image (H, W, 3) uint8. Performs one grab."""
        if not self._grab_frame():
            return None
        return self._last_rgb

    def get_depth_map(self) -> Optional[np.ndarray]:
        """Return the last captured depth map (H, W) float32 in meters.

        No grab is performed; use get_camera_frames() or get_rgb_image() first to capture a frame.
        """
        return self._last_depth_np

    def is_ready(self) -> bool:
        """Check if camera is ready to capture frames.

        Returns:
            True if camera is initialized and ready
        """
        return self.initialized

    def has_imu(self) -> bool:
        """Return True when this device exposes a ZED onboard IMU."""
        return self._has_imu

    def get_imu_sample(self) -> Optional[ZedImuSample]:
        """Return the IMU sample from the last successful ``get_camera_frames()`` / grab."""
        return self._last_imu

    def has_tracking(self) -> bool:
        """Return True when ZED positional tracking is enabled."""
        return self._has_tracking

    def get_tracking_lin_vel_sensor(self) -> Optional[np.ndarray]:
        """Linear velocity (3,) m/s in the camera frame from the last successful grab."""
        return self._last_tracking_lin_vel_sensor

    def get_frame_rate(self) -> float:
        """Get actual frame rate.

        Returns:
            Frame rate in Hz
        """
        if self.last_frame_time_ns == 0:
            return 0.0

        # This is a simplified version - could track multiple frame times for better accuracy
        return self.fps  # Return configured FPS for now


def create_zed_camera(
    depth_mode: str,
    resolution: tuple[int, int],
    fps: int,
    serial_number: Optional[int] = None,
) -> Optional[ZedCamera]:
    """Factory function to create ZED camera with error handling.

    Args:
        resolution: Camera resolution (width, height)
        fps: Frames per second
        depth_mode: Depth mode
        serial_number: Optional USB serial to select a specific ZED when multiple are connected.

    Returns:
        ZedCamera instance if successful, None if initialization fails
    """
    try:
        camera = ZedCamera(
            depth_mode=depth_mode,
            resolution=resolution,
            fps=fps,
            serial_number=serial_number,
        )
        return camera
    except RuntimeError as e:
        logger.error(f"Failed to create ZED camera: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating ZED camera: {e}", exc_info=True)
        return None

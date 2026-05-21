"""Jetson HAL server implementation.

"""

import logging
import os
import time
from typing import Optional

import numpy as np

from hal.server import HalServerBase, HalServerConfig
from hal.server.robot_definition import RobotDefinition
from hal.client.data_structures.hardware import (
    HardwareObservations,
    JointCommand,
    RgbdCatalogObservation,
)
from hal.server.jetson.rgb_depth_camera import RgbDepthCamera
from hal.server.jetson.front_camera_factory import (
    FRONT_RGB_DEPTH_CAMERA_FACTORIES,
    create_front_rgb_depth_camera,
)
from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK
from hal.server.jetson.sensor_backend_jetson import (
    JETSON_SENSOR_CATALOG,
    JETSON_SENSOR_CATALOG_BY_ID,
    JetsonSensorCatalogEntry,
    JetsonSensorInterface,
    assert_hal_rgbd_catalog_config,
    front_observation_camera_catalog_entry,
)
from hal.server.sensor_interface import SensorInterface

from compute.parkour.model_definition import ObservationDimensions
from hal.server.jetson.depth_scan_features import (
    extract_depth_features_from_map,
    validate_depth_frame,
)

logger = logging.getLogger(__name__)


def _policy_scan_from_depth(
    depth_frame: np.ndarray, depth_feature_dim: int
) -> Optional[np.ndarray]:
    """Convert depth map (meters, H×W float32) to policy scan vector; None if unusable."""
    if depth_feature_dim <= 0:
        return None
    if not validate_depth_frame(depth_frame):
        return None
    try:
        feats = extract_depth_features_from_map(depth_frame, depth_feature_dim)
    except Exception as e:
        logger.warning("Depth scan feature extraction failed: %s", e)
        return None
    if len(feats) != depth_feature_dim:
        logger.error(
            "Depth scan length mismatch: %s != %s", len(feats), depth_feature_dim
        )
        return None
    return feats


def _read_optional_int_env(var_name: str) -> Optional[int]:
    raw = os.environ.get(var_name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s must be an integer; ignoring", var_name)
        return None


def _expected_rgb_depth_shapes_for_entry(
    entry: JetsonSensorCatalogEntry,
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Return ((H, W, 3), (H, W)) for RGB uint8 and depth float32."""
    rgb_w, rgb_h = entry.resolution
    depth_w, depth_h = entry.depth_resolution
    return (rgb_h, rgb_w, 3), (depth_h, depth_w)


class JetsonHalServer(HalServerBase):
    """HAL server for Jetson robot deployment.

    RGB-D devices are driven by ``JETSON_SENSOR_CATALOG``: the ``is_primary`` row always opens;
    additional ``rgbd`` rows open when ``hal_open_rgbd`` is True (ZED serial from
    ``zed_usb_serial_env``; MaixSense HTTP from ``maixsense_host_env`` / optional
    ``maixsense_port_env``). Legacy ``camera_*`` / ``side_*`` encode the **policy** scan slices; **metric depth
    for every opened stream** (including side / collision cameras) is in
    ``HardwareObservations.rgbd_by_catalog_id``.
    """

    def __init__(
        self,
        config: HalServerConfig,
        observation_dimensions: ObservationDimensions,
        action_dim: int,
        robot_definition: RobotDefinition,
        mcu_port: Optional[str] = None,
        mcu_baud: int = 115200,
        mcu_auto_connect: bool = True,
    ):
        """Initialize Jetson HAL server.

        Args:
            config: HAL server configuration
            observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition).
                Use a definition with ``num_side_scan > 0`` only with a matching checkpoint and
                a matching side catalog row (see ``JETSON_SENSOR_CATALOG`` ``policy_scan_slot``).
            action_dim: Action dimension (model output joint count)
            robot_definition: Robot definition for joint names and order (adapter uses get_joint_names()).
            mcu_port: Serial port for MCU connection. If None, uses default from MCU SDK.
            mcu_baud: Baud rate for MCU serial communication.
            mcu_auto_connect: If True, automatically connect to MCU on initialization.
        """
        super().__init__(config)
        self.observation_dimensions = observation_dimensions
        self.action_dim = action_dim
        self.robot_definition = robot_definition
        _obs_cam = front_observation_camera_catalog_entry()
        self.camera_resolution = _obs_cam.resolution
        self.camera_fps = _obs_cam.fps
        _driver = _obs_cam.camera_driver
        if not _driver:
            raise RuntimeError(
                "camera_driver is unset; set it on the JETSON_SENSOR_CATALOG is_primary row "
                "before constructing JetsonHalServer"
            )
        self._camera_driver = _driver
        self.front_camera: Optional[RgbDepthCamera] = None
        self.state_source = None  # IMU/encoders (placeholder, real implementation in future)
        self.actuator_sink = None  # Motors (placeholder, real implementation in future)
        self._last_joint_positions: Optional[dict[str, float]] = None  # from command.to_positions_dict()
        self._hal_rgbd_cameras: dict[str, RgbDepthCamera] = {}
        self._primary_catalog_id: str = front_observation_camera_catalog_entry().id
        self._side_catalog_id: Optional[str] = None
        self._primary_chunk_missing_warned: bool = False
        self._rgbd_no_frame_logged: dict[str, bool] = {}
        for row in JETSON_SENSOR_CATALOG:
            if row.policy_scan_slot == "side":
                self._side_catalog_id = row.id
                break
        self.side_camera: Optional[RgbDepthCamera] = None

        # GStreamer multi-sensor interface (optional)
        self._sensor_interface: Optional[SensorInterface] = None

        # Initialize Krabby MCU SDK when robot definition has MCU joints
        self._mcusdk: Optional[KrabbyMCUSDK] = None
        mcu_joints = self.robot_definition.get_mcu_joints()
        if mcu_joints:
            try:
                self._mcusdk = KrabbyMCUSDK(
                    mcu_joints=mcu_joints,
                    port=mcu_port,
                    baud=mcu_baud,
                    auto_connect=mcu_auto_connect,
                )
                logger.info("KrabbyMCUSDK initialized for JetsonHalServer")
            except (ImportError, RuntimeError, ValueError) as e:
                logger.warning(
                    f"KrabbyMCUSDK not available: {e}. "
                    "MCU commands will not be sent. Install firmware package and use a robot definition with mcu_joints."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize KrabbyMCUSDK: {e}. "
                    "MCU commands will not be sent. Check MCU connection and configuration.",
                    exc_info=True,
                )

    def get_sensor_interface(self) -> SensorInterface:
        """Return the GStreamer multi-sensor interface (list_sensors, get_gstreamer_handle, build_pipeline)."""
        if self._sensor_interface is None:
            self._sensor_interface = JetsonSensorInterface()
        return self._sensor_interface

    def initialize_cameras(self) -> None:
        """Initialize all HAL ``rgbd`` cameras selected by ``JETSON_SENSOR_CATALOG``.

        Opens every catalog row that qualifies (primary always attempted; others per
        ``hal_open_rgbd``). Rows that fail to open are skipped with warnings. The primary
        front camera is also skippable: if it does not open, the server continues with
        no RGB-D devices until hardware is available (observations omit primary vision).
        """
        assert_hal_rgbd_catalog_config()

        for cam in self._hal_rgbd_cameras.values():
            cam.close()
        self._hal_rgbd_cameras.clear()
        self._rgbd_no_frame_logged.clear()

        logger.info(
            "Initializing HAL RGB-D cameras from catalog (primary driver=%s)...",
            self._camera_driver,
        )

        for entry in JETSON_SENSOR_CATALOG:
            if entry.type != "rgbd" or not entry.camera_driver:
                continue
            if not entry.is_primary and not entry.hal_open_rgbd:
                continue
            if entry.camera_driver not in FRONT_RGB_DEPTH_CAMERA_FACTORIES:
                logger.warning(
                    "Skipping HAL RGB-D catalog %s: driver %r not in FRONT_RGB_DEPTH_CAMERA_FACTORIES",
                    entry.id,
                    entry.camera_driver,
                )
                continue

            zed_serial: Optional[int] = None
            if entry.camera_driver == "zed":
                zed_env_name = (entry.zed_usb_serial_env or "").strip()
                if zed_env_name:
                    zed_serial = _read_optional_int_env(zed_env_name)

            res = entry.resolution
            fps = entry.fps

            cam = create_front_rgb_depth_camera(
                entry.camera_driver,
                resolution=res,
                fps=fps,
                depth_mode=entry.depth_mode,
                zed_serial_number=zed_serial,
                maixsense_host=entry.maixsense_host,
                maixsense_port=entry.maixsense_port,
                maixsense_host_env=entry.maixsense_host_env,
                maixsense_port_env=entry.maixsense_port_env,
            )
            if cam is None or not cam.is_ready():
                if cam is not None:
                    cam.close()
                if entry.is_primary:
                    logger.warning(
                        "Primary HAL RGB-D %s failed to initialize or is not ready; "
                        "continuing without front camera (observations omit primary vision until fixed)",
                        entry.id,
                    )
                else:
                    logger.warning("HAL RGB-D %s failed to initialize; skipping", entry.id)
                continue

            self._hal_rgbd_cameras[entry.id] = cam
            logger.info(
                "HAL RGB-D catalog %s ready (driver=%s, %dx%d @ %d Hz)",
                entry.id,
                entry.camera_driver,
                res[0],
                res[1],
                fps,
            )

        self.front_camera = self._hal_rgbd_cameras.get(self._primary_catalog_id)
        if self.front_camera is None:
            logger.warning(
                "Primary catalog camera %r not opened (device missing, driver error, or catalog). "
                "HAL continues without front RGB-D; policy vision and teleop will be degraded until fixed.",
                self._primary_catalog_id,
            )

        self.side_camera = (
            self._hal_rgbd_cameras.get(self._side_catalog_id)
            if self._side_catalog_id
            else None
        )

        if self.observation_dimensions.num_side_scan > 0 and self.side_camera is None:
            logger.warning(
                "num_side_scan=%d but side HAL RGB-D not available "
                "(catalog hal_open_rgbd, driver init, or ZED USB serial env for side row)",
                self.observation_dimensions.num_side_scan,
            )

    def initialize_sensors(self) -> None:
        """Initialize state sensors (IMU/encoders).

        **WARNING**: This is currently a placeholder implementation.
        Real sensor data (IMU, encoders) is required for production deployment.
        Placeholder data (zeros) will cause incorrect policy behavior and is unsafe.

        TODO: Implement real sensor initialization:
        - Initialize IMU driver
        - Initialize encoder drivers  
        - Configure sensor parameters
        - Set self.state_source to real sensor interface
        """
        # Placeholder - actual implementation needs sensor drivers
        logger.warning(
            "⚠️  PLACEHOLDER MODE: Sensors not initialized. "
            "Using placeholder data (zeros) for base pose, velocities, and joint velocities. "
            "This will cause INCORRECT POLICY BEHAVIOR and is UNSAFE for production. "
            "Implement real sensor drivers before deployment."
        )

    def initialize_actuators(self) -> None:
        """Initialize actuators (motors).

        This is a placeholder. Real implementation would:
        - Initialize motor controllers
        - Configure motor parameters
        - Enable motors
        """
        # Placeholder - actual implementation needs motor drivers
        logger.info("Actuator initialization (placeholder)")

    def _build_depth_features(self) -> Optional[np.ndarray]:
        """Build front policy scan vector from one RGB-D capture (same depth as observations)."""
        if self.front_camera is None:
            return None
        _rgb, depth_frame = self.front_camera.get_camera_frames()
        if depth_frame is None:
            return None
        nf = self.observation_dimensions.num_scan_front
        return _policy_scan_from_depth(np.asarray(depth_frame, dtype=np.float32), nf)

    def _build_state_vector(self) -> Optional[np.ndarray]:
        """Build state vector from sensors.

        **WARNING**: Currently uses placeholder data for base pose, velocities, and joint velocities.
        Real sensor data is required for production deployment.

        Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
                joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
        
        Total length: 13 + 2*ACTION_DIM (13 base state + joint positions + joint velocities)

        Returns:
            State vector as float32 array

        TODO: When real sensors are implemented, extract:
        - Base pose from IMU/odometry (if state_source is available)
        - Base velocities from IMU
        - Joint positions from encoders
        - Joint velocities from encoder derivatives
        """
        # If we have real state source, use it
        if self.state_source is not None:
            # TODO: Implement real sensor data extraction from IMU/encoders
            # Real implementation would:
            # - Get base pose from IMU/odometry
            # - Get base velocities from IMU
            # - Get joint positions from encoders
            # - Get joint velocities from encoder derivatives
            # For now, fall through to placeholder
            logger.warning(
                "state_source is set but real sensor data extraction not implemented. "
                "Using placeholder data."
            )

        # Placeholder implementation: identity pose + echo joint positions
        # ⚠️ WARNING: This uses zeros for base pose, velocities, and joint velocities.
        # This will cause INCORRECT POLICY BEHAVIOR. Real sensors are required.
        
        # Base position: (0, 0, 0) - identity position (PLACEHOLDER - requires IMU/odometry)
        base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Base quaternion: (0, 0, 0, 1) - identity quaternion (PLACEHOLDER - requires IMU)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Base velocities: (0, 0, 0) - zero velocities (PLACEHOLDER - requires IMU)
        base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Joint positions: echo last commanded targets (best available data)
        # This is the only "real" data we have (from commands we sent)
        if self._last_joint_positions is not None:
            names = self.robot_definition.get_joint_names()
            joint_pos = np.array(
                [self._last_joint_positions.get(n, 0.0) for n in names],
                dtype=np.float32,
            )
        else:
            joint_pos = np.zeros(self.action_dim, dtype=np.float32)

        # Joint velocities: (0, 0, ...) - zero velocities (PLACEHOLDER - requires encoders)
        joint_vel = np.zeros(self.action_dim, dtype=np.float32)

        # Concatenate into state vector
        state_vector = np.concatenate([
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            joint_pos,
            joint_vel,
        ]).astype(np.float32)

        return state_vector

    def set_observation(self) -> None:
        """Set observation from real sensors as hardware observations.
        
        Constructs HardwareObservations from raw sensor data.
        Extracts:
        - Joint positions from state vector (echoed from last command)
        - Depth maps from front camera (if available)
        - RGB images from front camera (if available)
        
        Base class will loop until observation is published or throw if client isn't consuming.
        
        For each opened HAL RGB-D catalog camera, frames are grabbed every tick. If the driver
        returns no RGB or no depth (``None``), that stream uses **zero** tensors at the catalog
        resolution and zero scan features when applicable—**camera temporarily unavailable**.
        If ``get_camera_frames()`` raises, or returned arrays do not match the catalog resolution,
        a **RuntimeError** is raised (fail fast). Primary **camera_rgb** / **camera_depth** are
        omitted only when the primary camera was never opened. WARNING: Zero placeholders are
        unsafe for real policy behavior until the camera recovers.
        """
        # Build state vector (required for publishing observations)
        state_vector = self._build_state_vector()
        if state_vector is None:
            raise RuntimeError(
                "Failed to build state vector. "
                "State vector is required for publishing observations to the inference client."
            )

        # Extract components from state vector
        # Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
        #         joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
        # Total: 3 + 4 + 3 + 3 = 13 before joint positions
        base_pos = state_vector[0:3]
        base_quat = state_vector[3:7]  # (x, y, z, w) format
        base_lin_vel = state_vector[7:10]
        base_ang_vel = state_vector[10:13]
        joint_pos = state_vector[13:13+self.action_dim]
        joint_vel = state_vector[13+self.action_dim:13+2*self.action_dim]
        
        # Observation joint count from robot definition (e.g. 12 quad, 18 hex)
        obs_joint_count = self.robot_definition.get_total_joint_count()
        joint_positions = np.zeros(obs_joint_count, dtype=np.float32)
        num_joints = min(len(joint_pos), obs_joint_count)
        joint_positions[:num_joints] = joint_pos[:num_joints].astype(np.float32)
        
        joint_velocities = np.zeros(obs_joint_count, dtype=np.float32)
        num_joints_vel = min(len(joint_vel), obs_joint_count)
        joint_velocities[:num_joints_vel] = joint_vel[:num_joints_vel].astype(np.float32)
        
        # Extract base velocities (body frame) - for now use world frame values as placeholder
        # TODO: Transform to body frame when real IMU data is available
        base_ang_vel_b = base_ang_vel.astype(np.float32)
        base_lin_vel_b = base_lin_vel.astype(np.float32)
        
        # Base quaternion (world frame, x,y,z,w format)
        base_quat_w = base_quat.astype(np.float32)
        
        # Contact forces (placeholder - 5 values, normalized to [-0.5, 0.5])
        contact_forces = np.zeros(5, dtype=np.float32)
        
        # Previous action (from last command or zeros if none)
        previous_action = np.zeros(obs_joint_count, dtype=np.float32)
        if self._last_joint_positions is not None:
            names = self.robot_definition.get_joint_names()
            num_prev = min(len(names), obs_joint_count)
            for i in range(num_prev):
                previous_action[i] = self._last_joint_positions.get(names[i], 0.0)

        # Catalog RGB-D: one grab per opened HAL camera. If the driver returns no frame
        # (rgb or depth None), use zero tensors at the catalog resolution. Shape mismatches
        # and other errors from get_camera_frames propagate (not converted to zeros).
        primary_entry = JETSON_SENSOR_CATALOG_BY_ID[self._primary_catalog_id]
        camera_width, camera_height = primary_entry.resolution
        primary_depth_width, primary_depth_height = primary_entry.depth_resolution
        expected_rgb_shape = (camera_height, camera_width, 3)
        expected_depth_shape = (primary_depth_height, primary_depth_width)
        nf_scan = self.observation_dimensions.num_scan_front
        ns_scan = self.observation_dimensions.num_side_scan

        rgbd_by_catalog_id: dict[str, RgbdCatalogObservation] = {}
        for cid, cam in self._hal_rgbd_cameras.items():
            entry = JETSON_SENSOR_CATALOG_BY_ID.get(cid)
            if entry is None:
                logger.error(
                    "HAL RGB-D camera id %r not in JETSON_SENSOR_CATALOG_BY_ID; skipping",
                    cid,
                )
                continue

            exp_rgb_shape, exp_depth_shape = _expected_rgb_depth_shapes_for_entry(entry)
            rgb_frame, depth_frame = cam.get_camera_frames()

            placeholder = False
            if rgb_frame is None or depth_frame is None:
                if not self._rgbd_no_frame_logged.get(cid, False):
                    logger.error(
                        "HAL RGB-D catalog %s: no frame from camera (rgb_ok=%s depth_ok=%s); "
                        "using zero tensors shaped rgb=%s depth=%s",
                        cid,
                        rgb_frame is not None,
                        depth_frame is not None,
                        exp_rgb_shape,
                        exp_depth_shape,
                    )
                    self._rgbd_no_frame_logged[cid] = True
                rgb_u8 = np.zeros(exp_rgb_shape, dtype=np.uint8)
                depth_f = np.zeros(exp_depth_shape, dtype=np.float32)
                placeholder = True
            else:
                self._rgbd_no_frame_logged[cid] = False
                rgb_u8 = np.asarray(rgb_frame, dtype=np.uint8)
                depth_f = np.asarray(depth_frame, dtype=np.float32)
                if (
                    rgb_u8.shape != exp_rgb_shape
                    or depth_f.shape != exp_depth_shape
                ):
                    raise RuntimeError(
                        f"HAL RGB-D catalog {cid!r}: frame shape mismatch "
                        f"rgb={rgb_u8.shape} depth={depth_f.shape} "
                        f"(expected rgb {exp_rgb_shape}, depth {exp_depth_shape})"
                    )

            scan_c: Optional[np.ndarray] = None
            if not placeholder:
                if entry.is_primary and nf_scan > 0:
                    scan_c = _policy_scan_from_depth(depth_f, nf_scan)
                elif entry.policy_scan_slot == "side" and ns_scan > 0:
                    scan_c = _policy_scan_from_depth(depth_f, ns_scan)
            else:
                if entry.is_primary and nf_scan > 0:
                    scan_c = np.zeros(nf_scan, dtype=np.float32)
                elif entry.policy_scan_slot == "side" and ns_scan > 0:
                    scan_c = np.zeros(ns_scan, dtype=np.float32)

            rgbd_by_catalog_id[cid] = RgbdCatalogObservation(
                rgb=rgb_u8,
                depth=depth_f,
                scan_features=(
                    np.asarray(scan_c, dtype=np.float32)
                    if scan_c is not None
                    else None
                ),
            )

        camera_rgb: Optional[np.ndarray] = None
        camera_depth: Optional[np.ndarray] = None
        scan_features: Optional[np.ndarray] = None
        chunk_p = rgbd_by_catalog_id.get(self._primary_catalog_id)
        if chunk_p is None:
            if not self._primary_chunk_missing_warned:
                logger.warning(
                    "Primary catalog %s: no HAL RGB-D chunk (camera not opened); "
                    "camera_rgb / camera_depth omitted",
                    self._primary_catalog_id,
                )
                self._primary_chunk_missing_warned = True
        else:
            self._primary_chunk_missing_warned = False
            if chunk_p.rgb.shape != expected_rgb_shape:
                raise RuntimeError(
                    f"Primary catalog {self._primary_catalog_id!r}: rgb shape "
                    f"{chunk_p.rgb.shape} != expected {expected_rgb_shape}"
                )
            if chunk_p.depth.shape != expected_depth_shape:
                raise RuntimeError(
                    f"Primary catalog {self._primary_catalog_id!r}: depth shape "
                    f"{chunk_p.depth.shape} != expected {expected_depth_shape}"
                )
            camera_rgb = chunk_p.rgb
            camera_depth = chunk_p.depth
            scan_features = chunk_p.scan_features
            if scan_features is not None:
                scan_features = np.asarray(scan_features, dtype=np.float32)

        side_scan_features: Optional[np.ndarray] = None
        if self._side_catalog_id:
            chunk_s = rgbd_by_catalog_id.get(self._side_catalog_id)
            if chunk_s is not None:
                side_scan_features = chunk_s.scan_features
                if side_scan_features is not None:
                    side_scan_features = np.asarray(
                        side_scan_features, dtype=np.float32
                    )

        hw_obs = HardwareObservations(
            joint_positions=joint_positions,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=time.time_ns(),
            base_ang_vel_b=base_ang_vel_b,
            base_lin_vel_b=base_lin_vel_b,
            base_quat_w=base_quat_w,
            joint_velocities=joint_velocities,
            contact_forces=contact_forces,
            previous_action=previous_action,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth,
            scan_features=scan_features,
            side_scan_features=side_scan_features,
            side_camera_rgb=None,
            side_camera_depth=None,
            rgbd_by_catalog_id=rgbd_by_catalog_id if rgbd_by_catalog_id else None,
        )

        # Publish hardware observation via base-class publisher
        super().set_observation(hw_obs)

    def apply_command(self, command: JointCommand) -> None:
        """Apply joint command to actuators.
        
        **Synchronous method** that applies commands **immediately** (no queuing).
        Applies the provided command directly to the robot actuators via MCU SDK.
        Does not perform any background work to keep the robot moving - the main loop
        must call this method regularly at the target control rate (typically 100 Hz).
        
        The robot continues moving based on the last applied command until the next
        command is received. If this method is not called regularly, the robot will
        stop moving after the last command's effect completes.
        
        Args:
            command: JointCommand instance to apply
        
        Raises:
            RuntimeError: If command is invalid (empty or None)
        """
        if command is None:
            raise RuntimeError("Command cannot be None")

        # Use command's dict view for validation and state echo
        cmd_dict = command.to_positions_dict()
        if len(cmd_dict) == 0:
            raise RuntimeError("Received empty joint command")

        joint_names = self.robot_definition.get_joint_names()
        if len(cmd_dict) != len(joint_names):
            hint = ""
            if len(cmd_dict) == 18 and len(joint_names) == 12:
                hint = (
                    " Portal/teleop sends Krabby hex (18 joints); this server was built with a 12-joint "
                    'definition (--robot go2). Use --robot hex for crab hex teleop, or switch teleop mapper.'
                )
            elif len(cmd_dict) == 12 and len(joint_names) == 18:
                hint = (
                    " Received 12-DOF commands but HAL expects Krabby hex (18). "
                    "Use --robot go2 if your command source targets Unitree topology, "
                    "or fix the upstream command dimensions."
                )
            raise RuntimeError(
                f"Joint command length {len(cmd_dict)} does not match robot definition joint count "
                f"{len(joint_names)}.{hint}"
            )

        # Store command for state echo (echo joint state from last commanded targets).
        # This allows set_observation() to echo back the commanded positions as current state.
        # TODO: In _build_state_vector()/set_observation(), read joint positions from encoders
        # instead of _last_joint_positions; echoing command means observed state can lead actual
        # pose and policy never sees tracking error.
        self._last_joint_positions = cmd_dict

        # Apply command via MCU SDK if available (SDK uses command.to_positions_dict())
        if self._mcusdk is not None:
            try:
                self._mcusdk.apply_command(command)
            except Exception as e:
                logger.error(
                    f"Error applying command via MCU SDK: {e}. "
                    "Command logged but not sent to MCU.",
                    exc_info=True
                )
                raise e
        else:
            # MCU SDK not available - log command for debugging
            logger.debug(
                f"[JOINT COMMAND] MCU SDK not available. Command not sent. "
                f"timestamp={command.timestamp_ns}, joint_pos={list(cmd_dict.values())}"
            )

    def close(self) -> None:
        """Close camera, MCU connection, and all server resources."""
        # Close MCU SDK first
        if self._mcusdk is not None:
            self._mcusdk.close()
            self._mcusdk = None
        
        for cam in self._hal_rgbd_cameras.values():
            cam.close()
        self._hal_rgbd_cameras.clear()
        self.side_camera = None
        self.front_camera = None

        # Close server resources (sockets, context)
        super().close()


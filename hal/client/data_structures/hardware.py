"""Hardware data structures for Krabby robot.

These structures represent raw hardware sensor data and desired joint positions.
They are designed for zero-copy operations where possible.

Wire format (single-blob, no multipart):
  Observations and commands are serialized as one contiguous byte buffer per ZMQ
  message (one part). Reasons:
  - ZMQ CONFLATE works only with single-part messages; with one part we can set
    CONFLATE on the SUB socket and get "latest only" semantics without draining.
  - One atomic recv: no risk of EAGAIN partway through a multipart message.
  - Simpler HWM behavior and fewer syscalls.
  Blob layout: 4-byte metadata length (uint32 LE) + metadata JSON + array bytes
  in a fixed order (sizes derived from metadata). Optional fields use JSON ``null`` and omit
  the corresponding payload segment. Joint commands may include ``source`` (``inference`` /
  ``operator``); omitted ``source`` is treated as ``inference`` for backward compatibility.
"""
import json
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class JointCommandSource(str, Enum):
    """Origin of a joint command; HAL prefers ``OPERATOR`` over ``INFERENCE`` when both are queued."""

    INFERENCE = "inference"
    OPERATOR = "operator"


@dataclass
class RgbdCatalogObservation:
    """One RGB-D capture keyed by ``JETSON_SENSOR_CATALOG`` row ``id`` (Jetson HAL).

    ``depth`` is always the metric depth map for that sensor (meters). Use it for collision /
    proximity and other perception that is **not** the primary locomotion policy scan.
    ``scan_features`` is only filled when HAL maps this row into the policy scan slots (primary
    or ``policy_scan_slot="side"``); do not assume every row has scan features.
    """

    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, meters
    scan_features: Optional[np.ndarray] = None  # 1D float32 when HAL computed a policy scan slice


@dataclass
class HardwareObservations:
    """Hardware observation data.
    
    Contains all raw sensor data from the hardware:
    - Joint positions (robot-dependent DOF)
    - Camera resolution metadata (camera_height, camera_width)
    - **Policy / locomotion (primary):** ``camera_rgb``, ``camera_depth``, ``scan_features`` —
      tied to the catalog primary rgbd row; depth here feeds the trained scan slice, not collision.
    - **Policy second scan (optional):** legacy ``side_*`` when one catalog row has
      ``policy_scan_slot="side"``. ``side_camera_rgb`` / ``side_camera_depth`` use that
      stream's own H×W (may differ from ``camera_height`` × ``camera_width``).
    - **All HAL rgbd streams (including side / extra cameras):** ``rgbd_by_catalog_id`` carries
      full-resolution **RGB + depth** per catalog ``id``. Prefer ``rgbd_by_catalog_id[id].depth``
      for **collision detection** and other geometry; resolutions may differ from
      ``camera_height`` × ``camera_width``.
    
    Front camera format (camera_rgb / camera_depth):
    - Resolution: same as camera_height x camera_width (self-describing).
    - Encoding: camera_rgb = uint8 (H, W, 3) BGR or RGB; camera_depth = float32 (H, W) in meters.
    - Timestamp: timestamp_ns applies to the whole observation including images.
    
    Zero-copy guarantees:
    - Arrays are stored as numpy arrays (may be views or copies depending on source)
    - Scalar values (timestamp, camera dimensions) are copied
    """
    
    joint_positions: np.ndarray  # Shape: (n,) n = robot joint count, dtype: float32
    camera_height: int  # Height of camera images (front camera when camera_rgb/camera_depth present)
    camera_width: int  # Width of camera images
    timestamp_ns: int
    
    # Robot state fields (required - HAL server must provide this data)
    base_ang_vel_b: np.ndarray  # Shape: (3,), dtype: float32 - Base angular velocity (body frame)
    base_lin_vel_b: np.ndarray  # Shape: (3,), dtype: float32 - Base linear velocity (body frame)
    base_quat_w: np.ndarray  # Shape: (4,), dtype: float32 - Base quaternion (world frame, x,y,z,w)
    joint_velocities: np.ndarray  # Shape: (n,) n = robot joint count, dtype: float32 - Joint velocities
    contact_forces: np.ndarray  # Shape: (5,), dtype: float32 - Contact forces (5 values from environment, normalized to [-0.5, 0.5])
    previous_action: np.ndarray  # Shape: (n,) n = robot joint count, dtype: float32 - Previous joint command
    
    # Environment-specific fields (for matching environment observation manager)
    delta_yaw: float = 0.0  # Target yaw - current yaw (from parkour_event)
    delta_next_yaw: float = 0.0  # Next target yaw - current yaw (from parkour_event)
    terrain_type_flag: float = 1.0  # 1 if not flat, 0 if flat (from environment)
    flat_terrain_flag: float = 0.0  # 1 if flat, 0 if not flat (from environment)
    scan_features: Optional[np.ndarray] = None  # Shape: (132,), dtype: float32 - Scan features from observation manager (measured_heights)
    privileged_latent: Optional[np.ndarray] = None  # Shape: (29,), dtype: float32 - Privileged latent features (available in simulation, None on hardware)
    # Front camera (single): ZED 2i RGB + depth or Isaac synthetic equivalent
    camera_rgb: Optional[np.ndarray] = None   # Shape: (camera_height, camera_width, 3), dtype: uint8
    camera_depth: Optional[np.ndarray] = None  # Shape: (camera_height, camera_width), dtype: float32, meters
    side_scan_features: Optional[np.ndarray] = None  # Depth-derived features for side camera (1D float32)
    side_camera_rgb: Optional[np.ndarray] = None  # (Hs, Ws, 3) uint8; Hs/Ws independent of front
    side_camera_depth: Optional[np.ndarray] = None  # (Hs, Ws) float32 meters; matches side rgb H×W
    rgbd_by_catalog_id: Optional[dict[str, RgbdCatalogObservation]] = None

    def __post_init__(self) -> None:
        """Validate hardware observations."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError(f"Camera dimensions must be positive, got {self.camera_height}x{self.camera_width}")
        
        # Validate joint positions (1D, length 1–24 per robot)
        if self.joint_positions.ndim != 1 or self.joint_positions.size < 1 or self.joint_positions.size > 24:
            raise ValueError(
                f"joint_positions must be 1D with size in [1, 24], got shape {self.joint_positions.shape}"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Validate required robot state fields
        if self.base_ang_vel_b.shape != (3,):
            raise ValueError(f"base_ang_vel_b shape {self.base_ang_vel_b.shape} != (3,)")
        if self.base_lin_vel_b.shape != (3,):
            raise ValueError(f"base_lin_vel_b shape {self.base_lin_vel_b.shape} != (3,)")
        if self.base_quat_w.shape != (4,):
            raise ValueError(f"base_quat_w shape {self.base_quat_w.shape} != (4,)")
        if self.joint_velocities.ndim != 1 or self.joint_velocities.size != self.joint_positions.size:
            raise ValueError(
                f"joint_velocities must be 1D and same length as joint_positions ({self.joint_positions.size}), got shape {self.joint_velocities.shape}"
            )
        if self.contact_forces.shape != (5,):
            raise ValueError(f"contact_forces shape {self.contact_forces.shape} != (5,)")
        if self.previous_action.ndim != 1 or self.previous_action.size != self.joint_positions.size:
            raise ValueError(
                f"previous_action must be 1D and same length as joint_positions ({self.joint_positions.size}), got shape {self.previous_action.shape}"
            )
        
        # Validate optional front camera fields (must be both set or both unset; shapes must match metadata)
        expected_shape_2d = (self.camera_height, self.camera_width)
        expected_shape_3d = (self.camera_height, self.camera_width, 3)
        if self.camera_rgb is not None or self.camera_depth is not None:
            if self.camera_rgb is None or self.camera_depth is None:
                raise ValueError("camera_rgb and camera_depth must be both set or both None")
            if self.camera_rgb.shape != expected_shape_3d:
                raise ValueError(
                    f"camera_rgb must be {expected_shape_3d}, got {self.camera_rgb.shape}"
                )
            if self.camera_rgb.dtype != np.uint8:
                self.camera_rgb = self.camera_rgb.astype(np.uint8)
            if self.camera_depth.shape != expected_shape_2d:
                raise ValueError(
                    f"camera_depth must be {expected_shape_2d}, got {self.camera_depth.shape}"
                )
            if self.camera_depth.dtype != np.float32:
                self.camera_depth = self.camera_depth.astype(np.float32)

        if self.side_camera_rgb is not None or self.side_camera_depth is not None:
            if self.side_camera_rgb is None or self.side_camera_depth is None:
                raise ValueError("side_camera_rgb and side_camera_depth must be both set or both None")
            if self.side_camera_rgb.ndim != 3 or self.side_camera_rgb.shape[2] != 3:
                raise ValueError(
                    f"side_camera_rgb must be (H, W, 3), got {self.side_camera_rgb.shape}"
                )
            if self.side_camera_rgb.dtype != np.uint8:
                self.side_camera_rgb = self.side_camera_rgb.astype(np.uint8)
            if self.side_camera_depth.ndim != 2:
                raise ValueError(
                    f"side_camera_depth must be 2D, got {self.side_camera_depth.shape}"
                )
            if self.side_camera_depth.dtype != np.float32:
                self.side_camera_depth = self.side_camera_depth.astype(np.float32)
            if self.side_camera_rgb.shape[:2] != self.side_camera_depth.shape:
                raise ValueError(
                    f"side_camera_rgb H×W {self.side_camera_rgb.shape[:2]} != "
                    f"side_camera_depth {self.side_camera_depth.shape}"
                )

        if self.side_scan_features is not None:
            if self.side_scan_features.ndim != 1:
                raise ValueError(f"side_scan_features must be 1D, got shape {self.side_scan_features.shape}")
            self.side_scan_features = (
                self.side_scan_features.astype(np.float32)
                if self.side_scan_features.dtype != np.float32
                else self.side_scan_features
            )

        if self.rgbd_by_catalog_id is not None:
            for cid, obs in self.rgbd_by_catalog_id.items():
                if obs.rgb.ndim != 3 or obs.rgb.shape[2] != 3:
                    raise ValueError(
                        f"rgbd_by_catalog_id[{cid!r}].rgb must be (H,W,3), got {obs.rgb.shape}"
                    )
                if obs.rgb.dtype != np.uint8:
                    obs.rgb = obs.rgb.astype(np.uint8)
                if obs.depth.ndim != 2:
                    raise ValueError(
                        f"rgbd_by_catalog_id[{cid!r}].depth must be 2D, got {obs.depth.shape}"
                    )
                if obs.depth.dtype != np.float32:
                    obs.depth = obs.depth.astype(np.float32)
                # rgbd_by_catalog_id may intentionally carry mixed-resolution RGB/depth
                # streams from certain camera drivers (for example MaixSense). Keep
                # validation to rank/dtype and let per-stream consumers handle sizing.
                if obs.scan_features is not None:
                    if obs.scan_features.ndim != 1:
                        raise ValueError(
                            f"rgbd_by_catalog_id[{cid!r}].scan_features must be 1D"
                        )
                    if obs.scan_features.dtype != np.float32:
                        obs.scan_features = obs.scan_features.astype(np.float32)

        # Ensure all are float32
        if self.base_ang_vel_b.dtype != np.float32:
            self.base_ang_vel_b = self.base_ang_vel_b.astype(np.float32)
        if self.base_lin_vel_b.dtype != np.float32:
            self.base_lin_vel_b = self.base_lin_vel_b.astype(np.float32)
        if self.base_quat_w.dtype != np.float32:
            self.base_quat_w = self.base_quat_w.astype(np.float32)
        if self.joint_velocities.dtype != np.float32:
            self.joint_velocities = self.joint_velocities.astype(np.float32)
        if self.contact_forces.dtype != np.float32:
            self.contact_forces = self.contact_forces.astype(np.float32)
        if self.previous_action.dtype != np.float32:
            self.previous_action = self.previous_action.astype(np.float32)
    
    def to_bytes(self) -> bytes:
        """Serialize to a single byte buffer for ZMQ transport (one message part).
        
        Layout: 4B metadata_len (uint32 LE) + metadata JSON + array bytes in fixed order.
        Order: joint_positions, base_ang_vel_b, base_lin_vel_b, base_quat_w,
        joint_velocities, contact_forces, previous_action; then payload segments for
        scan_features, privileged_latent, camera_rgb, camera_depth, then optional
        ``side_scan_features``, ``side_camera_rgb``, ``side_camera_depth`` — only when non-null;
        then optional ``rgbd_by_catalog_id`` (ordered catalog rgbd chunks).

        Metadata always includes ``scan_features``, ``privileged_latent``, ``camera_rgb``,
        ``camera_depth``, ``side_scan_features``, ``side_camera_rgb``, ``side_camera_depth``,
        ``rgbd_by_catalog_id``
        (each ``null`` or a shape/dtype object). Older blobs may omit side / rgbd keys; decoders treat
        missing keys as null.
        Single-blob allows ZMQ CONFLATE to work (latest-only) and one atomic recv.
        """
        joint_pos = np.ascontiguousarray(self.joint_positions, dtype=np.float32)
        base_ang_vel_b = np.ascontiguousarray(self.base_ang_vel_b, dtype=np.float32)
        base_lin_vel_b = np.ascontiguousarray(self.base_lin_vel_b, dtype=np.float32)
        base_quat_w = np.ascontiguousarray(self.base_quat_w, dtype=np.float32)
        joint_vel = np.ascontiguousarray(self.joint_velocities, dtype=np.float32)
        contact_forces = np.ascontiguousarray(self.contact_forces, dtype=np.float32)
        prev_action = np.ascontiguousarray(self.previous_action, dtype=np.float32)

        metadata: dict = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "base_ang_vel_b": {"shape": list(base_ang_vel_b.shape), "dtype": str(base_ang_vel_b.dtype)},
            "base_lin_vel_b": {"shape": list(base_lin_vel_b.shape), "dtype": str(base_lin_vel_b.dtype)},
            "base_quat_w": {"shape": list(base_quat_w.shape), "dtype": str(base_quat_w.dtype)},
            "joint_velocities": {"shape": list(joint_vel.shape), "dtype": str(joint_vel.dtype)},
            "contact_forces": {"shape": list(contact_forces.shape), "dtype": str(contact_forces.dtype)},
            "previous_action": {"shape": list(prev_action.shape), "dtype": str(prev_action.dtype)},
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "timestamp_ns": self.timestamp_ns,
            "delta_yaw": self.delta_yaw,
            "delta_next_yaw": self.delta_next_yaw,
            "terrain_type_flag": self.terrain_type_flag,
            "flat_terrain_flag": self.flat_terrain_flag,
        }
        if self.scan_features is not None:
            scan_features = np.ascontiguousarray(self.scan_features, dtype=np.float32)
            metadata["scan_features"] = {"shape": list(scan_features.shape), "dtype": str(scan_features.dtype)}
        else:
            metadata["scan_features"] = None
        if self.privileged_latent is not None:
            priv_latent = np.ascontiguousarray(self.privileged_latent, dtype=np.float32)
            metadata["privileged_latent"] = {"shape": list(priv_latent.shape), "dtype": str(priv_latent.dtype)}
        else:
            metadata["privileged_latent"] = None
        if self.camera_rgb is not None and self.camera_depth is not None:
            camera_rgb = np.ascontiguousarray(self.camera_rgb, dtype=np.uint8)
            camera_depth = np.ascontiguousarray(self.camera_depth, dtype=np.float32)
            metadata["camera_rgb"] = {"shape": list(camera_rgb.shape), "dtype": str(camera_rgb.dtype)}
            metadata["camera_depth"] = {"shape": list(camera_depth.shape), "dtype": str(camera_depth.dtype)}
        else:
            metadata["camera_rgb"] = None
            metadata["camera_depth"] = None
        if self.side_scan_features is not None:
            ssf = np.ascontiguousarray(self.side_scan_features, dtype=np.float32)
            metadata["side_scan_features"] = {"shape": list(ssf.shape), "dtype": str(ssf.dtype)}
        else:
            metadata["side_scan_features"] = None
        if self.side_camera_rgb is not None and self.side_camera_depth is not None:
            srgb = np.ascontiguousarray(self.side_camera_rgb, dtype=np.uint8)
            sdep = np.ascontiguousarray(self.side_camera_depth, dtype=np.float32)
            metadata["side_camera_rgb"] = {"shape": list(srgb.shape), "dtype": str(srgb.dtype)}
            metadata["side_camera_depth"] = {"shape": list(sdep.shape), "dtype": str(sdep.dtype)}
        else:
            metadata["side_camera_rgb"] = None
            metadata["side_camera_depth"] = None

        if self.rgbd_by_catalog_id:
            order = sorted(self.rgbd_by_catalog_id.keys())
            entries: dict = {}
            for cid in order:
                o = self.rgbd_by_catalog_id[cid]
                entries[cid] = {
                    "rgb": {"shape": list(o.rgb.shape), "dtype": str(o.rgb.dtype)},
                    "depth": {"shape": list(o.depth.shape), "dtype": str(o.depth.dtype)},
                    "scan_features": (
                        {
                            "shape": list(o.scan_features.shape),
                            "dtype": str(o.scan_features.dtype),
                        }
                        if o.scan_features is not None
                        else None
                    ),
                }
            metadata["rgbd_by_catalog_id"] = {"order": order, "entries": entries}
        else:
            metadata["rgbd_by_catalog_id"] = None

        metadata_json = json.dumps(metadata).encode("utf-8")
        metadata_len = len(metadata_json)
        buf = bytearray()
        buf += struct.pack("<I", metadata_len)
        buf += metadata_json
        buf += joint_pos.tobytes()
        buf += base_ang_vel_b.tobytes()
        buf += base_lin_vel_b.tobytes()
        buf += base_quat_w.tobytes()
        buf += joint_vel.tobytes()
        buf += contact_forces.tobytes()
        buf += prev_action.tobytes()
        if self.scan_features is not None:
            buf += np.ascontiguousarray(self.scan_features, dtype=np.float32).tobytes()
        if self.privileged_latent is not None:
            buf += np.ascontiguousarray(self.privileged_latent, dtype=np.float32).tobytes()
        if self.camera_rgb is not None and self.camera_depth is not None:
            buf += np.ascontiguousarray(self.camera_rgb, dtype=np.uint8).tobytes()
            buf += np.ascontiguousarray(self.camera_depth, dtype=np.float32).tobytes()
        if self.side_scan_features is not None:
            buf += np.ascontiguousarray(self.side_scan_features, dtype=np.float32).tobytes()
        if self.side_camera_rgb is not None and self.side_camera_depth is not None:
            buf += np.ascontiguousarray(self.side_camera_rgb, dtype=np.uint8).tobytes()
            buf += np.ascontiguousarray(self.side_camera_depth, dtype=np.float32).tobytes()
        if self.rgbd_by_catalog_id:
            for cid in sorted(self.rgbd_by_catalog_id.keys()):
                o = self.rgbd_by_catalog_id[cid]
                buf += np.ascontiguousarray(o.rgb, dtype=np.uint8).tobytes()
                buf += np.ascontiguousarray(o.depth, dtype=np.float32).tobytes()
                if o.scan_features is not None:
                    buf += np.ascontiguousarray(o.scan_features, dtype=np.float32).tobytes()
        return bytes(buf)

    @classmethod
    def from_bytes(cls, data: bytes) -> "HardwareObservations":
        """Deserialize from single-blob wire format.
        
        Layout: 4B metadata_len (uint32 LE) + metadata JSON + array bytes in fixed order.
        See to_bytes() for order. Single-blob avoids multipart and enables CONFLATE.
        
        Args:
            data: One byte buffer (one ZMQ message part).
        
        Returns:
            HardwareObservations instance.
        
        Raises:
            ValueError: If buffer is too short, invalid JSON, or invalid metadata/arrays.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"from_bytes: data must be bytes or bytearray, got {type(data)}")
        if len(data) < 4:
            raise ValueError(f"from_bytes: need at least 4 bytes (metadata length), got {len(data)}")
        metadata_len = struct.unpack("<I", data[0:4])[0]
        offset = 4
        if offset + metadata_len > len(data):
            raise ValueError(
                f"from_bytes: metadata length {metadata_len} extends past buffer (offset={offset}, len={len(data)})"
            )
        metadata = json.loads(data[offset : offset + metadata_len].decode("utf-8"))
        offset += metadata_len
        if not isinstance(metadata, dict):
            raise ValueError(f"from_bytes: metadata must be dict, got {type(metadata)}")

        required_base = ("joint_positions", "base_ang_vel_b", "base_lin_vel_b", "base_quat_w", "joint_velocities", "contact_forces", "previous_action")
        for k in required_base:
            if k not in metadata:
                raise ValueError(f"from_bytes: metadata must include {k!r}")

        def read_array(name: str, current_offset: int) -> tuple[np.ndarray, int]:
            entry = metadata[name]
            shape = tuple(entry["shape"])
            dtype = np.dtype(entry["dtype"])
            nbytes = int(np.prod(shape)) * dtype.itemsize
            if current_offset + nbytes > len(data):
                raise ValueError(f"from_bytes: array {name!r} needs {nbytes} bytes at offset {current_offset}, buffer len {len(data)}")
            arr = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)), offset=current_offset).copy()
            arr = arr.reshape(shape)
            return arr, nbytes

        joint_pos, n = read_array("joint_positions", offset)
        offset += n
        base_ang_vel_b, n = read_array("base_ang_vel_b", offset)
        offset += n
        base_lin_vel_b, n = read_array("base_lin_vel_b", offset)
        offset += n
        base_quat_w, n = read_array("base_quat_w", offset)
        offset += n
        joint_vel, n = read_array("joint_velocities", offset)
        offset += n
        contact_forces, n = read_array("contact_forces", offset)
        offset += n
        prev_action, n = read_array("previous_action", offset)
        offset += n

        joint_pos = joint_pos.astype(np.float32)
        base_ang_vel_b = base_ang_vel_b.astype(np.float32)
        base_lin_vel_b = base_lin_vel_b.astype(np.float32)
        base_quat_w = base_quat_w.astype(np.float32)
        joint_vel = joint_vel.astype(np.float32)
        contact_forces = contact_forces.astype(np.float32)
        prev_action = prev_action.astype(np.float32)

        if joint_pos.ndim != 1 or joint_pos.size < 1 or joint_pos.size > 24:
            raise ValueError(f"from_bytes: joint_positions shape invalid: {joint_pos.shape}")
        if base_ang_vel_b.shape != (3,):
            raise ValueError(f"from_bytes: base_ang_vel_b shape {base_ang_vel_b.shape} != (3,)")
        if base_lin_vel_b.shape != (3,):
            raise ValueError(f"from_bytes: base_lin_vel_b shape {base_lin_vel_b.shape} != (3,)")
        if base_quat_w.shape != (4,):
            raise ValueError(f"from_bytes: base_quat_w shape {base_quat_w.shape} != (4,)")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"from_bytes: joint_velocities shape {joint_vel.shape} != joint_positions {joint_pos.shape}")
        if contact_forces.shape != (5,):
            raise ValueError(f"from_bytes: contact_forces shape {contact_forces.shape} != (5,)")
        if prev_action.shape != joint_pos.shape:
            raise ValueError(f"from_bytes: previous_action shape {prev_action.shape} != joint_positions {joint_pos.shape}")

        camera_height = metadata.get("camera_height")
        camera_width = metadata.get("camera_width")
        if camera_height is None or camera_width is None:
            raise ValueError("from_bytes: metadata must include camera_height and camera_width")
        camera_height = int(camera_height)
        camera_width = int(camera_width)
        if camera_height <= 0 or camera_width <= 0:
            raise ValueError(f"from_bytes: camera_height/width must be positive, got {camera_height}x{camera_width}")

        for req in (
            "scan_features",
            "privileged_latent",
            "camera_rgb",
            "camera_depth",
        ):
            if req not in metadata:
                raise ValueError(f"from_bytes: metadata must include key {req!r}")

        sf_meta = metadata["scan_features"]
        scan_features = None
        if sf_meta is not None:
            scan_features, n = read_array("scan_features", offset)
            offset += n
            scan_features = scan_features.astype(np.float32)

        pl_meta = metadata["privileged_latent"]
        privileged_latent = None
        if pl_meta is not None:
            privileged_latent, n = read_array("privileged_latent", offset)
            offset += n
            privileged_latent = privileged_latent.astype(np.float32)

        cr_meta = metadata["camera_rgb"]
        cd_meta = metadata["camera_depth"]
        camera_rgb, camera_depth = None, None
        if cr_meta is None and cd_meta is None:
            pass
        elif cr_meta is not None and cd_meta is not None:
            camera_rgb, n = read_array("camera_rgb", offset)
            offset += n
            camera_depth, n = read_array("camera_depth", offset)
            offset += n
            camera_depth = camera_depth.astype(np.float32)
            if camera_rgb.shape != (camera_height, camera_width, 3):
                raise ValueError(f"from_bytes: camera_rgb shape {camera_rgb.shape} != ({camera_height},{camera_width},3)")
            if camera_depth.shape != (camera_height, camera_width):
                raise ValueError(f"from_bytes: camera_depth shape {camera_depth.shape} != ({camera_height},{camera_width})")
        else:
            raise ValueError(
                "from_bytes: camera_rgb and camera_depth must both be null or both non-null in metadata"
            )

        side_scan_features = None
        ssf_meta = metadata.get("side_scan_features")
        if ssf_meta is not None:
            side_scan_features, n = read_array("side_scan_features", offset)
            offset += n
            side_scan_features = side_scan_features.astype(np.float32)

        scr_meta = metadata.get("side_camera_rgb")
        scd_meta = metadata.get("side_camera_depth")
        side_camera_rgb, side_camera_depth = None, None
        if scr_meta is None and scd_meta is None:
            pass
        elif scr_meta is not None and scd_meta is not None:
            side_camera_rgb, n = read_array("side_camera_rgb", offset)
            offset += n
            side_camera_depth, n = read_array("side_camera_depth", offset)
            offset += n
            side_camera_depth = side_camera_depth.astype(np.float32)
            if side_camera_rgb.ndim != 3 or side_camera_rgb.shape[2] != 3:
                raise ValueError(
                    f"from_bytes: side_camera_rgb must be (H,W,3), got {side_camera_rgb.shape}"
                )
            if side_camera_depth.ndim != 2:
                raise ValueError(
                    f"from_bytes: side_camera_depth must be 2D, got {side_camera_depth.shape}"
                )
            if side_camera_rgb.shape[:2] != side_camera_depth.shape:
                raise ValueError(
                    f"from_bytes: side rgb H×W {side_camera_rgb.shape[:2]} != depth {side_camera_depth.shape}"
                )
        else:
            raise ValueError(
                "from_bytes: side_camera_rgb and side_camera_depth must both be null or both non-null in metadata"
            )

        rgbd_by_catalog_id = None
        rpack = metadata.get("rgbd_by_catalog_id")
        if rpack is not None:
            order = rpack["order"]
            entries = rpack["entries"]
            out_rgbd: dict[str, RgbdCatalogObservation] = {}
            for cid in order:
                em = entries[cid]

                def _read_blob(entry: dict, cur: int) -> tuple[np.ndarray, int]:
                    shape_t = tuple(entry["shape"])
                    dt = np.dtype(entry["dtype"])
                    nb = int(np.prod(shape_t)) * dt.itemsize
                    if cur + nb > len(data):
                        raise ValueError(
                            f"from_bytes: rgbd {cid!r} needs {nb} bytes at {cur}, len {len(data)}"
                        )
                    arr = (
                        np.frombuffer(
                            data,
                            dtype=dt,
                            count=int(np.prod(shape_t)),
                            offset=cur,
                        )
                        .copy()
                        .reshape(shape_t)
                    )
                    return arr, nb

                rgb_a, n = _read_blob(em["rgb"], offset)
                offset += n
                dep_a, n = _read_blob(em["depth"], offset)
                offset += n
                dep_a = dep_a.astype(np.float32)
                rgb_a = rgb_a.astype(np.uint8)
                scan_part: Optional[np.ndarray] = None
                if em["scan_features"] is not None:
                    scan_part, n = _read_blob(em["scan_features"], offset)
                    offset += n
                    scan_part = scan_part.astype(np.float32).reshape(-1)
                out_rgbd[cid] = RgbdCatalogObservation(
                    rgb=rgb_a, depth=dep_a, scan_features=scan_part
                )
            rgbd_by_catalog_id = out_rgbd

        timestamp_ns = int(metadata.get("timestamp_ns", 0))
        if timestamp_ns < 0:
            raise ValueError(f"from_bytes: timestamp_ns must be non-negative, got {timestamp_ns}")

        return cls(
            joint_positions=joint_pos,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=timestamp_ns,
            base_ang_vel_b=base_ang_vel_b,
            base_lin_vel_b=base_lin_vel_b,
            base_quat_w=base_quat_w,
            joint_velocities=joint_vel,
            contact_forces=contact_forces,
            previous_action=prev_action,
            delta_yaw=float(metadata.get("delta_yaw", 0.0)),
            delta_next_yaw=float(metadata.get("delta_next_yaw", 0.0)),
            terrain_type_flag=float(metadata.get("terrain_type_flag", 1.0)),
            flat_terrain_flag=float(metadata.get("flat_terrain_flag", 0.0)),
            scan_features=scan_features,
            privileged_latent=privileged_latent,
            camera_rgb=camera_rgb,
            camera_depth=camera_depth,
            side_scan_features=side_scan_features,
            side_camera_rgb=side_camera_rgb,
            side_camera_depth=side_camera_depth,
            rgbd_by_catalog_id=rgbd_by_catalog_id,
        )


@dataclass
class JointCommand:
    """Joint command structure.
    
    Target joint positions for hardware control. Length is robot-defined:
    - Quad (4 legs × 3 DOF): 12 joints
    - Hex (6 legs × 3 DOF): 18 joints
    
    Zero-copy guarantees:
    - joint_positions array may be a view if source is compatible
    - timestamp is always copied (scalar)
    """
    
    _joint_positions: np.ndarray  # Shape: (n,) per robot definition, dtype: float32; use to_positions_dict() to read
    timestamp_ns: int  # Timestamp when command was created
    observation_timestamp_ns: int  # Timestamp of the observation this command responds to.
        # Used for tracking command-observation relationships and measuring round-trip latency.
    joint_names: tuple[str, ...]  # Names in same order as positions; keeps dict view in sync.
    source: JointCommandSource = JointCommandSource.INFERENCE

    def __post_init__(self) -> None:
        """Validate desired joint positions."""
        if not isinstance(self.source, JointCommandSource):
            raise ValueError(f"source must be JointCommandSource, got {type(self.source)}")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if len(self.joint_names) != self._joint_positions.size:
            raise ValueError(
                f"joint_names length {len(self.joint_names)} must match joint_positions size {self._joint_positions.size}"
            )
        if self._joint_positions.ndim != 1 or self._joint_positions.size == 0:
            raise ValueError(
                f"joint_positions must be 1D non-empty, got shape {self._joint_positions.shape}"
            )
        if self._joint_positions.size > 24:
            raise ValueError(
                f"joint_positions length {self._joint_positions.size} exceeds max 24"
            )
        if self._joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            object.__setattr__(self, "_joint_positions", self._joint_positions.astype(np.float32))
        
        # Runtime type validation: Validate values (check for NaN/Inf)
        if not np.isfinite(self._joint_positions).all():
            nan_count = np.isnan(self._joint_positions).sum()
            inf_count = np.isinf(self._joint_positions).sum()
            raise ValueError(f"Invalid joint_positions values: {nan_count} NaN, {inf_count} Inf")
    
    def to_positions_dict(self) -> dict[str, float]:
        """Return joint positions as a dict keyed by this command's joint names (canonical order)."""
        return dict(zip(self.joint_names, (float(x) for x in self._joint_positions)))
    
    def to_bytes(self) -> bytes:
        """Serialize to a single byte buffer for ZMQ transport (one message part).
        
        Layout: 4B metadata_len (uint32 LE) + metadata JSON + joint_positions bytes.
        Single-blob for consistency with observation format and one atomic send/recv.
        """
        joint_pos = np.ascontiguousarray(self._joint_positions, dtype=np.float32)
        metadata = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "timestamp_ns": self.timestamp_ns,
            "observation_timestamp_ns": self.observation_timestamp_ns,
            "joint_names": list(self.joint_names),
            "source": self.source.value,
        }
        metadata_json = json.dumps(metadata).encode("utf-8")
        return struct.pack("<I", len(metadata_json)) + metadata_json + joint_pos.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "JointCommand":
        """Deserialize from single-blob wire format.
        
        Layout: 4B metadata_len (uint32 LE) + metadata JSON + joint_positions bytes.
        """
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"from_bytes: data must be bytes or bytearray, got {type(data)}")
        if len(data) < 4:
            raise ValueError(f"from_bytes: need at least 4 bytes (metadata length), got {len(data)}")
        metadata_len = struct.unpack("<I", data[0:4])[0]
        offset = 4
        if offset + metadata_len > len(data):
            raise ValueError(
                f"from_bytes: metadata length {metadata_len} extends past buffer (len={len(data)})"
            )
        metadata = json.loads(data[offset : offset + metadata_len].decode("utf-8"))
        offset += metadata_len
        shape = tuple(metadata["joint_positions"]["shape"])
        dtype = np.dtype(metadata["joint_positions"]["dtype"])
        nbytes = int(np.prod(shape)) * dtype.itemsize
        if offset + nbytes > len(data):
            raise ValueError(f"from_bytes: joint_positions extends past buffer (need {nbytes} at offset {offset})")
        joint_pos = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)), offset=offset).copy()
        joint_pos = joint_pos.reshape(shape).astype(np.float32)
        if "joint_names" not in metadata:
            raise ValueError("wire format must include joint_names (list of strings in same order as joint_positions)")
        joint_names = tuple(metadata["joint_names"])
        if len(joint_names) != joint_pos.size:
            raise ValueError(
                f"metadata joint_names length {len(joint_names)} must match array size {joint_pos.size}"
            )
        raw_src = metadata.get("source", JointCommandSource.INFERENCE.value)
        try:
            source = JointCommandSource(str(raw_src))
        except ValueError:
            source = JointCommandSource.INFERENCE
        return cls(
            _joint_positions=joint_pos,
            timestamp_ns=int(metadata.get("timestamp_ns", 0)),
            observation_timestamp_ns=int(metadata.get("observation_timestamp_ns", 0)),
            joint_names=joint_names,
            source=source,
        )
    


"""Map `HardwareObservations` to ROS 2 CDR bytes via rosbags typestore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rosbags.typesys.store import Typestore

    from data_collection.config import TopicEnable
    from hal.client.data_structures.hardware import HardwareObservations

IMAGE_MSGTYPE = "sensor_msgs/msg/Image"


def _split_stamp(ns: int) -> tuple[int, int]:
    sec = int(ns // 1_000_000_000)
    nanosec = int(ns % 1_000_000_000)
    return sec, nanosec


def _header(ts: "Typestore", stamp_ns: int, frame_id: str):
    Header = ts.types["std_msgs/msg/Header"]
    Stamp = ts.types["builtin_interfaces/msg/Time"]
    sec, nanosec = _split_stamp(stamp_ns)
    return Header(stamp=Stamp(sec=sec, nanosec=nanosec), frame_id=frame_id)


def serialize_image_rgb8(
    ts: "Typestore", stamp_ns: int, frame_id: str, rgb: np.ndarray
) -> bytes:
    """``rgb`` (H, W, 3) uint8, row-major; ``encoding`` is ``rgb8``."""
    Img = ts.types["sensor_msgs/msg/Image"]
    h, w, _ = rgb.shape
    data = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    step = w * 3
    msg = Img(
        header=_header(ts, stamp_ns, frame_id),
        height=int(h),
        width=int(w),
        encoding="rgb8",
        is_bigendian=0,
        step=int(step),
        data=np.frombuffer(data, dtype=np.uint8),
    )
    return ts.serialize_cdr(msg, "sensor_msgs/msg/Image")


def serialize_image_mono8(
    ts: "Typestore", stamp_ns: int, frame_id: str, gray: np.ndarray
) -> bytes:
    """Single-channel uint8 (H, W)."""
    Img = ts.types["sensor_msgs/msg/Image"]
    h, w = gray.shape
    data = np.ascontiguousarray(gray, dtype=np.uint8).tobytes()
    msg = Img(
        header=_header(ts, stamp_ns, frame_id),
        height=int(h),
        width=int(w),
        encoding="mono8",
        is_bigendian=0,
        step=int(w),
        data=np.frombuffer(data, dtype=np.uint8),
    )
    return ts.serialize_cdr(msg, "sensor_msgs/msg/Image")


def serialize_image_depth_32fc1(
    ts: "Typestore", stamp_ns: int, frame_id: str, depth_m: np.ndarray
) -> bytes:
    """Metric depth (H, W) float32 meters, ``32FC1``."""
    Img = ts.types["sensor_msgs/msg/Image"]
    h, w = depth_m.shape
    arr = np.ascontiguousarray(depth_m, dtype=np.float32)
    data = arr.tobytes()
    step = w * 4
    msg = Img(
        header=_header(ts, stamp_ns, frame_id),
        height=int(h),
        width=int(w),
        encoding="32FC1",
        is_bigendian=0,
        step=int(step),
        data=np.frombuffer(data, dtype=np.uint8),
    )
    return ts.serialize_cdr(msg, "sensor_msgs/msg/Image")


def serialize_joint_state(
    ts: "Typestore",
    stamp_ns: int,
    frame_id: str,
    names: tuple[str, ...],
    position: np.ndarray,
    velocity: np.ndarray,
) -> bytes:
    JS = ts.types["sensor_msgs/msg/JointState"]
    n = int(position.size)
    if names and len(names) != n:
        names = tuple(f"joint_{i}" for i in range(n))
    elif not names:
        names = tuple(f"joint_{i}" for i in range(n))
    pos = position.astype(np.float64)
    vel = velocity.astype(np.float64)
    effort = np.zeros(n, dtype=np.float64)
    msg = JS(
        header=_header(ts, stamp_ns, frame_id),
        name=list(names),
        position=pos,
        velocity=vel,
        effort=effort,
    )
    return ts.serialize_cdr(msg, "sensor_msgs/msg/JointState")


def serialize_imu(ts: "Typestore", stamp_ns: int, frame_id: str, obs: "HardwareObservations") -> bytes:
    """Populate ``sensor_msgs/Imu`` from base state.

    - **Orientation:** ``base_quat_w`` (x, y, z, w), world frame.
    - **Angular velocity:** ``base_ang_vel_b`` (rad/s), base frame.
    - **Linear acceleration:** zeros (not estimated on this path); ``base_lin_vel_b`` is body-frame
      linear velocity and is **not** copied into ``Imu`` to avoid mislabeling velocity as acceleration.
    """
    Imu = ts.types["sensor_msgs/msg/Imu"]
    Q = ts.types["geometry_msgs/msg/Quaternion"]
    V = ts.types["geometry_msgs/msg/Vector3"]
    q = obs.base_quat_w
    ori = Q(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
    av = obs.base_ang_vel_b
    ang = V(x=float(av[0]), y=float(av[1]), z=float(av[2]))
    lin = V(x=0.0, y=0.0, z=0.0)
    cov = np.full(9, -1.0, dtype=np.float64)
    msg = Imu(
        header=_header(ts, stamp_ns, frame_id),
        orientation=ori,
        orientation_covariance=cov,
        angular_velocity=ang,
        angular_velocity_covariance=cov,
        linear_acceleration=lin,
        linear_acceleration_covariance=cov,
    )
    return ts.serialize_cdr(msg, "sensor_msgs/msg/Imu")


def catalog_camera_topic(catalog_id: str, stream: str) -> str:
    """ROS topic for one catalog RGB-D stream (``stream`` is ``rgb`` or ``depth``)."""
    return f"/camera/{catalog_id}/{stream}"


def is_catalog_camera_topic(topic: str) -> bool:
    return topic.startswith("/camera/") and topic.count("/") == 3


def observation_to_writes(
    ts: "Typestore",
    obs: "HardwareObservations",
    topics: "TopicEnable",
    joint_names: tuple[str, ...],
) -> list[tuple[str, str, bytes]]:
    """Return list of (topic_name, msg_type, cdr_bytes) for this observation."""
    from hal.client.data_structures.hardware import HardwareObservations

    if not isinstance(obs, HardwareObservations):
        raise TypeError(obs)
    out: list[tuple[str, str, bytes]] = []
    t = obs.timestamp_ns

    rgbd = obs.rgbd_by_catalog_id or {}
    for cid in sorted(rgbd.keys()):
        entry = rgbd[cid]
        frame = f"camera_{cid}"
        rgb_topic = catalog_camera_topic(cid, "rgb")
        if entry.rgb.ndim == 2:
            out.append(
                (
                    rgb_topic,
                    IMAGE_MSGTYPE,
                    serialize_image_mono8(ts, t, frame, entry.rgb),
                )
            )
        else:
            out.append(
                (
                    rgb_topic,
                    IMAGE_MSGTYPE,
                    serialize_image_rgb8(ts, t, frame, entry.rgb),
                )
            )
        out.append(
            (
                catalog_camera_topic(cid, "depth"),
                IMAGE_MSGTYPE,
                serialize_image_depth_32fc1(ts, t, frame, entry.depth),
            )
        )

    if topics.joints_state:
        out.append(
            (
                "/joints/state",
                "sensor_msgs/msg/JointState",
                serialize_joint_state(
                    ts,
                    t,
                    "base",
                    joint_names,
                    obs.joint_positions,
                    obs.joint_velocities,
                ),
            )
        )
    if topics.joints_command:
        out.append(
            (
                "/joints/command",
                "sensor_msgs/msg/JointState",
                serialize_joint_state(
                    ts,
                    t,
                    "base",
                    joint_names,
                    obs.previous_action,
                    np.zeros_like(obs.previous_action),
                ),
            )
        )
    if topics.imu:
        out.append(("/imu", "sensor_msgs/msg/Imu", serialize_imu(ts, t, "base_link", obs)))

    return out

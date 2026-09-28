"""Map `HardwareObservations` to ROS 2 CDR bytes via rosbags typestore."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rosbags.typesys.store import Typestore

    from data_collection.config import TopicEnable
    from hal.client.data_structures.hardware import HardwareObservations

IMAGE_MSGTYPE = "sensor_msgs/msg/Image"

JOINTS_STATE_TOPIC = "/joints/state"
JOINTS_COMMAND_TOPIC = "/joints/command"
IMU_TOPIC = "/imu"
JOINT_STATE_MSGTYPE = "sensor_msgs/msg/JointState"
IMU_MSGTYPE = "sensor_msgs/msg/Imu"

BASE_TWIST_TOPIC = "/base/twist"
BASE_TWIST_MSGTYPE = "geometry_msgs/msg/TwistStamped"


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
    JS = ts.types[JOINT_STATE_MSGTYPE]
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
    return ts.serialize_cdr(msg, JOINT_STATE_MSGTYPE)


def serialize_imu(ts: "Typestore", stamp_ns: int, frame_id: str, obs: "HardwareObservations") -> bytes:
    """Populate ``sensor_msgs/Imu`` from base state.

    - **Orientation:** ``base_quat_w`` (x, y, z, w), world frame.
    - **Angular velocity:** ``base_ang_vel_b`` (rad/s), base frame.
    - **Linear acceleration:** zeros (ZED IMU accel is not on ``HardwareObservations``).
    - **Linear velocity:** recorded on ``BASE_TWIST_TOPIC`` (``/base/twist``), not in ``Imu``.
    """
    Imu = ts.types[IMU_MSGTYPE]
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
    return ts.serialize_cdr(msg, IMU_MSGTYPE)


def serialize_base_twist_stamped(
    ts: "Typestore", stamp_ns: int, frame_id: str, obs: "HardwareObservations"
) -> bytes:
    """Body-frame twist: linear ``base_lin_vel_b``, angular ``base_ang_vel_b`` (m/s, rad/s)."""
    TwistStamped = ts.types[BASE_TWIST_MSGTYPE]
    Twist = ts.types["geometry_msgs/msg/Twist"]
    V = ts.types["geometry_msgs/msg/Vector3"]
    lv = obs.base_lin_vel_b
    av = obs.base_ang_vel_b
    twist = Twist(
        linear=V(x=float(lv[0]), y=float(lv[1]), z=float(lv[2])),
        angular=V(x=float(av[0]), y=float(av[1]), z=float(av[2])),
    )
    msg = TwistStamped(header=_header(ts, stamp_ns, frame_id), twist=twist)
    return ts.serialize_cdr(msg, BASE_TWIST_MSGTYPE)


def catalog_camera_topic(catalog_id: str, stream: str) -> str:
    """ROS topic for one catalog RGB-D stream (``stream`` is ``rgb`` or ``depth``)."""
    return f"/camera/{catalog_id}/{stream}"


def catalog_camera_topic_msgtypes(
    catalog_ids: tuple[str, ...] | list[str],
) -> list[tuple[str, str]]:
    """(topic, msgtype) pairs for pre-registering catalog cameras in rosbag2."""
    return [
        (catalog_camera_topic(cid, stream), IMAGE_MSGTYPE)
        for cid in catalog_ids
        for stream in ("rgb", "depth")
    ]


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
        rgb_topic = catalog_camera_topic(cid, "rgb")
        depth_topic = catalog_camera_topic(cid, "depth")
        # Use the ROS topic as frame_id so MCAP/Foxglove stream names match playback topics
        # (``camera_{id}`` caused viewers to infer ``/camera/front/rgb`` from ``front_rgbd``).
        if entry.rgb.ndim == 2:
            rgb_bytes = serialize_image_mono8(ts, t, rgb_topic, entry.rgb)
        else:
            rgb_bytes = serialize_image_rgb8(ts, t, rgb_topic, entry.rgb)
        out.append((rgb_topic, IMAGE_MSGTYPE, rgb_bytes))

        depth_bytes = serialize_image_depth_32fc1(ts, t, depth_topic, entry.depth)
        out.append((depth_topic, IMAGE_MSGTYPE, depth_bytes))

    if topics.joints_state:
        out.append(
            (
                JOINTS_STATE_TOPIC,
                JOINT_STATE_MSGTYPE,
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
                JOINTS_COMMAND_TOPIC,
                JOINT_STATE_MSGTYPE,
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
        out.append((IMU_TOPIC, IMU_MSGTYPE, serialize_imu(ts, t, "base_link", obs)))
    if topics.base_twist:
        out.append(
            (
                BASE_TWIST_TOPIC,
                BASE_TWIST_MSGTYPE,
                serialize_base_twist_stamped(ts, t, "base_link", obs),
            )
        )

    return out

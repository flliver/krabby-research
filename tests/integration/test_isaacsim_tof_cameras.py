"""Isaac Sim ToF depth camera integration checks.

Docker (portable; see ``images/isaacsim/README.md``)::

    PYTHONUNBUFFERED=1 timeout 600 docker run --rm --gpus all \\
      --entrypoint /workspace/run_test_runner.sh \\
      krabby-isaacsim:latest test_isaacsim_tof_cameras

Local Isaac Lab (conda env from ``./isaaclab.sh --conda``, usually
``env_isaaclab``) — full steps in ``scripts/isaac/run_tof_integ_local.py``::

    python scripts/isaac/run_tof_integ_local.py

Pytest collection is marked ``isaacsim`` so default desktop pytest skips it.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pytest

pytestmark = pytest.mark.isaacsim

TOF_CATALOG_IDS = ("belly_left_tof", "belly_right_tof", "rear_tof")
GO2_BELLY_RIGHT_POS = (-0.05, -0.12, -0.05)


def assert_tof_depth_frames(
    rgbd_by_catalog_id: dict,
    *,
    expected_shape: tuple[int, int] = (100, 100),
    catalog_ids: Iterable[str] = TOF_CATALOG_IDS,
    require_finite: bool = True,
) -> None:
    """Assert each ToF catalog id delivers a float32 depth channel at SEN0581 resolution."""
    assert rgbd_by_catalog_id is not None
    for cid in catalog_ids:
        assert cid in rgbd_by_catalog_id, f"missing ToF catalog id {cid!r}"
        entry = rgbd_by_catalog_id[cid]
        depth = entry.channels.get("depth")
        assert depth is not None, f"{cid}: expected depth channel, got {sorted(entry.channels)}"
        assert depth.shape == expected_shape, f"{cid}: shape {depth.shape} != {expected_shape}"
        assert depth.dtype == np.float32, f"{cid}: dtype {depth.dtype} != float32"
        assert "rgb" not in entry.channels, f"{cid}: depth-only row must not carry rgb"
        if require_finite:
            finite = np.isfinite(depth) & (depth > 0.0)
            assert finite.any(), f"{cid}: expected some positive finite depth"


def assert_box_distance_in_depth(
    depth: np.ndarray,
    *,
    expected_m: float,
    tol_m: float = 0.15,
    valid_min_m: float = 0.15,
    valid_max_m: float = 1.5,
    min_fraction: float = 0.02,
) -> None:
    """Check that a cluster of in-range depths is near ``expected_m`` (box visibility).

    Parkour ground often fills most of the FOV, so the global median is not a reliable
    box range. Require a minimum fraction of in-range pixels near ``expected_m``.
    """
    valid = depth[np.isfinite(depth) & (depth >= valid_min_m) & (depth <= valid_max_m)]
    assert valid.size > 0, "no in-range depth pixels for box distance check"
    near = valid[(valid >= expected_m - tol_m) & (valid <= expected_m + tol_m)]
    frac = float(near.size) / float(valid.size)
    assert frac >= min_fraction, (
        f"only {frac:.1%} of in-range pixels within {tol_m} m of {expected_m:.3f} m "
        f"(need >={min_fraction:.0%}); median={float(np.median(valid)):.3f} m"
    )


def belly_right_optical_forward_body() -> np.ndarray:
    """Unit +Z camera axis of ``belly_right_tof`` expressed in the robot body frame."""
    from hal.server.isaac.sim_rgbd_camera_cfgs import (
        _belly_tof_rot,
        _rotmat_camera_from_quat_ros_wxyz,
    )

    q = _belly_tof_rot(outward_yaw_deg=-90.0)
    r_cam = _rotmat_camera_from_quat_ros_wxyz(q)
    forward = r_cam @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward / np.linalg.norm(forward)


def place_tof_test_box_on_optical_axis(
    env,
    *,
    distance_m: float = 0.35,
    box_name: str = "tof_test_box",
    mount_body: tuple[float, float, float] = GO2_BELLY_RIGHT_POS,
) -> np.ndarray:
    """Move ``box_name`` to ``distance_m`` along belly-right optical axis; return world XYZ."""
    import torch

    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    root_quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    # root_quat is wxyz
    w, x, y, z = root_quat
    r_body = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    cam_body = np.asarray(mount_body, dtype=np.float64)
    cam_w = root_pos + r_body @ cam_body
    forward_w = r_body @ belly_right_optical_forward_body()
    target = cam_w + float(distance_m) * forward_w

    box = env.scene[box_name]
    pose = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    pose[:, 0] = float(target[0])
    pose[:, 1] = float(target[1])
    pose[:, 2] = float(target[2])
    pose[:, 3] = 1.0  # identity quat wxyz
    box.write_root_pose_to_sim(pose)
    box.reset()
    return target


def test_tof_assert_helpers_unit() -> None:
    """Pure helper checks (no Isaac) so desktop pytest still exercises this module lightly."""
    from types import SimpleNamespace

    depth = np.full((100, 100), 0.4, dtype=np.float32)
    obs = {
        "belly_left_tof": SimpleNamespace(channels={"depth": depth}),
        "belly_right_tof": SimpleNamespace(channels={"depth": depth.copy()}),
        "rear_tof": SimpleNamespace(channels={"depth": depth.copy()}),
    }
    assert_tof_depth_frames(obs)
    assert_box_distance_in_depth(depth, expected_m=0.4, tol_m=0.05, min_fraction=0.5)


def run_tof_hal_client_check(
    *,
    env,
    robot_definition,
    observation_bind: str = "inproc://tof_obs",
    command_bind: str = "inproc://tof_cmd",
    settle_steps: int = 8,
    box_expected_m: float | None = None,
    box_catalog_id: str = "belly_right_tof",
) -> dict:
    """Initialize Isaac HAL, poll one observation, assert ToF frames. Returns rgbd map."""
    from hal.client.client import HalClient
    from hal.client.config import HalClientConfig
    from hal.server import HalServerConfig
    from hal.server.isaac import IsaacSimHalServer
    from hal.server.zmq_endpoints import zmq_bind_to_connect_url

    from compute.parkour.model_definition import PARKOUR_MODEL_OBSERVATION_DEFINITION

    observation_dimensions = PARKOUR_MODEL_OBSERVATION_DEFINITION.get_observation_dimensions(
        robot_definition
    )
    server = IsaacSimHalServer(
        HalServerConfig(observation_bind=observation_bind, command_bind=command_bind),
        robot_definition,
        env=env,
        observation_dimensions=observation_dimensions,
    )
    server.initialize()
    client = HalClient(
        HalClientConfig(
            observation_endpoint=zmq_bind_to_connect_url(observation_bind),
            command_endpoint=zmq_bind_to_connect_url(command_bind),
        ),
        context=server.get_transport_context(),
    )
    client.initialize()
    time.sleep(0.05)
    try:
        import torch

        # HAL rejects duplicate/all-zero policy obs unless the env advances between publishes.
        n_act = int(env.action_space.shape[-1])
        zero_action = torch.zeros((env.num_envs, n_act), device=env.device)
        for _ in range(settle_steps):
            obs_dict, *_ = env.step(zero_action)
            server._latest_obs_dict = obs_dict
            server._latest_obs_tensor = obs_dict["policy"]
            server.set_observation()
            time.sleep(0.02)
        obs = None
        for _ in range(50):
            obs_dict, *_ = env.step(zero_action)
            server._latest_obs_dict = obs_dict
            server._latest_obs_tensor = obs_dict["policy"]
            server.set_observation()
            obs = client.poll(timeout_ms=100)
            if obs is not None and obs.rgbd_by_catalog_id:
                break
            time.sleep(0.02)
        assert obs is not None, "HAL client received no observation"
        assert_tof_depth_frames(obs.rgbd_by_catalog_id)
        if box_expected_m is not None:
            chunk = obs.rgbd_by_catalog_id[box_catalog_id]
            assert_box_distance_in_depth(
                chunk.channels["depth"], expected_m=box_expected_m
            )
        return dict(obs.rgbd_by_catalog_id)
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            server.close()
        except Exception:
            pass


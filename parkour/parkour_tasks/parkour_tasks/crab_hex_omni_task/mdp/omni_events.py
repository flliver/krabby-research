# SPDX-License-Identifier: BSD-3-Clause
"""Ring spawn event for the omni task (plan D9, spec B6).

The corridor reset (``parkour_isaaclab.envs.mdp.events.reset_root_state``) places every
robot on its tile's platform, ``size_y + offset`` metres BEHIND the tile centre, facing the
course (``+x``), with the default orientation. The ring has no course: the robot starts in
the middle of the clear disc and may be commanded in any world direction, so the spawn is
the TILE CENTRE and the spawn YAW is uniform over ``yaw_range``.

The random spawn yaw is what makes "world travel direction" and "world facing target" mean
anything during training: with a fixed spawn yaw the body frame and the world frame would
be identical at every episode start, and a policy could satisfy an omni command by learning
the body-frame command alone. It is also why the omni scene keeps ``init_state.rot``
identity -- the yaw is applied HERE, so the config dump shows one source of truth.

``offset`` keeps its corridor NAME but changes meaning: it is the maximum RADIUS of a
uniform-disc jitter around the tile centre (default 0.0, i.e. spawn exactly at the centre).
The name is load-bearing: ``ParkourEvent.__init__`` reads
``event_manager.get_term_cfg('reset_root_state').params['offset']`` unconditionally, so the
omni term cfg must still carry a parameter with that name.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .omni_exposure import SPAWN_PLATFORM, note_spawn_xy_to_parkour

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "reset_root_state_ring",
    "note_spawn_xy_to_parkour",
    "sample_uniform_disc",
    "quat_apply",
    "quat_from_yaw",
    "quat_mul",
    "yaw_from_quat",
]

# ---------------------------------------------------------------------------------------
# Quaternion helpers shared by the two omni reset events (this one and the omni RSI event).
# All of them are transcriptions of the ``isaaclab.utils.math`` functions of the same name,
# with the same (w, x, y, z) convention and the same expression trees. They are transcribed
# rather than imported so both reset events stay importable -- and therefore unit-testable
# -- without Isaac Sim: a test that stubbed ``quat_apply`` would be asserting against its
# own stub, which is exactly the class of silent error these events are prone to.
# ---------------------------------------------------------------------------------------


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(w, x, y, z)`` quaternions, shape ``(N, 4)``.

    Transcribed from ``isaaclab.utils.math.quat_mul`` (same expression tree, so the same
    floating-point result) rather than imported: this module's siblings are unit-tested
    without Isaac, and a stubbed ``quat_mul`` in a test would be testing the stub.
    """
    if q1.shape != q2.shape:
        raise ValueError(f"quaternion shape mismatch: {tuple(q1.shape)} != {tuple(q2.shape)}")
    shape = q1.shape
    a = q1.reshape(-1, 4)
    b = q2.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` ``(w, x, y, z)`` rotation of ``yaw`` radians about world ``+z``."""
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` ``(N, 3)`` by ``quat`` ``(N, 4)`` ``(w, x, y, z)`` -- body -> world.

    Transcribed from ``isaaclab.utils.math.quat_apply``.
    """
    shape = vec.shape
    q = quat.reshape(-1, 4)
    v = vec.reshape(-1, 3)
    xyz = q[:, 1:]
    t = xyz.cross(v, dim=-1) * 2
    return (v + q[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """World yaw of a ``(w, x, y, z)`` quaternion, radians in ``[-pi, pi]``.

    The angle ``isaaclab.utils.math.yaw_quat`` extracts before rebuilding a yaw-only
    quaternion: ``atan2(2(wz + xy), 1 - 2(y^2 + z^2))``.
    """
    q = quat.reshape(-1, 4)
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)).view(quat.shape[:-1])


def sample_uniform_disc(n: int, radius: float, device) -> torch.Tensor:
    """``(n, 2)`` uniform sample on the disc of ``radius`` (area-uniform, ``r = R*sqrt(u)``).

    ``radius <= 0`` returns exact zeros, so the default spawn is bit-exactly the tile
    centre and consumes no random numbers -- an unarmed knob must not move the RNG stream.
    """
    if not (radius > 0.0):
        return torch.zeros(n, 2, device=device)
    u = torch.rand(n, device=device)
    ang = torch.rand(n, device=device) * (2.0 * math.pi)
    r = float(radius) * torch.sqrt(u)
    return torch.stack((r * torch.cos(ang), r * torch.sin(ang)), dim=-1)


def reset_root_state_ring(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.0,
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Place resetting envs at their tile centre with a random world yaw.

    Args:
        env: the environment.
        env_ids: envs being reset.
        asset_cfg: the articulation to place (default ``robot``).
        offset: max radius of the uniform-disc spawn jitter, metres (0 = exact centre).
            NAME is load-bearing, see the module docstring.
        yaw_range: ``(lo, hi)`` uniform band for the spawn yaw, radians. ``(a, a)`` pins the
            yaw (the eval harness does this to make a world-frame hold reproducible).

    The default root state supplies the spawn height and the nominal orientation; the yaw
    rotation is applied in the WORLD frame (``q = q_z(psi) * q_default``), so the robot's
    roll/pitch relative to the ground are unchanged and its body ``+x`` ends up pointing at
    world angle ``psi`` exactly (the default orientation is a pure yaw, identity for this
    plant). Linear and angular velocities are written as exact zeros.
    """
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    origin = env.scene.env_origins[env_ids].clone()
    origin[:, -1] = 0.0

    n = int(root_states.shape[0])
    positions = root_states[:, 0:3] + origin
    positions[:, :2] += sample_uniform_disc(n, float(offset), env.device)

    lo, hi = float(yaw_range[0]), float(yaw_range[1])
    if not lo <= hi:
        raise ValueError(f"yaw_range must satisfy lo <= hi, got ({lo}, {hi})")
    psi = torch.rand(n, device=env.device) * (hi - lo) + lo
    orientations = quat_mul(quat_from_yaw(psi), root_states[:, 3:7])

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros_like(root_states[:, 7:13]), env_ids=env_ids)
    note_spawn_xy_to_parkour(env, env_ids, positions[:, :2], SPAWN_PLATFORM)

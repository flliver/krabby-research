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

The module also holds the PAYLOAD reset event (M18 Task 4 pipeline step 3, "M1a", 2026-10-16):
:class:`payload_reset`, armed only by ``config/crab_hex/payload_knobs.py`` (default off; see the
payload section at the end of this file).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .omni_exposure import SPAWN_PLATFORM, note_spawn_xy_to_parkour

# NOTE(M1a payload, 2026-10-16): guarded, because two CPU suites (``test_crab_hex_omni_rsi.py``,
# ``test_rsi_cam_stage.py``) path-load this module under a stub ``isaaclab.managers`` that defines
# only ``SceneEntityCfg``. In every Kit process the real class is importable (the line above just
# imported its module); the stand-in can never reach a live env silently, because IsaacLab's
# manager refuses a class term that is not a real ``ManagerTermBase`` subclass (TypeError).
try:
    from isaaclab.managers import ManagerTermBase
except ImportError:  # pragma: no cover - CPU stub only

    class ManagerTermBase:  # noqa: D101 - interface of isaaclab.managers.ManagerTermBase
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

        @property
        def num_envs(self) -> int:
            return self._env.num_envs

        @property
        def device(self):
            return self._env.device

        def reset(self, env_ids=None) -> None:
            pass

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
    # payload reset event (M1a, 2026-10-16)
    "PAYLOAD_SEED_SALT",
    "payload_seed",
    "box_inertia_diag",
    "composite_mass_properties",
    "draw_payload",
    "check_payload_params",
    "payload_reset",
    "payload_term",
    "payload_run_meta",
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


# =======================================================================================
# Payload reset event (M18 Task 4 pipeline step 3, "M1a", 2026-10-16; D5)
# =======================================================================================
#
# A solid box load rigidly added to the robot's ``body`` link, re-drawn at every reset. Armed
# only by ``config/crab_hex/payload_knobs.py`` (``KRABBY_PAYLOAD_*``, default off: unset, the
# term does not exist in the cfg). The design rules, each one a silent failure if broken:
#
# * ABSOLUTE, never cumulative. The stock IsaacLab / parkour CoM events ADD to whatever the
#   PhysX view holds (``parkour_isaaclab/envs/mdp/events.py`` ``randomize_rigid_body_com``),
#   so calling one at every reset walks the CoM away and, on a partial reset, also re-adds to
#   envs that were not reset. Here every write is computed from each env's CAPTURED base, so
#   the 10th reset of an env lands where the 1st did and the other envs are never touched.
# * LAZY capture. The base (mass, CoM pose, inertia of the body link, every env) is read from
#   PhysX at the FIRST reset call, not in ``__init__``: the startup DR events (p2a: mass
#   -0.5:1.5 kg, CoM +-0.01 m) have run by then, so they are preserved exactly, per env.
# * 0 kg writes NOTHING. An env drawn 0 kg that carries no payload is not written at all (no
#   float round trip, no PhysX re-diagonalisation); one that carried a payload last episode is
#   restored to its captured base values bit for bit. ``KRABBY_PAYLOAD_KG=0:0`` is therefore a
#   true no-op on mass, CoM and inertia: no ``set_*`` call is ever made.
# * PRIVATE generator, seeded from the env seed (salted, so it is never the global stream):
#   the payload draws leave every other random stream of the env untouched.
#
# PhysX conventions (omni.physics.tensors ArticulationView): ``get_masses`` (N, L);
# ``get_coms`` (N, L, 7) = principal-axes frame pose in the link frame (x, y, z, qx, qy, qz,
# qw); ``get_inertias`` (N, L, 9) = column-major inertia ABOUT THE CoM, EXPRESSED IN THE LINK
# FRAME. ``set_inertias`` re-diagonalises and may update the principal-axes rotation, so the
# write order is masses -> coms (position + captured rotation) -> inertias. All CPU tensors.

#: ``b"payload"`` -- added to the env seed so the private stream never equals the global
#: stream ``torch.manual_seed(env_seed)`` produced.
PAYLOAD_SEED_SALT = 0x7061796C6F6164
#: The cfg attribute (and event-manager term name) ``payload_knobs.PAYLOAD_EVENT`` creates.
PAYLOAD_EVENT_NAME = "payload_reset"


def payload_seed(env_seed) -> int:
    """The private generator's seed for an env seed (``None`` -> as for 0)."""
    base = 0 if env_seed is None else int(env_seed)
    return (base + PAYLOAD_SEED_SALT) % (1 << 63)


def box_inertia_diag(mass, size) -> torch.Tensor:
    """``(n, 3)`` float64 principal moments of solid boxes of ``size = (a, b, h)`` m, about their centres.

    ``I = m/12 * (b^2 + h^2, a^2 + h^2, a^2 + b^2)`` with ``a`` along x, ``b`` along y, ``h`` along z.
    """
    m = torch.as_tensor(mass, dtype=torch.float64).reshape(-1, 1)
    a, b, h = (float(s) for s in size)
    sq = torch.tensor([b * b + h * h, a * a + h * h, a * a + b * b], dtype=torch.float64)
    return m * sq / 12.0


def _point_inertia(mass: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``m (|d|^2 E - d d^T)``, ``(n, 3, 3)``: the parallel-axis term of a mass at offset ``d``."""
    eye = torch.eye(3, dtype=d.dtype)
    dd = (d * d).sum(-1)
    return mass.reshape(-1, 1, 1) * (dd.reshape(-1, 1, 1) * eye - d.unsqueeze(-1) * d.unsqueeze(-2))


def composite_mass_properties(base_mass, base_com, base_inertia, add_mass, add_com, add_inertia):
    """Two rigid parts -> one: total mass, mass-weighted CoM and the inertia about that CoM.

    Everything in ONE frame (the body link frame): ``*_com`` ``(n, 3)`` are positions in it and
    ``*_inertia`` ``(n, 3, 3)`` are each part's inertia about its OWN CoM, expressed in its axes.
    ``I = I_b + m_b S(c_b - c) + I_a + m_a S(c_a - c)``, ``S(d) = |d|^2 E - d d^T``. float64.
    """
    mb = torch.as_tensor(base_mass, dtype=torch.float64).reshape(-1)
    ma = torch.as_tensor(add_mass, dtype=torch.float64).reshape(-1)
    cb = torch.as_tensor(base_com, dtype=torch.float64).reshape(-1, 3)
    ca = torch.as_tensor(add_com, dtype=torch.float64).reshape(-1, 3)
    ib = torch.as_tensor(base_inertia, dtype=torch.float64).reshape(-1, 3, 3)
    ia = torch.as_tensor(add_inertia, dtype=torch.float64).reshape(-1, 3, 3)
    total = mb + ma
    com = (mb.unsqueeze(-1) * cb + ma.unsqueeze(-1) * ca) / total.unsqueeze(-1)
    inertia = ib + _point_inertia(mb, cb - com) + ia + _point_inertia(ma, ca - com)
    return total, com, inertia


def draw_payload(n: int, generator: torch.Generator, kg_range, zero_frac, com_xy, com_z):
    """``n`` payload draws -> ``(kg (n,), com (n, 3))`` float64, from ``generator`` only.

    One ``(n, 4)`` uniform block per call (fixed consumption): column 0 decides "no payload this
    episode" (``u < zero_frac``), column 1 the mass ``lo + (hi - lo) u``, columns 2-3 the CoM x / y
    ``r (2u - 1)``; the CoM z is ``com_z``. A 0-kg row's CoM is recorded as zeros (there is no
    payload). ``L:L`` gives exactly L and ``com_xy = 0`` exactly +0.0, so a level row is
    deterministic whatever the generator state.
    """
    u = torch.rand((int(n), 4), generator=generator, dtype=torch.float64)
    lo, hi = float(kg_range[0]), float(kg_range[1])
    kg = lo + (hi - lo) * u[:, 1]
    kg = torch.where(u[:, 0] < float(zero_frac), torch.zeros_like(kg), kg)
    r = float(com_xy)
    com = torch.empty((int(n), 3), dtype=torch.float64)
    com[:, 0] = r * (2.0 * u[:, 2] - 1.0) + 0.0          # + 0.0 turns -0.0 into +0.0
    com[:, 1] = r * (2.0 * u[:, 3] - 1.0) + 0.0
    com[:, 2] = float(com_z)
    com[kg <= 0.0] = 0.0
    return kg, com


def check_payload_params(kg_range, zero_frac, com_xy, com_z, box_size) -> None:
    """Refuse a malformed payload term (a hand-built cfg can bypass the knob parsers)."""
    lo, hi = (float(x) for x in kg_range)
    vals = (lo, hi, float(zero_frac), float(com_xy), float(com_z), *(float(s) for s in box_size))
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"payload_reset: non-finite param in {vals}")
    if not 0.0 <= lo <= hi:
        raise ValueError(f"payload_reset: kg_range needs 0 <= lo <= hi, got {tuple(kg_range)}")
    if not 0.0 <= float(zero_frac) <= 1.0:
        raise ValueError(f"payload_reset: zero_frac must be in [0, 1], got {zero_frac}")
    if float(com_xy) < 0.0 or float(com_z) < 0.0:
        raise ValueError(f"payload_reset: com_xy and com_z must be >= 0, got {com_xy}, {com_z}")
    if len(tuple(box_size)) != 3 or min(float(s) for s in box_size) <= 0.0:
        raise ValueError(f"payload_reset: box_size must be three positive lengths, got {box_size}")


class payload_reset(ManagerTermBase):
    """Reset-mode event: a fresh payload per env per episode, written as an ABSOLUTE composite.

    Params (all required; see ``payload_knobs.payload_event_params``): ``asset_cfg`` (the robot,
    ``body_names="body"``, exactly one link), ``kg_range`` ``(lo, hi)``, ``zero_frac``,
    ``com_xy``, ``com_z``, ``box_size`` ``(a, b, h)``.

    What the harness / run_meta reads (all per env, the CURRENT episode, float64 CPU):
    ``payload_kg`` ``(N,)``, ``payload_com`` ``(N, 3)`` (body frame; zeros where 0 kg), ``loaded``
    ``(N,)`` bool, ``resets`` ``(N,)`` draws so far, the captured base ``base_mass`` / ``base_com``
    / ``base_inertia`` (None before the first reset) -- or all of it as one JSON-able dict through
    :meth:`run_meta` (module helper :func:`payload_run_meta`). Training logs two scalars,
    ``Payload/kg_mean`` and ``Payload/loaded_frac`` (over all envs), into ``extras["log"]``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        check_payload_params(params["kg_range"], params["zero_frac"], params["com_xy"], params["com_z"],
                             params["box_size"])
        self.asset_cfg = params["asset_cfg"]
        self.asset = env.scene[self.asset_cfg.name]
        body_ids = getattr(self.asset_cfg, "body_ids", None)
        if body_ids is None or isinstance(body_ids, slice) or len(body_ids) != 1:
            raise ValueError(f"payload_reset needs exactly one body in asset_cfg, got body_ids={body_ids!r}")
        self.body_id = int(body_ids[0])
        self.env_seed = getattr(getattr(env, "cfg", None), "seed", None)
        self.seed = payload_seed(self.env_seed)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        n = int(env.num_envs)
        self.base_mass: torch.Tensor | None = None       # (N,) float32, captured at the first reset
        self.base_com: torch.Tensor | None = None        # (N, 7) float32 pose (pos + principal-axes quat)
        self.base_inertia: torch.Tensor | None = None    # (N, 9) float32 column-major, about the base CoM
        self.payload_kg = torch.zeros(n, dtype=torch.float64)
        self.payload_com = torch.zeros(n, 3, dtype=torch.float64)
        self.loaded = torch.zeros(n, dtype=torch.bool)
        self.resets = torch.zeros(n, dtype=torch.long)
        self.writes = 0                                   # PhysX write batches issued (diagnostic)

    # ------------------------------------------------------------------ the event
    def __call__(self, env, env_ids, asset_cfg, kg_range, zero_frac, com_xy, com_z, box_size):
        ids = self._ids(env_ids)
        if self.base_mass is None:
            self._capture()
        if ids.numel() == 0:
            return
        kg, com = draw_payload(int(ids.numel()), self.generator, kg_range, zero_frac, com_xy, com_z)
        on = kg > 0.0
        write = on | self.loaded[ids]
        self.payload_kg[ids] = kg
        self.payload_com[ids] = com
        self.loaded[ids] = on
        self.resets[ids] += 1
        if bool(write.any()):
            self._write(ids[write], kg[write], com[write], box_size)
        self._log(env)

    def _ids(self, env_ids) -> torch.Tensor:
        n = int(self.payload_kg.shape[0])
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return torch.arange(n, dtype=torch.long)
        return torch.as_tensor(env_ids).to(device="cpu", dtype=torch.long).reshape(-1)

    def _capture(self) -> None:
        """Read every env's post-startup-DR base off PhysX (clones: the view refills its buffers)."""
        view = self.asset.root_physx_view
        b = self.body_id
        self.base_mass = view.get_masses()[:, b].detach().clone().cpu()
        self.base_com = view.get_coms()[:, b, :].detach().clone().cpu()
        self.base_inertia = view.get_inertias()[:, b, :].detach().clone().cpu()

    def _composite(self, ids: torch.Tensor, kg: torch.Tensor, com: torch.Tensor, box_size):
        """float32 ``(mass (n,), com (n, 3), inertia (n, 9))`` for ``ids``; 0-kg rows = the captured base exactly."""
        base_m = self.base_mass[ids]
        base_c = self.base_com[ids, :3]
        base_i = self.base_inertia[ids]
        # column-major 9 -> 3x3 is reshape + transpose (and back); exact either way for a symmetric tensor
        i_b = base_i.double().reshape(-1, 3, 3).transpose(1, 2)
        i_p = torch.diag_embed(box_inertia_diag(kg, box_size))
        m, c, i = composite_mass_properties(base_m, base_c, i_b, kg, com, i_p)
        new_m = m.to(torch.float32)
        new_c = c.to(torch.float32)
        new_i = i.transpose(1, 2).reshape(-1, 9).to(torch.float32)
        off = kg <= 0.0
        new_m[off] = base_m[off]
        new_c[off] = base_c[off]
        new_i[off] = base_i[off]
        return new_m, new_c, new_i

    def _write(self, w_ids: torch.Tensor, kg: torch.Tensor, com: torch.Tensor, box_size) -> None:
        view = self.asset.root_physx_view
        b = self.body_id
        new_m, new_c, new_i = self._composite(w_ids, kg, com, box_size)
        masses = view.get_masses().clone()
        coms = view.get_coms().clone()
        inertias = view.get_inertias().clone()
        masses[w_ids.to(masses.device), b] = new_m.to(masses.device)
        cidx = w_ids.to(coms.device)
        coms[cidx, b, :3] = new_c.to(coms.device)
        coms[cidx, b, 3:] = self.base_com[w_ids, 3:].to(coms.device)
        inertias[w_ids.to(inertias.device), b] = new_i.to(inertias.device)
        view.set_masses(masses, w_ids)
        view.set_coms(coms, w_ids)
        view.set_inertias(inertias, w_ids)
        self.writes += 1

    def _log(self, env) -> None:
        log = getattr(env, "extras", {}).get("log") if isinstance(getattr(env, "extras", None), dict) else None
        if isinstance(log, dict):
            log["Payload/kg_mean"] = float(self.payload_kg.mean())
            log["Payload/loaded_frac"] = float(self.loaded.to(torch.float64).mean())

    # ------------------------------------------------------------------ the record
    def run_meta(self) -> dict:
        """JSON-able record of the armed payload: params, seeds, and every env's realised values.

        ``payload_kg`` / ``payload_com_m`` / ``total_*`` describe the CURRENT episode of each env
        (they are re-drawn at every reset; a level row's are constant). ``captured`` is False, and
        the base / total entries None, before the first reset.
        """
        p = self.cfg.params
        out = {
            "event": PAYLOAD_EVENT_NAME,
            "body_id": self.body_id,
            "kg_range": [float(x) for x in p["kg_range"]],
            "zero_frac": float(p["zero_frac"]),
            "com_xy": float(p["com_xy"]),
            "com_z": float(p["com_z"]),
            "box_size_m": [float(x) for x in p["box_size"]],
            "env_seed": None if self.env_seed is None else int(self.env_seed),
            "generator_seed": int(self.seed),
            "captured": self.base_mass is not None,
            "writes": int(self.writes),
            "resets_per_env": self.resets.tolist(),
            "payload_kg": self.payload_kg.tolist(),
            "payload_com_m": self.payload_com.tolist(),
            "base_mass_kg": None, "base_com_m": None, "total_mass_kg": None, "total_com_m": None,
        }
        if self.base_mass is not None:
            ids = torch.arange(int(self.payload_kg.shape[0]), dtype=torch.long)
            m, c, _ = self._composite(ids, self.payload_kg, self.payload_com, p["box_size"])
            out["base_mass_kg"] = self.base_mass.double().tolist()
            out["base_com_m"] = self.base_com[:, :3].double().tolist()
            out["total_mass_kg"] = m.double().tolist()
            out["total_com_m"] = c.double().tolist()
        return out


def payload_term(env):
    """The live ``payload_reset`` term instance of ``env`` (or its ``.unwrapped``); None when the payload is off.

    None too while the event manager has not instantiated its class terms yet (before the sim plays).
    """
    env = getattr(env, "unwrapped", env)
    em = getattr(env, "event_manager", None)
    if em is None:
        return None
    names = [name for terms in em.active_terms.values() for name in terms]
    if PAYLOAD_EVENT_NAME not in names:
        return None
    func = em.get_term_cfg(PAYLOAD_EVENT_NAME).func
    return func if callable(getattr(func, "run_meta", None)) and not isinstance(func, type) else None


def payload_run_meta(env) -> dict | None:
    """``payload_term(env).run_meta()``, or None when the payload is off (the run_meta hook for the harness)."""
    term = payload_term(env)
    return None if term is None else term.run_meta()

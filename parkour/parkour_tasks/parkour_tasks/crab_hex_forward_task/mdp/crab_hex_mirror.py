"""Left/right mirror maps for crab-hex observations and actions (symmetry-regularized PPO).

The crab hexapod's L/R mirror maps tripod set A = {FL, MR, RL} exactly onto set B =
{FR, ML, RR}. A mirror-equivariant policy therefore cannot encode a permanent lead-set
preference — the "handedness" the seed-basin search (parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-12_1550_
seed_basin_search/) showed unconstrained training breaks randomly and the reward campaigns
showed no pricing can undo.

This module supplies the ``data_augmentation_func`` expected by ``PPOWithExtractor``'s
rsl-rl-style ``symmetry_cfg``: ``f(obs, actions, env, obs_type) -> (obs_aug, act_aug)`` where
each returned tensor is ``cat([original, mirrored])`` along the batch dim (either input may be
None). Maps are built at runtime from RESOLVED name lists (articulation joint order, action-term
order, contact-sensor order, height-scanner pattern cfg) — never hand-typed indices.

Sign conventions (see the campaign ledger 2026-08-13_0035_mirror_symmetry/RESULTS.md for the
full derivation):

- ``*_Body_CamShaft_*``: L/R swap, sign flip. The Whitworth cam map
  ``theta_hip = atan2(K sin t, 1 + K cos t)`` is odd, so mirrored defaults negate exactly.
  Under velocity actions (2026-08-13) the channel is a signed shaft SPEED and the obs
  channel is the wrapped-to-[-pi,pi] shaft angle: both are odd quantities, so the same
  -1 sign applies unchanged.
- ``*_Body_Hip_*`` (passive): swap, sign flip (yaw-type; defaults negate L/R).
- ``*_Hip_Femur_*``: swap, no flip (leg-plane pitch joint; defaults equal L/R).
- ``*_Femur_Tibia_*``: swap, sign flip (180-deg Z USD flip on right legs). Since
  2026-08-20 the knee defaults are exactly mirrored (+/-0.2341, actuator mid-stroke from
  the measured linkage), so the mirror is exact — the old -0.07/+0.10 roll-balance
  asymmetry (and its "approximation" caveat) is retired with the hardware geometry.

Proprio head (15 dims of the crab obs step): sign flips on roll-axis and yaw-axis channels and
on every lateral (y) channel; see ``_HEAD_SIGNS``.
"""

from __future__ import annotations

import torch

# Per-step proprio head layout of CrabHexParkourObservations (before the joint blocks):
# [wx, wy, wz, roll, pitch, 0*dy, delta_yaw, delta_next_yaw, 0*cmd_vx, 0*cmd_vy, cmd_vx,
#  env_idx, invert_env_idx, lin_vx, lin_vy, clock_sin, clock_cos]
# Clock dims (gait-formation-v2 Phase 1): the mirrored gait is the same alternating-tripod
# schedule advanced by pi (tripod sets swap under L/R), and sin/cos(phi+pi) = -sin/-cos, so
# both dims carry sign -1.
_HEAD_SIGNS = [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
               -1.0, -1.0]
_N_HEAD = len(_HEAD_SIGNS)

# joint-name substring -> mirror sign
_JOINT_SIGN_RULES = (
    ("_Body_CamShaft_", -1.0),
    ("_Body_Hip_", -1.0),
    ("_Hip_Femur_", 1.0),
    ("_Femur_Tibia_", -1.0),
)

_LEG_SWAP = {"FL": "FR", "FR": "FL", "ML": "MR", "MR": "ML", "RL": "RR", "RR": "RL"}


def _mirror_name(name: str, leg_swap: dict[str, str] = _LEG_SWAP) -> str:
    leg = name[:2]
    if leg not in leg_swap:
        raise ValueError(f"Cannot determine leg prefix for '{name}'")
    return leg_swap[leg] + name[2:]


def _joint_sign(name: str, rules: tuple[tuple[str, float], ...] = _JOINT_SIGN_RULES) -> float:
    for token, sign in rules:
        if token in name:
            return sign
    raise ValueError(f"No mirror sign rule matches joint '{name}'")


def build_name_permutation(names: list[str], leg_swap: dict[str, str] = _LEG_SWAP) -> list[int]:
    """Index permutation such that out[i] = in[perm[i]] gives the mirrored vector."""
    index = {n: i for i, n in enumerate(names)}
    perm = []
    for n in names:
        m = _mirror_name(n, leg_swap)
        if m not in index:
            raise ValueError(f"Mirror partner '{m}' of '{n}' not in name list")
        perm.append(index[m])
    return perm


def build_scan_permutation(nx: int, ny: int, ordering: str, axis: str = "y") -> list[int]:
    """Flip of a grid_pattern ray layout across the mirror plane.

    ``axis="y"`` is the lateral (left/right) flip of this module's L/R mirror; ``axis="x"``
    is the fore-aft flip a sibling task's mirror plane (x = 0) needs.

    isaaclab grid_pattern: ordering 'xy' -> meshgrid(x, y, indexing='xy') -> flat index
    iy * nx + ix; ordering 'yx' -> indexing 'ij' -> flat index ix * ny + iy.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"Unknown mirror axis '{axis}'")
    perm = [0] * (nx * ny)
    if ordering == "xy":
        for iy in range(ny):
            for ix in range(nx):
                if axis == "y":
                    perm[iy * nx + ix] = (ny - 1 - iy) * nx + ix
                else:
                    perm[iy * nx + ix] = iy * nx + (nx - 1 - ix)
    elif ordering == "yx":
        for ix in range(nx):
            for iy in range(ny):
                if axis == "y":
                    perm[ix * ny + iy] = ix * ny + (ny - 1 - iy)
                else:
                    perm[ix * ny + iy] = (nx - 1 - ix) * ny + iy
    else:
        raise ValueError(f"Unknown grid ordering '{ordering}'")
    return perm


class CrabHexMirror:
    """Builds and applies the full obs/action mirror for one env instance.

    The keyword-only arguments carry the mirror *plane*: the defaults (``None`` -> this
    module's constants, ``scan_axis="y"``, ``vec3_signs=(1, -1, 1)``) are the crab-hex
    left/right mirror. A sibling task whose travel axis differs subclasses this and passes
    its own constants; nothing about the forward behaviour depends on them.
    """

    def __init__(
        self,
        joint_names: list[str],
        action_joint_names: list[str],
        contact_body_names: list[str],
        scan_nx: int,
        scan_ny: int,
        scan_ordering: str,
        history_length: int,
        device: torch.device | str,
        *,
        head_signs: list[float] | None = None,
        joint_sign_rules: tuple[tuple[str, float], ...] | None = None,
        leg_swap: dict[str, str] | None = None,
        scan_axis: str = "y",
        vec3_signs: tuple[float, float, float] = (1.0, -1.0, 1.0),
    ):
        head_signs = _HEAD_SIGNS if head_signs is None else head_signs
        joint_sign_rules = _JOINT_SIGN_RULES if joint_sign_rules is None else joint_sign_rules
        leg_swap = _LEG_SWAP if leg_swap is None else leg_swap

        n_head = len(head_signs)
        nj = len(joint_names)
        na = len(action_joint_names)
        nc = len(contact_body_names)
        ns = scan_nx * scan_ny

        joint_perm = build_name_permutation(joint_names, leg_swap)
        joint_sign = [_joint_sign(joint_names[p], joint_sign_rules) for p in joint_perm]
        act_perm = build_name_permutation(action_joint_names, leg_swap)
        act_sign = [_joint_sign(action_joint_names[p], joint_sign_rules) for p in act_perm]
        contact_perm = build_name_permutation(contact_body_names, leg_swap)
        scan_perm = build_scan_permutation(scan_nx, scan_ny, scan_ordering, scan_axis)

        n_prop = n_head + 2 * nj + na + nc

        # Assemble the per-step (n_prop) map.
        step_perm: list[int] = list(range(n_head))
        step_sign: list[float] = list(head_signs)
        base = n_head
        for block_sign_from_perm, size, perm, signs in (
            (True, nj, joint_perm, joint_sign),      # joint_pos - default
            (True, nj, joint_perm, joint_sign),      # joint_vel
            (True, na, act_perm, act_sign),          # last action
            (True, nc, contact_perm, [1.0] * nc),    # contact fill
        ):
            step_perm.extend(base + p for p in perm)
            step_sign.extend(signs)
            base += size
        assert base == n_prop

        # Full policy obs: n_prop + scan + priv_explicit(9) + priv_latent(6 + 2*nj) + history.
        full_perm: list[int] = list(step_perm)
        full_sign: list[float] = list(step_sign)
        off = n_prop
        full_perm.extend(off + p for p in scan_perm)
        full_sign.extend([1.0] * ns)
        off += ns
        # priv_explicit: three 3-vectors in body frame (lin_vel*2, zeros, zeros): vec3_signs each
        for _ in range(3):
            full_perm.extend([off, off + 1, off + 2])
            full_sign.extend(vec3_signs)
            off += 3
        # priv_latent: mass(1)+, com(3) vec3_signs, friction(1)+, stiff_ratio(nj) perm+, damp_ratio(nj) perm+
        full_perm.append(off); full_sign.append(1.0); off += 1
        full_perm.extend([off, off + 1, off + 2]); full_sign.extend(vec3_signs); off += 3
        full_perm.append(off); full_sign.append(1.0); off += 1
        for _ in range(2):
            full_perm.extend(off + p for p in joint_perm)
            full_sign.extend([1.0] * nj)  # ratios are magnitudes: permute, no flip
            off += nj
        # history: history_length copies of the per-step map
        for _ in range(history_length):
            full_perm.extend(off + p for p in step_perm)
            full_sign.extend(step_sign)
            off += n_prop

        self.n_prop = n_prop
        self.obs_dim = off
        self.obs_perm = torch.tensor(full_perm, dtype=torch.long, device=device)
        self.obs_sign = torch.tensor(full_sign, dtype=torch.float32, device=device)
        self.act_perm = torch.tensor(act_perm, dtype=torch.long, device=device)
        self.act_sign = torch.tensor(act_sign, dtype=torch.float32, device=device)

    def mirror_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"obs dim {obs.shape[-1]} != expected {self.obs_dim}")
        return obs[..., self.obs_perm] * self.obs_sign

    def mirror_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., self.act_perm] * self.act_sign


_MIRROR_CACHE: dict[int, CrabHexMirror] = {}

_N_SCAN = None  # computed from the scanner pattern at build time
_N_PRIV = 9 + 5  # priv_explicit(9) + priv_latent head (mass 1 + com 3 + friction 1)


def _resolve_layout(env, obs_dim: int, n_head: int = _N_HEAD) -> dict:
    """Resolve the live env's name lists / scan grid / history length into ``CrabHexMirror`` kwargs.

    Mirror-plane independent: a sibling task's mirror reuses this verbatim and supplies its
    own sign constants. ``n_head`` is the per-step proprio head length the mirror will use
    (17 for both the L/R and the fore-aft mirror -- same observation term).
    """
    uenv = getattr(env, "unwrapped", env)
    robot = uenv.scene["robot"]
    action_term = uenv.action_manager.get_term("joint_pos")
    action_names = list(getattr(action_term, "_joint_names", None) or action_term.joint_names)
    sensor = uenv.scene.sensors["contact_forces"]
    # same set the obs term's ".*_Footpad" SceneEntityCfg resolves to, in sensor body order
    contact_names = [n for n in sensor.body_names if n.endswith("_Footpad")]
    scanner = uenv.scene.sensors["height_scanner"]
    pat = scanner.cfg.pattern_cfg
    import math

    nx = int(math.floor(pat.size[0] / pat.resolution + 1e-9)) + 1
    ny = int(math.floor(pat.size[1] / pat.resolution + 1e-9)) + 1
    nj = len(robot.joint_names)
    n_prop = n_head + 2 * nj + len(action_names) + len(contact_names)
    # obs_dim = n_prop + scan + priv(9 + 5 + 2*nj) + H * n_prop
    remainder = obs_dim - n_prop - nx * ny - _N_PRIV - 2 * nj
    if remainder < 0 or remainder % n_prop != 0:
        raise ValueError(
            f"obs dim {obs_dim} inconsistent with n_prop={n_prop}, scan={nx * ny}, "
            f"priv={_N_PRIV + 2 * nj} — obs layout changed; update crab_hex_mirror.py"
        )
    history_length = remainder // n_prop
    return dict(
        joint_names=list(robot.joint_names),
        action_joint_names=action_names,
        contact_body_names=contact_names,
        scan_nx=nx,
        scan_ny=ny,
        scan_ordering=getattr(pat, "ordering", "xy"),
        history_length=history_length,
        device=uenv.device,
    )


def _get_mirror(env, obs_dim: int) -> CrabHexMirror:
    """Build (once per env) from resolved names; history length inferred from ``obs_dim``."""
    key = id(env)
    if key not in _MIRROR_CACHE:
        _MIRROR_CACHE[key] = CrabHexMirror(**_resolve_layout(env, obs_dim))
    return _MIRROR_CACHE[key]


def _augment(get_mirror, cache: dict, obs=None, actions=None, env=None, obs_type="policy"):
    """Shared body of the rsl-rl ``data_augmentation_func`` entry points.

    ``get_mirror``/``cache`` are the per-mirror builder and its cache, so a sibling task's
    entry point reuses this with its own mirror class and its own cache dict.
    """
    if obs is None and id(env) not in cache:
        raise RuntimeError(
            "the symmetry augmentation entry point must see an obs batch before an actions-only call "
            "(the mirror is sized from the live obs dim)"
        )
    mirror = get_mirror(env, obs.shape[-1] if obs is not None else 0)
    obs_out = None
    act_out = None
    if obs is not None:
        if obs.shape[-1] != mirror.obs_dim:
            raise ValueError(
                f"symmetry augmentation: {obs_type} obs dim {obs.shape[-1]} != policy dim "
                f"{mirror.obs_dim}; extend the map before using a distinct critic space"
            )
        obs_out = torch.cat([obs, mirror.mirror_obs(obs)], dim=0)
    if actions is not None:
        act_out = torch.cat([actions, mirror.mirror_actions(actions)], dim=0)
    return obs_out, act_out


def crab_hex_symmetry_augmentation(obs=None, actions=None, env=None, obs_type="policy"):
    """rsl-rl symmetry entry point: returns (cat([obs, mirror(obs)]), cat([act, mirror(act)]))."""
    return _augment(_get_mirror, _MIRROR_CACHE, obs, actions, env, obs_type)

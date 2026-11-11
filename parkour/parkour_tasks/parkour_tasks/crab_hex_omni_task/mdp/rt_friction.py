# SPDX-License-Identifier: BSD-3-Clause
"""The rough-terrain FRICTION CONTRACT (``experiments/2026-11-10_rough_terrain/PLAN.md`` Design "Friction contract",
step 2, 2026-11-10). DEFAULT OFF.

Armed only by ``omni_terrain_knobs.apply_rt_friction_knobs`` (``KRABBY_RT_FRICTION=gravel:LO:HI``, under
``KRABBY_FLAT_TERRAIN_GEOM=site`` + ``KRABBY_SITE_LADDER=rt``), which sets the ground's friction combine mode to
``multiply`` at mu 1.0 and replaces the robot's startup material event (TDR's ``randomize_rigid_body_material``) by
:func:`rt_friction_contract`. With the ground at 1.0 under ``multiply``, a contact's effective mu IS the robot shape's,
so the event writes the EFFECTIVE mu straight onto EVERY shape of an env (static = dynamic, restitution 0 as TDR's
event writes it):

* an env on a non-gravel column: ``mu_eff = (b + 1) / 2``, ``b`` one of ``num_buckets`` bucket values ``U(friction_range)``
  (TDR's 64 x U(0.6, 2.0)) picked uniformly per env -- under TDR's ``average`` combine against the ground's 1.0 that
  is exactly TDR's effective mu, so TDR's friction distribution is kept;
* an env on a gravel column (``gravel_sub_terrains``; columns never change, so one draw at startup is the env's
  episode-long mu): ``mu_eff ~ U(gravel_range)``.

Every draw comes from a PRIVATE numpy generator keyed by the env seed and the env INDEX (:func:`contract_mu`): the bucket
values from ``(salt, seed, 0)``, an env's own pick from ``(salt, seed, 1, env_index)`` -- so env ``i`` gets the same mu
in every process built with the same seed, whatever the env count, the global RNG streams or the other envs (the
pairing two teachers' eval rows rely on). No global RNG is touched.

The privileged friction observation (``parkour_isaaclab/envs/mdp/observations.py``) reads ``2 mu_eff - 1`` when the env
cfg carries :data:`RT_FRICTION_ATTR`: TDR's own input (the shape's raw ``b``) on non-gravel envs, ``2 U(LO, HI) - 1`` on
gravel ones.

Module scope is numpy / torch only (CPU-testable); the event body touches only ``env.scene`` and ``env.cfg``.
"""
from __future__ import annotations

import math

import numpy as np
import torch

__all__ = [
    "RT_FRICTION_ATTR", "RT_FRICTION_SEED_SALT", "RT_MU_ATTR", "rt_friction_seed", "bucket_values", "contract_mu",
    "friction_obs", "gravel_env_mask", "rt_friction_contract",
]

#: The env-cfg attribute the knob writes (``omni_terrain_knobs.RT_FRICTION_ATTR``; a test pins the two).
RT_FRICTION_ATTR = "rt_friction"
#: ``b"rtfric"`` -- mixed into every key, so the stream never coincides with another private stream.
RT_FRICTION_SEED_SALT = 0x727466726963
#: The env attribute the event leaves the realised ``(N,)`` effective mu on (smokes / telemetry read it).
RT_MU_ATTR = "_krabby_rt_mu_eff"
#: The env attribute holding the ``(N,)`` bool gravel-env mask the event used.
RT_GRAVEL_ATTR = "_krabby_rt_gravel_env"


def rt_friction_seed(env_seed) -> int:
    """The contract's seed for an env seed (``None`` -> as for 0)."""
    return 0 if env_seed is None else int(env_seed) % (1 << 63)


def bucket_values(seed: int, friction_range, num_buckets: int) -> np.ndarray:
    """The ``num_buckets`` bucket values ``b ~ U(friction_range)`` (float64) of the contract at ``seed``."""
    lo, hi = (float(v) for v in friction_range)
    rng = np.random.default_rng([RT_FRICTION_SEED_SALT, rt_friction_seed(seed), 0])
    return lo + (hi - lo) * rng.random(int(num_buckets))


def contract_mu(seed: int, env_index, gravel, friction_range, num_buckets: int, gravel_range) -> np.ndarray:
    """``(n,)`` float64 effective mu of the envs ``env_index`` (ints) whose gravel flags are ``gravel`` (bools).

    Per env, from its own generator ``(salt, seed, 1, index)``: ``u0`` picks the bucket ``floor(u0 * num_buckets)``
    (``mu = (b + 1) / 2``), ``u1`` the gravel mu ``lo + (hi - lo) u1``; both are drawn for every env, so an env's
    numbers do not depend on its column."""
    idx = np.asarray(env_index, dtype=np.int64).reshape(-1)
    grav = np.asarray(gravel, dtype=bool).reshape(-1)
    if idx.shape != grav.shape:
        raise ValueError(f"env_index {idx.shape} and gravel {grav.shape} must have one entry per env")
    buckets = bucket_values(seed, friction_range, num_buckets)
    g_lo, g_hi = (float(v) for v in gravel_range)
    s = rt_friction_seed(seed)
    out = np.empty(idx.shape, dtype=np.float64)
    for k, i in enumerate(idx.tolist()):
        u0, u1 = np.random.default_rng([RT_FRICTION_SEED_SALT, s, 1, int(i)]).random(2)
        if grav[k]:
            out[k] = g_lo + (g_hi - g_lo) * u1
        else:
            b = buckets[min(int(math.floor(u0 * len(buckets))), len(buckets) - 1)]
            out[k] = 0.5 * (b + 1.0)
    return out


def friction_obs(mu_eff):
    """The privileged friction input under the contract: ``2 mu_eff - 1`` (TDR's ``b`` on non-gravel envs)."""
    return 2.0 * mu_eff - 1.0


def gravel_env_mask(terrain_names, terrain_types, gravel_keys) -> np.ndarray:
    """``(N,)`` bool: the env's column (``terrain_types``) is a gravel column (``terrain_names`` ``(R, C[, 1])`` in
    ``gravel_keys``). A column mixing gravel and other tiles (random, non-curriculum layouts of a mixed grid) is
    refused: an env's level moves within its column, so its mu would not match its tile."""
    names = np.asarray(terrain_names)
    if names.ndim == 3:
        names = names[..., 0]
    is_g = np.isin(names.astype(str), np.asarray(tuple(gravel_keys), dtype=str))
    col_any, col_all = is_g.any(axis=0), is_g.all(axis=0)
    mixed = np.nonzero(col_any & ~col_all)[0]
    if mixed.size:
        raise RuntimeError(f"rt friction contract: terrain columns {mixed.tolist()} mix {tuple(gravel_keys)} tiles with "
                           "other kinds; the contract needs every column all-gravel or gravel-free")
    types = np.asarray(torch.as_tensor(terrain_types).detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    return col_all[types]


def rt_friction_contract(
    env,
    env_ids,
    asset_cfg,
    friction_range: tuple[float, float],
    num_buckets: int,
    gravel_range: tuple[float, float],
    gravel_sub_terrains: tuple[str, ...] = ("rt_gravel",),
) -> None:
    """The contract's startup material event (replaces TDR's ``randomize_rigid_body_material``): EVERY shape of each
    env in ``env_ids`` (all envs when None) gets the env's :func:`contract_mu` as static and dynamic friction,
    restitution 0. Leaves the ``(N,)`` mu on ``env._krabby_rt_mu_eff`` and the gravel mask on
    ``env._krabby_rt_gravel_env``."""
    asset = env.scene[asset_cfg.name]
    n = int(env.scene.num_envs)
    ids = torch.arange(n, device="cpu") if env_ids is None else torch.as_tensor(env_ids).detach().to("cpu").long()
    terrain = env.scene.terrain
    gen = getattr(terrain, "terrain_generator_class", None)
    names = getattr(gen, "terrain_names", None) if gen is not None else None
    if names is None or getattr(terrain, "terrain_types", None) is None:
        raise RuntimeError("rt friction contract: the terrain importer has no generator terrain_names / terrain_types")
    gravel_all = gravel_env_mask(names, terrain.terrain_types, gravel_sub_terrains)
    seed = getattr(env.cfg, "seed", None)
    mu = contract_mu(seed, ids.numpy(), gravel_all[ids.numpy()], friction_range, num_buckets, gravel_range)
    materials = asset.root_physx_view.get_material_properties()
    mu_t = torch.as_tensor(mu, dtype=materials.dtype, device=materials.device)
    materials[ids, :, 0] = mu_t[:, None]
    materials[ids, :, 1] = mu_t[:, None]
    materials[ids, :, 2] = 0.0
    asset.root_physx_view.set_material_properties(materials, ids)
    full = getattr(env, RT_MU_ATTR, None)
    if full is None:
        full = torch.full((n,), float("nan"), dtype=torch.float64)
    full[ids] = torch.as_tensor(mu, dtype=torch.float64)
    setattr(env, RT_MU_ATTR, full)
    setattr(env, RT_GRAVEL_ATTR, torch.as_tensor(gravel_all))
    g = gravel_all[ids.numpy()]
    print(f"[rt_friction] {len(mu)} envs: {int(g.sum())} gravel mu in [{mu[g].min() if g.any() else math.nan:.3f}, "
          f"{mu[g].max() if g.any() else math.nan:.3f}], {int((~g).sum())} others mu in "
          f"[{mu[~g].min() if (~g).any() else math.nan:.3f}, {mu[~g].max() if (~g).any() else math.nan:.3f}] "
          f"(seed {seed})", flush=True)

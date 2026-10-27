# SPDX-License-Identifier: BSD-3-Clause
"""Terrain-relative swing-lift terms (omni phase-2 stage p2b, ``P2B_PLAN.md`` step 1, 2026-10-08). DEFAULT OFF.

No rewards cfg names these functions: ``config/crab_hex/motion_knobs.apply_motion_knobs`` swaps
them in as the ``func`` of the forward cfg's EXISTING ``reward_foot_clearance`` /
``penalty_swing_min_clearance`` terms only under ``KRABBY_FOOT_CLEAR_TERRAIN=1``, so every recorded
run's reward manager -- and its config dump -- is untouched.

What changes, and only that
---------------------------
The library terms (``parkour_isaaclab/envs/mdp/rewards.py``) measure a swing foot's height above a
NOMINAL ground ``root_z + ground_offset_from_root_m``. On the obstacle field that pays a crouch (a
lower root raises every foot's "clearance" at fixed foot positions), measures a foot hanging into a
pit from the floor it is not over, and a foot lifted onto a block from the floor below it. Here the
height is ``foot_z - terrain_z(foot xy)``: the height field sampled under the foot, on whichever
tile the foot is ON (``omni_exposure.tiled_pixel_index``, the field-mode seam-aware lookup, so a foot
across a seam reads the neighbour's block or pit). A foot past the outermost tiles reads
``terrain_z = 0`` (the generator's base plane).

Everything else is the library's, restated line for line: the swing test (current contact force
``<= contact_force_threshold``), the relu floor / clamp of the income, the micro-swing indicator of
the penalty, the command gate (``rewards._cmd_gate``, signed -- under ``omni_command`` the clock
speed) and the parkour_flat mask (``rewards._parkour_flat_mask``, skipped when ``parkour_name`` is
None: ``KRABBY_FOOT_CLEAR_FLAT=1``). The signatures are the library's too, so the swap passes the
term's params through; ``ground_offset_from_root_m`` is accepted and IGNORED.

The terrain sample is the one ``experiments/2026-10-07_omni_phase2/scripts/p2_smoke.py``'s
``terrain_z`` takes (S3 clearance smoke): the generator's ``height_fields * vertical_scale``
assembled into one global field like ``exposure_stats.global_height_field`` (cached on the env,
on its device), indexed with the omni ``base_parkour`` term's tile frame (``env_origins`` = tile
centres, ``terrain_levels`` / ``terrain_types`` = own tile, ``_tile_size`` / ``_hscale`` /
``_tile_px`` / ``_grid_rc``). Nearest pixel, no interpolation -- the same sample the smoke and the
exposure flags use.

Pure torch at module scope: :func:`terrain_height_under` / :func:`foot_height_above_terrain` take
tensors only (CPU-tested in ``tests/unit/test_omni_p2b_knobs.py``); the env-facing functions duck-type
``env`` and import the library helpers lazily (:func:`_lib_rewards`), so a test can stand them in.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch

from parkour_isaaclab.envs.mdp.parkours.exposure_stats import global_height_field

from .omni_exposure import tiled_pixel_index

#: The omni parkour term whose tile frame indexes the height field (``OmniParkourEvent``). Fixed,
#: not the term's ``parkour_name`` param: ``KRABBY_FOOT_CLEAR_FLAT=1`` sets that to None.
TERRAIN_TERM_NAME = "base_parkour"
#: The env attribute the metric global height field is cached under (built once; terrain is static).
HEIGHT_FIELD_CACHE_ATTR = "_krabby_omni_height_field_m"

_LIB_REWARDS_MOD = "parkour_isaaclab.envs.mdp.rewards"

__all__ = [
    "TERRAIN_TERM_NAME", "HEIGHT_FIELD_CACHE_ATTR", "terrain_height_under", "foot_height_above_terrain",
    "global_height_field_m", "reward_foot_clearance_terrain", "penalty_swing_min_clearance_terrain",
]


# ------------------------------------------------------------------------------ pure helpers
def global_height_field_m(height_fields, vertical_scale: float, device=None) -> torch.Tensor:
    """``(rows, cols, W, L)`` generator fields (pixel units) -> the ``(rows*W, cols*L)`` float32
    global field in METRES, assembled exactly like ``exposure_stats.global_height_field``."""
    hf = torch.as_tensor(np.asarray(height_fields, dtype=np.float32)) * float(vertical_scale)
    hf = global_height_field(hf).contiguous()
    return hf if device is None else hf.to(device)


def terrain_height_under(
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    tile_centre_w: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    hf_global_m: torch.Tensor,
    size_xy: tuple[float, float],
    horizontal_scale: float,
    tile_px: tuple[int, int],
    grid_rc: tuple[int, int],
) -> torch.Tensor:
    """World z of the terrain under each ``(x_w, y_w)`` (``(n,)`` or ``(n, k)``), metres.

    The pixel of the tile the point is ON (own or a neighbour, :func:`omni_exposure.tiled_pixel_index`)
    in the global metric field :func:`global_height_field_m`; 0 where no tile exists (off the terrain).
    """
    nr, nc, ix, iy, on_terrain, _off_tile = tiled_pixel_index(
        x_w, y_w, tile_centre_w, row, col, size_xy, horizontal_scale, tile_px, grid_rc
    )
    w, ell = int(tile_px[0]), int(tile_px[1])
    z = hf_global_m[nr * w + ix, nc * ell + iy]
    return torch.where(on_terrain, z, torch.zeros_like(z))


def foot_height_above_terrain(
    foot_pos_w: torch.Tensor,
    tile_centre_w: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    hf_global_m: torch.Tensor,
    size_xy: tuple[float, float],
    horizontal_scale: float,
    tile_px: tuple[int, int],
    grid_rc: tuple[int, int],
) -> torch.Tensor:
    """``(N, F)`` height of each foot above the terrain directly under it: ``foot_z - terrain_z``.

    ``foot_pos_w`` is ``(N, F, 3)`` world positions; the root never enters, so lowering the body at
    fixed feet changes nothing (the crouch the root-relative library form pays for).
    """
    tz = terrain_height_under(foot_pos_w[..., 0], foot_pos_w[..., 1], tile_centre_w, row, col,
                              hf_global_m, size_xy, horizontal_scale, tile_px, grid_rc)
    return foot_pos_w[..., 2] - tz


# ------------------------------------------------------------------------------ env side
def _lib_rewards():
    """The library rewards module (``_cmd_gate`` / ``_parkour_flat_mask``), imported lazily (Isaac)."""
    return importlib.import_module(_LIB_REWARDS_MOD)


def _terrain_term(env):
    term = env.parkour_manager.get_term(TERRAIN_TERM_NAME)
    for attr in ("env_origins", "terrain", "_tile_size", "_hscale", "_tile_px", "_grid_rc"):
        if not hasattr(term, attr):
            raise RuntimeError(
                f"the terrain-relative clearance terms need the omni {TERRAIN_TERM_NAME!r} parkour term "
                f"(OmniParkourEvent) and its tile frame; {type(term).__name__} has no {attr!r}"
            )
    return term


def _height_field_m(env, term) -> torch.Tensor:
    """The global metric height field, built once per env and cached on it (on ``env.device``)."""
    hf = getattr(env, HEIGHT_FIELD_CACHE_ATTR, None)
    if hf is None:
        terrain = term.terrain
        hf = global_height_field_m(terrain.terrain_generator_class.height_fields,
                                   terrain.cfg.terrain_generator.vertical_scale, device=env.device)
        setattr(env, HEIGHT_FIELD_CACHE_ATTR, hf)
    return hf


def _foot_height_terrain(env, asset, foot_ids) -> torch.Tensor:
    """``(N, F)`` terrain-relative foot height for the env's live feet (the omni term's tile frame)."""
    term = _terrain_term(env)
    terrain = term.terrain
    return foot_height_above_terrain(
        asset.data.body_pos_w[:, foot_ids, :], term.env_origins[:, :2],
        terrain.terrain_levels, terrain.terrain_types, _height_field_m(env, term),
        term._tile_size, term._hscale, term._tile_px, term._grid_rc,
    )


def reward_foot_clearance_terrain(
    env,
    asset_cfg,
    sensor_cfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 0.1,
    min_clearance_m: float = 0.05,
    max_clearance_m: float = 0.20,
    min_forward_speed_cmd: float = 0.12,
    ground_offset_from_root_m: float = -1.0,
    parkour_name: str | None = "base_parkour",
) -> torch.Tensor:
    """``rewards.reward_foot_clearance`` with the swing height measured above the TERRAIN under the
    foot. ``ground_offset_from_root_m`` is accepted (the swap passes the term's params through) and
    ignored."""
    del ground_offset_from_root_m
    lib = _lib_rewards()
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_ids = sensor_cfg.body_ids
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_ids]
    in_contact = torch.norm(net_contact_forces, dim=-1) > contact_force_threshold
    in_swing = ~in_contact

    height_above_ground = _foot_height_terrain(env, asset, foot_ids)
    clearance = torch.relu(height_above_ground - min_clearance_m)
    clearance = torch.clamp(clearance, max=max_clearance_m)

    per_foot = in_swing.float() * clearance
    rew = torch.mean(per_foot, dim=1)

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = lib._cmd_gate(env, cmd, min_forward_speed_cmd, signed=True)
    rew = rew * forward_cmd.float()

    if parkour_name is not None:
        rew = rew * (1.0 - lib._parkour_flat_mask(env, parkour_name))
    return rew


def penalty_swing_min_clearance_terrain(
    env,
    asset_cfg,
    sensor_cfg,
    command_name: str = "base_velocity",
    contact_force_threshold: float = 0.1,
    min_clearance_m: float = 0.03,
    min_forward_speed_cmd: float = 0.12,
    ground_offset_from_root_m: float = -1.0,
    parkour_name: str | None = "base_parkour",
) -> torch.Tensor:
    """``rewards.penalty_swing_min_clearance`` (micro-swing indicator) with the swing height measured
    above the TERRAIN under the foot. ``ground_offset_from_root_m`` is accepted and ignored."""
    del ground_offset_from_root_m
    lib = _lib_rewards()
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors.get(sensor_cfg.name)
    if contact_sensor is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_ids = sensor_cfg.body_ids
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, foot_ids]
    in_swing = ~(torch.norm(net_contact_forces, dim=-1) > contact_force_threshold)
    height_above_ground = _foot_height_terrain(env, asset, foot_ids)
    micro_swing = in_swing.float() * (height_above_ground < min_clearance_m).float()
    penalty = torch.mean(micro_swing, dim=1)

    cmd = env.command_manager.get_command(command_name)
    forward_cmd = lib._cmd_gate(env, cmd, min_forward_speed_cmd, signed=True)
    penalty = penalty * forward_cmd.float()
    if parkour_name is not None:
        penalty = penalty * (1.0 - lib._parkour_flat_mask(env, parkour_name))
    return penalty

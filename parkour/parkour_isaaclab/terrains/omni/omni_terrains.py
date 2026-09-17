
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..utils import parkour_field_to_mesh_xy
from ..extreme_parkour.extreme_parkour_terrians import padding_height_field_raw, random_uniform_terrain

if TYPE_CHECKING:
    from . import omni_terrains_cfg

"""D8 (composed-moth plan): a ring of discrete raised blocks (bars and slabs) around a
clear central disc, for the omni task's spawn platform. No isaaclab import here -- only
the TYPE_CHECKING-guarded cfg-class import -- so this module (and `ring_height_field`,
its pure, seed-injected body) can be exercised in a unit test without a running Kit
process, the same way `extreme_parkour_terrians.py` already can.

Pixel/unit conventions follow the existing `extreme_parkour_terrians.py` functions
exactly, since both are wrapped by the same `terrains/utils.py` decorator family:
  * `cfg.size` is READ from inside the raw body (this function), by which point the
    `parkour_field_to_mesh*` decorator has already shrunk it by its own border (2 px at
    `border_width=0`, the common case for every sub-terrain cfg in this codebase) -- so
    pixel counts here are computed with `round()` (not the decorator's own `+1`) off the
    ALREADY-REDUCED `cfg.size`, matching every existing terrain function's own inner
    `width_pixels = int(cfg.size[0] / cfg.horizontal_scale)` line.
  * heights are stored in the raw field as integer counts of `cfg.vertical_scale` (not
    metres) until `padding_height_field_raw` rounds/casts to int16 -- again matching the
    existing functions.
  * `goals` are computed in PIXEL coordinates measured from the tile's (0, 0) corner (not
    yet centred); the decorator subtracts half the (reduced) tile size afterwards. Returned
    goal heights and the height field follow the `parkour_hurdle_terrain` convention
    (goal heights returned in METRES, not vertical-scale counts).
"""


def _tile_pixel_dims(cfg) -> tuple[int, int]:
    """Pixel dimensions of the (already border-reduced) tile. `round()`, not the bare
    truncation the decorator itself uses for the OUTER (pre-reduction) size, guards
    against the reduced size landing a hair under/over an exact pixel boundary due to
    floating-point noise introduced by the decorator's own `dim * horizontal_scale`
    reconstruction.
    """
    width_pixels = int(round(cfg.size[0] / cfg.horizontal_scale))
    length_pixels = int(round(cfg.size[1] / cfg.horizontal_scale))
    return width_pixels, length_pixels


def _sector_anchor_goals(cfg, num_goals: int) -> tuple[np.ndarray, np.ndarray]:
    """The `num_goals` sector anchors at the ring's mid-radius, evenly spaced in angle,
    in the same pixel/corner convention as every other terrain function's `goals` return.
    These are reference markers only (the omni parkour term never advances a "current
    goal" index over them), so `goal_heights` is a flat zero -- the anchor sits on the
    annulus, which may or may not be locally raised, and no single block "owns" it.
    """
    hs = cfg.horizontal_scale
    cx = 0.5 * cfg.size[0]
    cy = 0.5 * cfg.size[1]
    r_mid = 0.5 * (float(cfg.clear_radius) + float(cfg.ring_outer_radius))

    angles = 2.0 * np.pi * np.arange(num_goals) / num_goals
    goals = np.zeros((num_goals, 2))
    goals[:, 0] = (cx + r_mid * np.cos(angles)) / hs
    goals[:, 1] = (cy + r_mid * np.sin(angles)) / hs
    goal_heights = np.zeros(num_goals)
    return goals, goal_heights


def _oriented_box_mask(dx: np.ndarray, dy: np.ndarray, yaw: float, length: float, width: float) -> np.ndarray:
    """Boolean mask, same shape as `dx`/`dy`: True where the tile-centred point (dx, dy)
    [metres] falls inside a `length` (local x) x `width` (local y) box centred at the
    origin and rotated by `yaw` (radians, world frame)."""
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    return (np.abs(local_x) <= 0.5 * length) & (np.abs(local_y) <= 0.5 * width)


def ring_height_field(
    rng, difficulty: float, cfg: "omni_terrains_cfg.OmniRingTerrainCfg", num_goals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure body of `omni_ring_terrain`: a square tile with a clear central disc out to
    `cfg.clear_radius`, an annulus (out to `cfg.ring_outer_radius`) scattered with
    `cfg.n_blocks` randomly placed, randomly yawed rectangular blocks -- bars (hurdle-like:
    length U(1.0, 2.0) m, thickness U(0.16, 0.32) m, height sampled from
    `eval(cfg.hurdle_height_range)`) and slabs (U(0.6, 1.2) m square, height
    `eval(cfg.step_height)`, a single deterministic value at this difficulty, mirroring
    `cfg.step_height`'s use as a fixed step rather than a sampled range in
    `extreme_parkour_terrians.py`) -- unioned onto the field by `max`, plus 8 (`num_goals`)
    sector anchors at the ring's mid-radius.

    No cfg mutation, no global RNG use: `rng` is any object exposing `.uniform(lo, hi)`
    (both `numpy.random` -- the module itself, for production, matching the ambient-global-
    state convention every existing terrain function relies on -- and a
    `numpy.random.RandomState` instance -- for reproducible tests -- provide this).

    Block placement is confined to the annulus by construction, not by post-hoc masking:
    each block's centre radius is sampled from
    `U(clear_radius + half_diag, ring_outer_radius - half_diag)`, where `half_diag` is
    that block's own circumscribing-circle radius -- so the closest point of any block to
    the tile centre is never less than `clear_radius`, and the furthest point never
    exceeds `ring_outer_radius`, regardless of the block's sampled yaw.
    """
    width_pixels, length_pixels = _tile_pixel_dims(cfg)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    cx = 0.5 * cfg.size[0]
    cy = 0.5 * cfg.size[1]

    # pixel-centre coordinates, metres, relative to the tile centre
    xs = (np.arange(width_pixels) + 0.5) * hs - cx
    ys = (np.arange(length_pixels) + 0.5) * hs - cy
    dx, dy = np.meshgrid(xs, ys, indexing="ij")

    r_clear = float(cfg.clear_radius)
    r_out = float(cfg.ring_outer_radius)
    n_blocks = int(getattr(cfg, "n_blocks", 0))
    bar_fraction = float(getattr(cfg, "bar_fraction", 0.5))
    n_bars = int(round(n_blocks * bar_fraction))

    hurdle_height_range_m = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
    step_height_m = eval(cfg.step_height, {"difficulty": difficulty})

    for i in range(n_blocks):
        is_bar = i < n_bars
        yaw = rng.uniform(0.0, np.pi)
        if is_bar:
            length = rng.uniform(1.0, 2.0)
            thickness = rng.uniform(0.16, 0.32)
            height_m = rng.uniform(hurdle_height_range_m[0], hurdle_height_range_m[1])
        else:
            length = rng.uniform(0.6, 1.2)
            thickness = length
            height_m = step_height_m

        half_diag = 0.5 * float(np.sqrt(length ** 2 + thickness ** 2))
        lo, hi = r_clear + half_diag, r_out - half_diag
        r_c = rng.uniform(lo, hi) if hi > lo else 0.5 * (lo + hi)
        theta = rng.uniform(0.0, 2.0 * np.pi)
        bx = r_c * np.cos(theta)
        by = r_c * np.sin(theta)

        mask = _oriented_box_mask(dx - bx, dy - by, yaw, length, thickness)
        height_px = round(height_m / vs)
        height_field_raw = np.where(mask, np.maximum(height_field_raw, height_px), height_field_raw)

    goals, goal_heights = _sector_anchor_goals(cfg, num_goals)

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * hs, goal_heights


def flat_height_field(difficulty: float, cfg, num_goals: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure body of `omni_flat_terrain`: an all-zero field (plus roughness noise) and the
    same `num_goals` sector anchors `ring_height_field` places, so every sub-terrain type
    in the omni generator exposes the identical anchor layout regardless of whether it
    carries blocks."""
    width_pixels, length_pixels = _tile_pixel_dims(cfg)
    height_field_raw = np.zeros((width_pixels, length_pixels))

    goals, goal_heights = _sector_anchor_goals(cfg, num_goals)

    height_field_raw = padding_height_field_raw(height_field_raw, cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights


@parkour_field_to_mesh_xy
def omni_ring_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniRingTerrainCfg", num_goals: int):
    return ring_height_field(np.random, difficulty, cfg, num_goals)


@parkour_field_to_mesh_xy
def omni_flat_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniFlatTerrainCfg", num_goals: int):
    return flat_height_field(difficulty, cfg, num_goals)

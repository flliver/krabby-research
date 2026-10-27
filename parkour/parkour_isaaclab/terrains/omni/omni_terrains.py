
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


# ============================================================================ omni FIELD
#
# Omni phase-2 plan, step 2 (2026-10-06): ``KRABBY_FLAT_TERRAIN_GEOM=field``. The ring keeps its
# blocks inside an annulus (2.0-4.5 m), so a robot that walks past 4.5 m -- or into a tile corner
# -- is on flat ground again, and a fast omni policy already spends ~51 % of its steps beyond the
# disc. The field drops the annulus: blocks are scattered uniformly over the whole tile square
# (to +-``ring_outer_radius`` = 4.80 m, i.e. 0.5 * 9.92 - ``field_margin``), outside a small clear
# spawn pad, so neighbouring field tiles read as ONE continuous field with ~0.4-0.5 m seams. A
# third kind, the PIT (a sunken slab), joins bars and slabs. Everything above this banner --
# ``ring_height_field`` in particular -- is untouched (golden-pinned byte-identical).

FIELD_KIND_BAR = 0
FIELD_KIND_SLAB = 1
FIELD_KIND_PIT = 2

#: Clear ground pixels kept between any PIT pixel and any RAISED (bar/slab) pixel, in every direction
#: including the diagonals (Chebyshev pixel distance >= gap + 1). 2026-10-06 (omni phase 2, pre-smoke
#: fix): with ramp edges (``slope_threshold`` 1.5 -> a wall only above 0.12 m = 24 vertical units) a
#: raised pixel next to a pit pixel steps block + pit + roughness, which crossed 0.12 m from d ~ 0.31
#: (0.13 m walls at d >= 0.42) -- ramp mode was not ramp-only. With the gap every step is
#: block-to-ground or ground-to-pit, each <= one block's own height + roughness. A stand-in cfg may
#: override it as ``pit_gap_px`` (tests; ``<= 0`` switches the gap off -- the pre-gap behaviour); the shipped cfg
#: does not carry the field.
FIELD_PIT_GAP_PX = 2
#: A pit's keep-out (every raised block grown by the gap, on top of the pad) is larger than any other block's, so
#: a pit gets this multiple of ``cfg.max_tries`` centre draws: at 1x, 1 % of dense tiles lost a pit
#: (10 / 1000 seeded tiles, d 0.2-1.0); at 4x, 0 / 1000. Bars and slabs keep ``max_tries``.
FIELD_PIT_TRIES_MULT = 4


def field_kind_counts(n_blocks: int, bar_fraction: float, pit_fraction: float) -> tuple[int, int, int]:
    """``(n_bar, n_slab, n_pit)`` for a tile: deterministic counts, not per-block coin flips.

    ``round()`` per kind with the slab count taking the remainder, exactly as the ring derives
    ``n_bars = round(n_blocks * bar_fraction)``: a per-block Bernoulli would add a kind-mix
    variance between tiles of the same difficulty that the terrain curriculum would then read
    as difficulty. Raises when the fractions leave no room (``bar + pit > 1``) or are negative.
    """
    n = int(n_blocks)
    bf, pf = float(bar_fraction), float(pit_fraction)
    if n < 0:
        raise ValueError(f"n_blocks must be non-negative, got {n_blocks}")
    if bf < 0.0 or pf < 0.0 or bf + pf > 1.0 + 1e-9:
        raise ValueError(f"bar_fraction={bar_fraction} / pit_fraction={pit_fraction} must be >= 0 and sum to <= 1")
    n_bar = int(round(n * bf))
    n_pit = min(int(round(n * pf)), n - n_bar)
    return n_bar, n - n_bar - n_pit, n_pit


def _yawed_half_extents(yaw: float, length: float, width: float) -> tuple[float, float]:
    """Half-extents (x, y), metres, of the axis-aligned bounding box of a ``length`` x ``width``
    box rotated by ``yaw`` -- the box's own footprint, so a centre drawn inside
    ``[-H + ex, H - ex]`` keeps every block pixel inside the ``+-H`` square whatever the yaw."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return 0.5 * (c * length + s * width), 0.5 * (s * length + c * width)


def _dilate_square(mask: np.ndarray, r: int) -> np.ndarray:
    """``mask`` grown by ``r`` pixels in every direction (a ``(2r+1) x (2r+1)`` square structuring
    element, i.e. Chebyshev distance <= r), clipped to the array -- pure numpy, no scipy import."""
    if r <= 0 or not mask.any():
        return mask.copy()
    w, ell = mask.shape
    padded = np.zeros((w + 2 * r, ell + 2 * r), dtype=bool)
    padded[r:r + w, r:r + ell] = mask
    out = np.zeros_like(mask)
    for i in range(2 * r + 1):
        for j in range(2 * r + 1):
            out |= padded[i:i + w, j:j + ell]
    return out


def field_layout(rng, difficulty: float, cfg) -> tuple[np.ndarray, np.ndarray, int]:
    """Pure, seed-injected body of the field's block placement (no padding, no roughness).

    Args:
        rng: anything with ``.uniform(lo, hi)`` -- ``numpy.random`` (the module, production, the
            same ambient-global convention as the ring) or a ``RandomState`` (tests).
        difficulty: the tile's difficulty (feeds the three height/depth formulas).
        cfg: an ``OmniFieldTerrainCfg`` (or a stand-in) whose ``size`` is the decorator's
            ALREADY-REDUCED tile size (9.92 m at 10 m tiles), as for every terrain body.

    Returns:
        ``(raw, occupancy, n_placed)``: ``raw`` the stamped heights in vertical-scale units
        (float, 0 off-block, > 0 bars/slabs, < 0 pits), ``occupancy`` the exact bool "a block
        or pit covers this pixel" mask (what the field-mode telemetry indexes instead of the
        ring's median + 0.03 m rule), ``n_placed`` the number of blocks actually stamped.

    Per block, in index order (bars, then slabs, then pits -- :func:`field_kind_counts`):

    1. draw yaw ``U(0, pi)`` and the kind's dimensions -- bars ``cfg.bar_length_range`` x
       ``cfg.bar_thickness_range`` (the ring's U(1.0, 2.0) x U(0.16, 0.32)), slabs a square of
       side ``cfg.slab_side_range`` (U(0.6, 1.2)), pits a sunken slab of the same distribution
       -- and its height: bars ``U(eval(hurdle_height_range))``, slabs ``eval(step_height)``,
       pits ``eval(pit_depth)`` (all three are the ring's recal2b2 strings + ``0.02 + 0.06*d``);
    2. draw the centre uniformly in ``[-H + ex, H - ex] x [-H + ey, H - ey]`` (``H`` =
       ``cfg.ring_outer_radius``, ``ex``/``ey`` the yawed box's half-extents);
    3. REJECT the centre if any pixel of the box has ``r < pad_r + hs`` (the clear pad plus one
       pixel, so the rasterised edge can never touch the pad) or -- only when ``cfg.min_gap > 0``
       -- if the box grown by ``min_gap`` on every side touches an already-stamped block, or
       (the default; off only for a stand-in ``pit_gap_px <= 0``) if it comes within
       :data:`FIELD_PIT_GAP_PX` pixels of a block of the OPPOSITE sign:
       a pit pixel needs >= 2 clear pixels (Chebyshev, diagonals included) to every raised pixel
       and a raised pixel the same to every pit pixel, so in ramp mode no raised-to-pit step
       exists to cross the 0.12 m wall trigger (2026-10-06); redraw the CENTRE only (yaw, dims and
       height are kept, so the size distribution is exactly the ring's rather than skewed towards
       blocks that fit more easily) up to ``cfg.max_tries`` times (a pit: :data:`FIELD_PIT_TRIES_MULT`
       times that), then give up on that block (counted as unplaced, never forced in);
    4. stamp it: raised blocks union by ``max``; a pit lowers only pixels that are not raised
       (``min`` over pixels with height <= 0), so a pit never cuts a raised block and a raised
       block stamped later over a pit wins -- the result is independent of the order the two
       kinds overlap in. (Since the pit gap, raised and pit pixels can no longer overlap or touch
       when placed through this function; the stamping rule is kept as the documented fallback.)

    RNG stream (the pit gap, 2026-10-06): bars and slabs are drawn before any pit (the kind order
    above), so no raised block can be rejected for the gap and every bar/slab draw -- yaw, dims,
    height, every centre try -- is the pre-gap stream exactly. A pit whose centre lands within the
    gap of a raised block now draws one more centre pair, so from that pit on the stream shifts:
    the later pits of the tile, and -- under the production ``numpy.random`` global -- the tile's
    roughness and every draw after it in the generator. Bar/slab placement of each field tile is
    unchanged only as long as the global stream reaching the tile is (the first field tile of a
    seeded generator; any tile when seeded per tile, as the tests do).
    """
    width_pixels, length_pixels = _tile_pixel_dims(cfg)
    raw = np.zeros((width_pixels, length_pixels))
    occupancy = np.zeros((width_pixels, length_pixels), dtype=bool)

    hs = cfg.horizontal_scale
    vs = cfg.vertical_scale
    cx = 0.5 * cfg.size[0]
    cy = 0.5 * cfg.size[1]
    xs = (np.arange(width_pixels) + 0.5) * hs - cx
    ys = (np.arange(length_pixels) + 0.5) * hs - cy
    dx, dy = np.meshgrid(xs, ys, indexing="ij")
    r_grid = np.sqrt(dx * dx + dy * dy)

    half = float(cfg.ring_outer_radius)
    margin = float(getattr(cfg, "field_margin", 0.0))
    if half > min(cx, cy) - margin + 1e-9:
        raise ValueError(
            f"field half-width ring_outer_radius={half} exceeds the tile's half-size "
            f"{min(cx, cy):.4f} minus field_margin={margin}"
        )
    pad_keep_out = float(cfg.clear_radius) + hs
    min_gap = float(getattr(cfg, "min_gap", 0.0))
    max_tries = int(getattr(cfg, "max_tries", 64))
    pit_gap = int(getattr(cfg, "pit_gap_px", FIELD_PIT_GAP_PX))
    pit_tries = max_tries * (FIELD_PIT_TRIES_MULT if pit_gap > 0 else 1)
    # pixels within ``pit_gap`` of a raised / pit pixel: where the opposite kind may not go
    near_raised = np.zeros((width_pixels, length_pixels), dtype=bool)
    near_pit = np.zeros((width_pixels, length_pixels), dtype=bool)

    n_bar, n_slab, n_pit = field_kind_counts(
        int(getattr(cfg, "n_blocks", 0)), float(getattr(cfg, "bar_fraction", 0.4)),
        float(getattr(cfg, "pit_fraction", 0.2)),
    )
    kinds = [FIELD_KIND_BAR] * n_bar + [FIELD_KIND_SLAB] * n_slab + [FIELD_KIND_PIT] * n_pit

    hurdle_lo, hurdle_hi = eval(cfg.hurdle_height_range, {"difficulty": difficulty})
    step_height_m = eval(cfg.step_height, {"difficulty": difficulty})
    pit_depth_m = eval(cfg.pit_depth, {"difficulty": difficulty})
    bar_len, bar_thk, slab_side = cfg.bar_length_range, cfg.bar_thickness_range, cfg.slab_side_range

    n_placed = 0
    for kind in kinds:
        yaw = rng.uniform(0.0, np.pi)
        if kind == FIELD_KIND_BAR:
            length = rng.uniform(bar_len[0], bar_len[1])
            width = rng.uniform(bar_thk[0], bar_thk[1])
            value_px = round(rng.uniform(hurdle_lo, hurdle_hi) / vs)
        else:
            length = rng.uniform(slab_side[0], slab_side[1])
            width = length
            value_px = round(step_height_m / vs) if kind == FIELD_KIND_SLAB else -round(pit_depth_m / vs)
        ex, ey = _yawed_half_extents(yaw, length, width)

        mask = None
        for _ in range(pit_tries if kind == FIELD_KIND_PIT else max_tries):
            bx = rng.uniform(-half + ex, half - ex) if half > ex else 0.0
            by = rng.uniform(-half + ey, half - ey) if half > ey else 0.0
            cand = _oriented_box_mask(dx - bx, dy - by, yaw, length, width)
            if not cand.any() or (r_grid[cand] < pad_keep_out).any():
                continue
            if pit_gap > 0 and (cand & (near_pit if value_px > 0 else near_raised)).any():
                continue
            if min_gap > 0.0 and occupancy.any():
                grown = _oriented_box_mask(dx - bx, dy - by, yaw, length + 2.0 * min_gap, width + 2.0 * min_gap)
                if (grown & occupancy).any():
                    continue
            mask = cand
            break
        if mask is None:
            continue

        if value_px > 0:
            raw = np.where(mask, np.maximum(raw, value_px), raw)
            if pit_gap > 0:
                near_raised |= _dilate_square(mask, pit_gap)
        else:
            raw = np.where(mask & (raw <= 0), np.minimum(raw, value_px), raw)
            if pit_gap > 0:
                near_pit |= _dilate_square(mask, pit_gap)
        occupancy |= mask
        n_placed += 1

    return raw, occupancy, n_placed


def field_height_field(
    rng, difficulty: float, cfg: "omni_terrains_cfg.OmniFieldTerrainCfg", num_goals: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure body of ``omni_field_terrain``: :func:`field_layout`, then padding + roughness
    exactly as :func:`ring_height_field` applies them, plus the same sector anchors.

    Returns FOUR values -- the ring's three plus the pre-roughness stamped block heights
    (``int16`` vertical-scale units, the raw body's own pixel frame, zero off-block). The
    decorator turns that 4th value into (a) the exact block/pit occupancy channel it returns as
    a 7th value and (b), only when ``cfg.vertical_edges`` is set, the riser field that forces
    vertical walls on block edges (``convert_height_field_to_mesh(riser_field=...)``). Roughness
    is added AFTER the copy is taken, so roughness can never mark a pixel as a block or raise a
    wall.
    """
    raw, _occupancy, _n_placed = field_layout(rng, difficulty, cfg)
    stamped = np.rint(raw).astype(np.int16)

    goals, goal_heights = _sector_anchor_goals(cfg, num_goals)

    height_field_raw = padding_height_field_raw(raw.copy(), cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)

    return height_field_raw, goals * cfg.horizontal_scale, goal_heights, stamped


@parkour_field_to_mesh_xy
def omni_field_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniFieldTerrainCfg", num_goals: int):
    return field_height_field(np.random, difficulty, cfg, num_goals)


@parkour_field_to_mesh_xy
def omni_ring_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniRingTerrainCfg", num_goals: int):
    return ring_height_field(np.random, difficulty, cfg, num_goals)


@parkour_field_to_mesh_xy
def omni_flat_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniFlatTerrainCfg", num_goals: int):
    return flat_height_field(difficulty, cfg, num_goals)

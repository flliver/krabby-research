# SPDX-License-Identifier: BSD-3-Clause
"""Pure-torch exposure, sector and promotion bookkeeping for the omni ring task (plan D9).

No isaaclab imports: everything here runs on CPU in
``tests/unit/test_crab_hex_omni_exposure.py``, and :mod:`omni_parkour_event` calls exactly
these functions -- the arithmetic the tests pin is the arithmetic that trains. The one
non-torch import is the forward exposure ledger
(``parkour_isaaclab.envs.mdp.parkours.exposure_stats``), which is itself pure torch; the
omni ledger WRAPS it rather than reimplementing it so the keys the sideways/omni driver
greps (``reach_edge_frac``, ``reach_obst_frac``, ``field_frac_mean``, ``goals_passed_mean``,
``obst_coverage_k`` and their ``_rsi`` twins, the failure/ep-length keys) keep their exact
names, grouping rules and ring semantics.

WHAT THE FORWARD KEYS MEAN ON A RING
------------------------------------
The corridor task measures exposure along ``+x``: the platform edge, then obstacle goals
downrange. The ring has no downrange -- the robot spawns at the tile centre inside a clear
disc of radius ``clear_radius`` with blocks scattered in the annulus out to
``ring_outer_radius``, and the command can point anywhere. The five ledger keys are
therefore redefined, and ONLY these definitions are in play for the omni task:

``reach_edge``
    the episode left the clear disc at least once (``r_max > clear_radius``), where ``r``
    is the distance from the TILE CENTRE. The corridor analogue is "got off the platform".
``reach_obst``
    at least one step with at least one foot over a raised block pixel (see
    :func:`block_mask_from_height_field`). "Committed to an obstacle", as in the corridor.
``goals_passed``
    the number of DISTINCT sectors visited outside the clear disc (0..``n_sectors``). A
    robot that pivots on the spot sweeps every bearing but never leaves the disc, so the
    "outside" qualifier is what stops spinning from counting as coverage.
``obst_coverage_k``
    sector ``k-1`` was visited outside the disc (so ``goals_passed == sum_k coverage_k``).
    The corridor's "reached obstacle k" becomes "reached sector k": both answer "which
    parts of the field did the population actually see".
``field_frac`` / ``field_steps``
    share of the episode's steps spent outside the clear disc (the corridor's "past the
    platform edge").

Five omni-only keys are appended (:data:`OMNI_EXTRA_KEYS`): ``track_ratio_mean``,
``track_ratio_move_mean``, ``heading_err_rms_mean``, ``block_steps_mean``, ``r_max_mean``.

A NOTE ABOUT ``reach_obst`` / ``block_steps_mean`` (measured, not hypothetical)
-------------------------------------------------------------------------------
The block mask is derived from the terrain height field alone -- there is no "this pixel is
a block" array -- as ``h - pad > block_height_tol_m`` with ``pad`` the tile's median height.
The ring tiles also carry ``random_uniform_terrain`` roughness, ADDED on top of the stamped
blocks. Measured on the real decorated ring function at the real (SHIPPED, see the caveat
below) cfg, d in {0.05, 0.2, 0.7}:

======================  ====================  ============================  ==============
window                  roughness (metres)    pixels above median + 0.03 m  block height
======================  ====================  ============================  ==============
1a (light, d 0.05-0.2)  -0.030 .. +0.025      0 %                           0.0225-0.042 m
2a+ (default, d 0.7)    -0.030 .. +0.025      0 %                           0.055-0.072 m
======================  ====================  ============================  ==============

The roughness band is IDENTICAL at every difficulty (the shipped ``noise_range`` does not
scale with ``d`` -- see the caveat), and at the shipped ``block_height_tol_m = 0.03`` (plan D9,
decided 2026-09-16) zero pixels rise above the tolerance from roughness alone: no false
positives at any window, not just 1a. The residual bias runs the OTHER way instead --
under-reporting at 1a, whose lowest blocks (2.25-3.3 cm at d 0.05-0.2) sit AT the 0.03 m
tolerance, so the mask can miss the very lightest 1a obstacles rather than saturate on
roughness. **Neither key is gated on** -- they are watched-only telemetry (the driver's gates
read the eval harness) at every difficulty.

CAVEAT -- this is the SHIPPED preset, not the class default: ``OmniRingTerrainCfg``'s own
``noise_range`` default is ``(0.02, 0.06)`` (the wider forward-terrain literal), and NO plan-D9
preset ever leaves it there. ``_apply_crab_hex_easy_mixed_terrain`` -- the shared knob loop
every omni window runs through (``apply_flat_walk_knobs``, shared with the forward/sideways
tasks) -- overwrites EVERY sub-terrain's ``noise_range`` to ``(0.02, 0.02)`` at 1a AND at every
2a+ window alike (confirmed in the ``cfg_omni_1a.json`` / ``cfg_omni_2a.json`` config dumps);
the class default above is never what actually ships. A future preset that skips that knob
loop, or a class-level default change, would restore the wider (and difficulty-scaling) band
and this table would need re-measuring.

FIELD MODE (omni phase-2 plan, step 2, 2026-10-06)
--------------------------------------------------
Under ``KRABBY_FLAT_TERRAIN_GEOM=field`` the course is a clear spawn PAD (1.7 m) inside a field
of bars, slabs and pits that runs to within 0.16 m of every tile edge, so neighbouring field tiles
form one field. The parkour term detects this from the generator's sub-terrain CLASSES and then,
and only then, changes three things (ring-mode telemetry and its key list are untouched):

* the keys keep their names, but ``clear_radius`` is the PAD (the knob layer syncs the term's
  radius to it): ``reach_edge`` = left the pad, ``field_frac`` and the sector coverage are
  measured beyond the pad;
* "foot over a block" uses the generator's EXACT occupancy channel (``block_masks``, pits
  included) instead of the median + 0.03 m rule above, and indexes it with
  :func:`tiled_pixel_index`, so a foot on a NEIGHBOURING tile reads that tile's blocks (the
  ring's clamp-into-own-tile lookup would read the own tile's empty border instead);
* two field-only keys are appended (:data:`OMNI_FIELD_KEYS`): ``off_tile_frac_mean`` and
  ``off_tile_block_steps_mean``.

OBSTACLE-TILE CURRICULUM TELEMETRY (omni phase-2 CHARTER 8 / 5, 2026-10-06; ring AND field)
--------------------------------------------------------------------------------------------
The phase-2 live reads judge the terrain curriculum on OBSTACLE tiles only (``name != parkour_flat``,
the parkour term's ``is_obst_tile``): an all-tile ``terrain_levels`` mixes in the flat tiles' levels
and is never accepted in their place. Four keys, logged as ``Metrics/base_parkour/<key>`` on every
omni run, ring and field alike (telemetry only: no cfg field, so no config dump changes):

``terrain_levels_obst`` (:data:`OMNI_TERRAIN_WINDOW_KEYS`)
    mean ``terrain_levels`` over the envs whose CURRENT tile is an obstacle tile, read at the reset
    call before any tile moves (the same instant the base term's all-tile ``terrain_levels`` is
    read). Left out of that reset's log when no env is on an obstacle tile (an all-flat run).
``promote_count`` / ``demote_count`` (:data:`OMNI_TERRAIN_WINDOW_KEYS`)
    the number of ``promotion_mask`` up / down decisions APPLIED in that reset call to episodes that
    ended on an obstacle tile (``freeze_terrain_levels`` applies none, so 0). Per reset call, i.e.
    per env step with resets; rsl_rl then averages the reset calls of an iteration, so the logged
    value is "obstacle-tile promotions per reset step" and is > 0 iff any happened.
``track_ratio_obst_p50`` (:data:`OMNI_OBST_RATIO_KEY`)
    the MEDIAN of the per-episode ``ep_track_dist / ep_cmd_dist`` -- exactly the ratio
    :func:`promotion_mask` gates on -- over the obstacle-tile episodes in the ledger's logging window
    (the last ``exposure_window`` such episodes, the same rolling window every ledger mean uses),
    with the promotion rule's own exclusion (``ep_cmd_dist < min_cmd_dist_m`` holds, so it is left
    out) and non-finite ratios dropped. ABSENT (never 0.0, never NaN) until the window holds one
    episode, exactly like :data:`OMNI_SPAWN_SPLIT_KEYS`. A per-reset-call median would be a median
    of the 1-5 episodes one step ends, and its iteration mean would be a MEAN in all but name.
"""
from __future__ import annotations

import math

import torch

from parkour_isaaclab.envs.mdp.parkours.exposure_stats import (
    SPAWN_PLATFORM,
    SPAWN_RSI,
    ExposureLedger,
    RingMean,
)

__all__ = [
    "SPAWN_PLATFORM",
    "SPAWN_RSI",
    "OMNI_EXTRA_KEYS",
    "OMNI_SPAWN_SPLIT_KEYS",
    "BLOCK_PAD_QUANTILE",
    "OmniExposureLedger",
    "block_mask_from_height_field",
    "episode_exposure_omni",
    "in_clear_disc",
    "note_spawn_xy_to_parkour",
    "promotion_mask",
    "radius_xy",
    "sector_index",
    "tile_pixel_index",
    "global_pixel_index",
    # field mode (omni phase-2 plan step 2)
    "OMNI_FIELD_KEYS",
    "tiled_pixel_index",
    "global_block_mask",
    "feet_block_flags_tiled",
    # obstacle-tile curriculum telemetry (omni phase-2 CHARTER 8 / 5)
    "OMNI_OBST_RATIO_KEY",
    "OMNI_LAZY_KEYS",
    "OMNI_TERRAIN_WINDOW_KEYS",
    "obstacle_level_mean",
    "obstacle_promotion_counts",
    "terrain_window_extras",
]

TWO_PI = 2.0 * math.pi

OMNI_EXTRA_KEYS = (
    "track_ratio_mean",
    "track_ratio_move_mean",
    "heading_err_rms_mean",
    "block_steps_mean",
    "r_max_mean",
)
"""Omni-only ledger keys, appended to the forward ``ExposureLedger`` key list.

Grouping (stated once, because a mixed rule would be a silent trap):

* ``track_ratio_mean`` -- mean of ``ep_track_dist / ep_cmd_dist`` over episodes whose
  commanded distance reached ``min_cmd_dist_m`` (the same episodes the promotion rule
  judges; standing-dominated episodes are excluded, not counted as zero).
* ``track_ratio_move_mean`` -- the SAME quantity accumulated over ``SLOT_MOVE`` steps only
  (moving, not turning), with the same ``min_cmd_dist_m`` validity rule applied to its own
  commanded distance. WATCHED, never gated on: ``track_ratio_mean`` keeps driving promotion
  and keeps its meaning. It exists because the unrestricted ratio is not comparable across
  windows -- a ``move_turn`` row's body command swings away from the travel direction as the
  commanded turn executes (the "clamp-only re-cap"), so the denominator keeps growing while
  the robot is correctly turning, and the ratio drops when a window merely ENABLES turning.
  This key is emitted on every omni run, pooled or legacy, so the two are comparable.
* ``heading_err_rms_mean`` -- mean over ALL finished episodes of the per-episode RMS of
  ``wrap(heading_target - yaw)``.
* ``r_max_mean`` -- mean over ALL finished episodes of the furthest distance reached from
  the tile centre, metres.
* ``block_steps_mean`` -- mean over OBSTACLE-TILE episodes of the number of steps with a
  foot over a block pixel (identically zero on flat tiles, so grouping it with
  ``field_frac_mean``'s obstacle-tile rule keeps it independent of the flat proportion).
"""

OMNI_SPAWN_SPLIT_KEYS = (
    "track_ratio_mean_rsi",
    "track_ratio_mean_norsi",
    "track_ratio_move_mean_rsi",
    "track_ratio_move_mean_norsi",
)
"""The two tracking ratios split by SPAWN KIND (PPO-construction plan step 2, 2026-10-01).

Each is its parent key's mean (same validity rule) over the episodes that STARTED from an RSI
frame (``spawn_kind == SPAWN_RSI``) or from anything else, on ALL terrain -- flat and obstacle
tiles alike. That is deliberately NOT the forward ledger's ``*_rsi`` convention (obstacle-tile RSI
episodes only): the question these answer is "does a seeded fast gait keep tracking", which a
flat-tile restriction would mostly throw away.

EMPTY GROUPS ARE NOT REPORTED. :meth:`OmniExposureLedger.means` leaves a key out until its ring has
received an episode (so a run without RSI never shows ``*_rsi``), and the omni parkour term only
creates the metric when the key first appears -- never a 0.0 that would read as "tracks nothing".
They are kept OUT of :data:`OMNI_EXTRA_KEYS` / ``emit_keys`` for that reason: those keys are
emitted from the first iteration on, as 0.0 while empty.
"""

OMNI_FIELD_KEYS = (
    "off_tile_frac_mean",
    "off_tile_block_steps_mean",
)
"""Field-mode-only ledger keys (omni phase-2 plan step 2, 2026-10-06); never emitted on the ring.

* ``off_tile_frac_mean`` -- mean over ALL finished episodes of the share of the episode's steps on
  which at least one foot stood outside the episode's own tile (on a neighbouring tile, or off
  the terrain). WATCHED: it says how much of the "one continuous field" the population actually
  uses across seams.
* ``off_tile_block_steps_mean`` -- mean over ALL finished episodes of the number of steps with a
  foot over a block or pit of a NEIGHBOURING tile (the plan's S4 "off-tile block steps > 0").

Both are grouped over all episodes, not the obstacle-tile rule of ``block_steps_mean``: a robot on
a flat tile next to a field tile can reach that tile's blocks, so a flat-tile episode can score.
"""

OMNI_OBST_RATIO_KEY = "track_ratio_obst_p50"
"""Median per-episode promotion ratio over the ledger window's obstacle-tile episodes (module
docstring, OBSTACLE-TILE CURRICULUM TELEMETRY). The phase-2 calibration's r-bar is read from it."""

OMNI_LAZY_KEYS = OMNI_SPAWN_SPLIT_KEYS + (OMNI_OBST_RATIO_KEY,)
"""Ledger keys reported only once their group is non-empty (absent, never 0.0 / NaN, before that);
the parkour term creates each metric when its key first appears in :meth:`OmniExposureLedger.means`."""

OMNI_TERRAIN_WINDOW_KEYS = ("terrain_levels_obst", "promote_count", "demote_count")
"""Per-reset-call keys the parkour term adds to its ``reset`` log (module docstring). The names are
the EXACT tags the phase-2 live reads accept (``p2_live.py`` OBST_LEVEL_TAGS / PROMO_TAGS)."""

BLOCK_PAD_QUANTILE = 0.5
"""Per-tile quantile of the height field taken as the tile's "pad" (ground) level.

The median is the right default: blocks cover a minority of a ring tile's area and the
roughness is roughly symmetric about the pad. Exposed as a module constant rather than a
cfg field so a recalibration is a one-line, test-covered change."""


# --------------------------------------------------------------------------- geometry


def radius_xy(dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """Planar distance from the tile centre, metres. ``dx``/``dy`` are centre-relative."""
    return torch.sqrt(dx * dx + dy * dy)


def in_clear_disc(dx: torch.Tensor, dy: torch.Tensor, clear_radius: float) -> torch.Tensor:
    """True where the centre-relative point is INSIDE the clear disc (``r <= R``).

    The boundary belongs to the disc, so "left the disc" is the strict complement
    ``r > R`` -- the definition ``reach_edge`` uses. A robot standing exactly at the
    spawn (``r = 0``) is inside for every positive radius.
    """
    if not (clear_radius >= 0.0 and math.isfinite(clear_radius)):
        raise ValueError(f"clear_radius must be finite and non-negative, got {clear_radius}")
    return radius_xy(dx, dy) <= clear_radius


def sector_index(dx: torch.Tensor, dy: torch.Tensor, n_sectors: int) -> torch.Tensor:
    """Sector of a centre-relative bearing, ``int64`` in ``[0, n_sectors)``.

    Sectors are CENTRED on the ring's goal anchors, which sit at bearings
    ``k * 2*pi / n_sectors`` from world ``+x`` (plan D8: "goals = 8 sector anchors"): sector
    ``k`` covers ``[k*2pi/n - pi/n, k*2pi/n + pi/n)``. Centring rather than starting the
    sectors on the anchors matters because a robot commanded straight at anchor ``k`` walks
    along the sector's middle instead of tightrope-walking a boundary, so
    ``goals_passed`` counts one sector per radial excursion instead of two.

    Sector 0 therefore contains world ``+x``; the index increases counter-clockwise.
    ``dx = dy = 0`` gives sector 0 (``atan2(0, 0) == 0``) -- harmless, because sectors are
    only credited outside the clear disc.
    """
    n = int(n_sectors)
    if n <= 0:
        raise ValueError(f"n_sectors must be positive, got {n_sectors}")
    width = TWO_PI / n
    ang = torch.atan2(dy, dx)
    shifted = torch.remainder(ang + 0.5 * width, TWO_PI)
    return torch.clamp((shifted / width).floor().long(), 0, n - 1)


# ------------------------------------------------------------------ height-field lookup


def block_mask_from_height_field(
    hf_global: torch.Tensor,
    tile_px: tuple[int, int],
    tol_px: float,
    quantile: float = BLOCK_PAD_QUANTILE,
) -> torch.Tensor:
    """Per-pixel "this is a raised block" mask over the whole terrain.

    Args:
        hf_global: ``(rows*W, cols*L)`` global height field in RAW int16 units (the
            assembly :func:`exposure_stats.global_height_field` produces, i.e.
            ``global[row*W + a, col*L + b] == height_fields[row, col, a, b]``).
        tile_px: ``(W, L)`` pixels per tile, i.e. ``height_fields.shape[2:]``.
        tol_px: Height above the tile's pad level, in the SAME raw units, above which a
            pixel counts as a block (``block_height_tol_m / vertical_scale``).
        quantile: Per-tile quantile taken as the pad level (see :data:`BLOCK_PAD_QUANTILE`).

    Returns:
        ``(rows*W, cols*L)`` bool mask, indexed exactly like ``hf_global``.

    The pad is per TILE, not global, because tiles differ in difficulty (and therefore in
    roughness amplitude) and a global threshold would mark every pixel of a high tile.
    """
    if hf_global.dim() != 2:
        raise ValueError(f"hf_global must be 2-D, got shape {tuple(hf_global.shape)}")
    w, ell = int(tile_px[0]), int(tile_px[1])
    if w <= 0 or ell <= 0:
        raise ValueError(f"tile_px must be positive, got {tile_px}")
    g0, g1 = int(hf_global.shape[0]), int(hf_global.shape[1])
    if g0 % w or g1 % ell:
        raise ValueError(f"global field {(g0, g1)} is not a whole number of {(w, ell)} tiles")
    rows, cols = g0 // w, g1 // ell
    h = hf_global.reshape(rows, w, cols, ell).float()
    flat = h.permute(0, 2, 1, 3).reshape(rows, cols, w * ell)
    if abs(float(quantile) - 0.5) < 1e-12:
        pad = flat.median(dim=-1).values
    else:
        pad = torch.quantile(flat, float(quantile), dim=-1)
    mask = h > (pad[:, None, :, None] + float(tol_px))
    return mask.reshape(g0, g1)


def tile_pixel_index(
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    tile_centre_w: torch.Tensor,
    size_xy: tuple[float, float],
    horizontal_scale: float,
    tile_px: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """World xy -> TILE-LOCAL height-field pixel, clamped into the tile.

    The height field of a tile has ``W = size_x / hs + 1`` pixels spanning ``size_x``
    metres (``parkour_field_to_mesh`` builds mesh vertices at ``i * hs``), so pixel ``i``
    sits at tile-local ``i * hs`` and the tile centre is pixel ``(W - 1) / 2``.

    This deliberately does NOT use ``exposure_stats.world_to_pixel``, whose global formula
    ``(x_w + size_x*num_rows/2) / hs`` assumes ``size_x / hs`` pixels per tile while the
    array stores ``size_x / hs + 1``: that convention (shared with ``reward_feet_edge``)
    drifts by one pixel per tile row, up to 10 px = 0.8 m across a 10-row terrain. A 0.8 m
    mis-registration would silently scramble a block lookup whose features are 0.16-1.2 m
    wide, so the omni term indexes per tile, where the mapping is exact.

    Args:
        x_w, y_w: World coordinates, shape ``(n,)`` (or ``(n, k)``).
        tile_centre_w: ``(n, 2)`` world xy of each row's tile centre (``env_origins``).
        size_xy: ``(size_x, size_y)`` of a tile, metres.
        horizontal_scale: metres per pixel.
        tile_px: ``(W, L)`` pixels per tile.

    Returns:
        ``(ix, iy)`` int64 pixel indices in ``[0, W)`` / ``[0, L)``, shape of ``x_w``.
    """
    hs = float(horizontal_scale)
    if not hs > 0.0:
        raise ValueError(f"horizontal_scale must be positive, got {horizontal_scale}")
    w, ell = int(tile_px[0]), int(tile_px[1])
    cx = tile_centre_w[..., 0] - 0.5 * float(size_xy[0])
    cy = tile_centre_w[..., 1] - 0.5 * float(size_xy[1])
    while cx.dim() < x_w.dim():
        cx = cx.unsqueeze(-1)
        cy = cy.unsqueeze(-1)
    ix = torch.clamp(((x_w - cx) / hs).round().long(), 0, w - 1)
    iy = torch.clamp(((y_w - cy) / hs).round().long(), 0, ell - 1)
    return ix, iy


def global_pixel_index(
    row: torch.Tensor, col: torch.Tensor, ix: torch.Tensor, iy: torch.Tensor, tile_px: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tile index + tile-local pixel -> index into the global field assembled by
    :func:`exposure_stats.global_height_field` (``row*W + ix``, ``col*L + iy``)."""
    w, ell = int(tile_px[0]), int(tile_px[1])
    r = row
    c = col
    while r.dim() < ix.dim():
        r = r.unsqueeze(-1)
        c = c.unsqueeze(-1)
    return r * w + ix, c * ell + iy


def tiled_pixel_index(
    x_w: torch.Tensor,
    y_w: torch.Tensor,
    tile_centre_w: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    size_xy: tuple[float, float],
    horizontal_scale: float,
    tile_px: tuple[int, int],
    grid_rc: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World xy -> (tile row, tile col, tile-local pixel) of whichever tile the point is ON.

    :func:`tile_pixel_index` clamps every point into the episode's OWN tile, which is right for a
    ring (blocks stop at 4.5 m; the own tile's flat border is what a foot past it stands near) and
    wrong for the field, whose blocks run to 0.16 m from every edge: a foot across the seam would
    read the own tile's empty border pixel instead of the neighbour's block. Here the tile offset is
    ``floor((x - own corner) / size)`` per axis (0 = own tile, +-1 = the neighbour across a seam),
    and the pixel is taken in THAT tile's frame -- the same ``i * hs`` mapping as
    :func:`tile_pixel_index`, so on the own tile the two agree exactly.

    Tiles are ``size / hs + 1`` pixels wide but placed ``size`` apart, so a neighbour's pixel 0
    and the own tile's last pixel sit at the same world coordinate (the stored-126 / spaced-125
    convention); the floor makes the seam point belong to the neighbour.

    Args:
        x_w, y_w: world coordinates, ``(n,)`` or ``(n, k)``.
        tile_centre_w: ``(n, 2)`` world xy of each row's own tile centre (``env_origins``).
        row, col: ``(n,)`` own tile indices (``terrain_levels`` / ``terrain_types``).
        size_xy, horizontal_scale, tile_px: as for :func:`tile_pixel_index`.
        grid_rc: ``(num_rows, num_cols)`` of the terrain grid.

    Returns:
        ``(row', col', ix, iy, on_terrain, off_tile)`` shaped like ``x_w``: the tile actually
        under the point (clamped into the grid so it can always index), its pixel, whether that
        tile EXISTS (False past the outermost tiles -- callers treat it as "no block"), and whether
        it is not the own tile.
    """
    hs = float(horizontal_scale)
    if not hs > 0.0:
        raise ValueError(f"horizontal_scale must be positive, got {horizontal_scale}")
    w, ell = int(tile_px[0]), int(tile_px[1])
    n_rows, n_cols = int(grid_rc[0]), int(grid_rc[1])
    sx, sy = float(size_xy[0]), float(size_xy[1])
    cx = tile_centre_w[..., 0] - 0.5 * sx
    cy = tile_centre_w[..., 1] - 0.5 * sy
    r, c = row.long(), col.long()
    while cx.dim() < x_w.dim():
        cx, cy = cx.unsqueeze(-1), cy.unsqueeze(-1)
        r, c = r.unsqueeze(-1), c.unsqueeze(-1)
    lx, ly = x_w - cx, y_w - cy
    ox = torch.floor(lx / sx).long()
    oy = torch.floor(ly / sy).long()
    nr, nc = r + ox, c + oy
    on_terrain = (nr >= 0) & (nr < n_rows) & (nc >= 0) & (nc < n_cols)
    off_tile = (ox != 0) | (oy != 0)
    ix = torch.clamp(((lx - ox.to(lx.dtype) * sx) / hs).round().long(), 0, w - 1)
    iy = torch.clamp(((ly - oy.to(ly.dtype) * sy) / hs).round().long(), 0, ell - 1)
    return nr.clamp(0, n_rows - 1), nc.clamp(0, n_cols - 1), ix, iy, on_terrain, off_tile


def global_block_mask(block_masks: torch.Tensor | None, grid_shape: tuple[int, int, int, int]) -> torch.Tensor:
    """The generator's per-tile exact occupancy ``(rows, cols, W, L)`` -> the global
    ``(rows*W, cols*L)`` bool mask, assembled exactly like ``exposure_stats.global_height_field``
    (so :func:`global_pixel_index` indexes it). ``None`` (no tile returned a mask -- e.g. an
    all-flat field run) gives an all-False mask of ``grid_shape``."""
    r, c, w, ell = (int(v) for v in grid_shape)
    if block_masks is None:
        return torch.zeros(r * w, c * ell, dtype=torch.bool)
    bm = torch.as_tensor(block_masks).bool()
    if tuple(bm.shape) != (r, c, w, ell):
        raise ValueError(f"block_masks shape {tuple(bm.shape)} != height-field grid {(r, c, w, ell)}")
    return bm.permute(0, 2, 1, 3).reshape(r * w, c * ell)


def feet_block_flags_tiled(
    feet_x: torch.Tensor,
    feet_y: torch.Tensor,
    tile_centre_w: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    block_mask_global: torch.Tensor,
    size_xy: tuple[float, float],
    horizontal_scale: float,
    tile_px: tuple[int, int],
    grid_rc: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Field-mode per-step foot flags, ``(n,)`` bool each (``feet_*`` are ``(n, F)`` world xy):

    ``(over_block, over_block_off_tile, off_tile)`` -- at least one foot over a block/pit pixel
    of the tile it is ON (own or neighbour), at least one foot over a block/pit of a NEIGHBOURING
    tile, at least one foot not on the own tile. A foot past the outermost tiles is off-tile and
    over no block.
    """
    nr, nc, ix, iy, on_terrain, off_tile = tiled_pixel_index(
        feet_x, feet_y, tile_centre_w, row, col, size_xy, horizontal_scale, tile_px, grid_rc
    )
    w, ell = int(tile_px[0]), int(tile_px[1])
    over = block_mask_global[nr * w + ix, nc * ell + iy] & on_terrain
    return over.any(dim=-1), (over & off_tile).any(dim=-1), off_tile.any(dim=-1)


# ------------------------------------------------------------------------- promotion


def promotion_mask(
    track: torch.Tensor,
    cmd: torch.Tensor,
    move_up_frac: float,
    move_down_frac: float,
    min_cmd_dist: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Terrain-level promotion / demotion from the episode's TRACKING RATIO.

    The corridor term promotes on ``dis_to_start > frac * cmd_vx * episode_length_s``,
    which is <= 0 for any lateral or backward command (``cmd_vx <= 0``) and so can never
    promote an omni episode -- and would demote every one of them. The omni rule is the
    ratio of the distance actually travelled ALONG the command to the distance commanded:

        ``ratio = ep_track_dist / ep_cmd_dist``
        ``ep_track_dist += clamp(v_b . c_hat, 0) * dt``   (progress along the command only)
        ``ep_cmd_dist   += ||cmd_xy|| * dt``

    Both integrals are in metres, so the ratio is dimensionless and UN-SCALED by the
    horizon: the same 0.45/0.25 thresholds apply at any episode length or resampling rate,
    unlike the corridor's distance threshold (which had to be rescaled whenever the horizon
    moved -- see ``promote_fracs_for_horizon`` in the morph campaign).

    Episodes that were barely commanded to move (``ep_cmd_dist < min_cmd_dist``) HOLD: a
    stand-dominated episode has a meaningless ratio (0/0 or a large ratio from a metre of
    drift), and letting it vote would make the curriculum a function of the slot mix.

    Args:
        track: ``(n,)`` metres travelled along the command.
        cmd: ``(n,)`` metres commanded.
        move_up_frac: promote when ``ratio > move_up_frac`` (strict).
        move_down_frac: demote when ``ratio < move_down_frac`` (strict).
        min_cmd_dist: metres of commanded distance below which the episode holds.

    Returns:
        ``(up_mask, down_mask)`` bool tensors. They are mutually exclusive whenever
        ``move_down_frac <= move_up_frac``.
    """
    if track.shape != cmd.shape:
        raise ValueError(f"track/cmd shape mismatch: {tuple(track.shape)} != {tuple(cmd.shape)}")
    up_f, down_f = float(move_up_frac), float(move_down_frac)
    active = cmd >= float(min_cmd_dist)
    ratio = track / torch.clamp(cmd, min=1e-6)
    return active & (ratio > up_f), active & (ratio < down_f)


def tracking_ratio(track: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
    """``track / cmd`` with the same guard :func:`promotion_mask` uses (telemetry)."""
    return track / torch.clamp(cmd, min=1e-6)


# ------------------------------------------------------- obstacle-tile curriculum telemetry


def obstacle_level_mean(levels: torch.Tensor, is_obst: torch.Tensor) -> float | None:
    """Mean terrain level over the obstacle-tile envs; ``None`` when there is none (the key is then
    left out of the log rather than reported as 0.0 or NaN)."""
    m = is_obst.reshape(-1).bool()
    if not bool(m.any()):
        return None
    return float(levels.reshape(-1)[m.to(levels.device)].float().mean())


def obstacle_promotion_counts(
    move_up: torch.Tensor, move_down: torch.Tensor, is_obst: torch.Tensor
) -> tuple[int, int]:
    """``(promotions, demotions)`` among the episodes that ended on an obstacle tile.

    All three are aligned ``(n,)`` masks over the resetting envs; ``is_obst`` is the tile the episode
    ENDED on (the parkour term reads it before the tile moves)."""
    if not (move_up.shape == move_down.shape == is_obst.shape):
        raise ValueError(
            f"shape mismatch: up {tuple(move_up.shape)}, down {tuple(move_down.shape)}, obst {tuple(is_obst.shape)}"
        )
    obst = is_obst.bool().to(move_up.device)
    return int((move_up.bool() & obst).sum()), int((move_down.bool() & obst).sum())


def terrain_window_extras(level_obst: float | None, promote: int, demote: int) -> dict[str, float]:
    """The :data:`OMNI_TERRAIN_WINDOW_KEYS` entries of one reset call's log (unprefixed: the command
    manager adds ``Metrics/<term>/``). A ``None`` or non-finite level is left out."""
    out = {"promote_count": float(promote), "demote_count": float(demote)}
    if level_obst is not None and math.isfinite(level_obst):
        out["terrain_levels_obst"] = float(level_obst)
    return out


# --------------------------------------------------------------------- episode record


def episode_exposure_omni(
    r_max: torch.Tensor,
    sector_visited: torch.Tensor,
    block_steps: torch.Tensor,
    field_steps: torch.Tensor,
    ep_steps: torch.Tensor,
    clear_radius: float,
) -> dict[str, torch.Tensor]:
    """Per-episode exposure quantities under the omni definitions (module docstring).

    The returned dict has EXACTLY the keys ``ExposureLedger.push_episodes`` consumes, so
    the omni term reuses the forward ledger's grouping rules verbatim.

    Args:
        r_max: ``(n,)`` furthest distance from the tile centre reached, metres.
        sector_visited: ``(n, n_sectors)`` bool, sector visited outside the clear disc.
        block_steps: ``(n,)`` steps with a foot over a block pixel.
        field_steps: ``(n,)`` steps spent outside the clear disc.
        ep_steps: ``(n,)`` episode length in env steps.
        clear_radius: metres.
    """
    if sector_visited.dim() != 2 or sector_visited.shape[0] != r_max.shape[0]:
        raise ValueError("sector_visited must be (n, n_sectors) aligned with r_max")
    n_sect = int(sector_visited.shape[1])
    visited = sector_visited.float()
    out = {
        "reach_edge": (r_max > float(clear_radius)).float(),
        "reach_obst": (block_steps > 0).float(),
        "goals_passed": visited.sum(dim=1),
        "field_steps": field_steps.float(),
        "field_frac": field_steps.float() / ep_steps.clamp(min=1).float(),
    }
    for k in range(1, n_sect + 1):
        out[f"obst_coverage_{k}"] = visited[:, k - 1]
    return out


class OmniExposureLedger:
    """The forward :class:`ExposureLedger` plus the five omni-only rings.

    Keeping the forward ledger as a member (rather than subclassing and duplicating
    ``ring_keys``) means the shared keys can never drift from the corridor task's, which is
    what lets the same driver parse both campaigns' telemetry.
    """

    def __init__(
        self, n_sectors: int, capacity: int = 1024, device: str | torch.device = "cpu", field_mode: bool = False
    ):
        self.n_sectors = int(n_sectors)
        self.base = ExposureLedger(self.n_sectors, capacity=capacity, device=device)
        self._extra: dict[str, RingMean] = {k: RingMean(capacity, device) for k in OMNI_EXTRA_KEYS}
        self._split: dict[str, RingMean] = {k: RingMean(capacity, device) for k in OMNI_SPAWN_SPLIT_KEYS}
        # Field mode only (omni phase-2 step 2): the OMNI_FIELD_KEYS rings. Empty dict on the ring,
        # so a ring ledger's keys and means are exactly what they were.
        self.field_mode = bool(field_mode)
        self._field: dict[str, RingMean] = (
            {k: RingMean(capacity, device) for k in OMNI_FIELD_KEYS} if self.field_mode else {}
        )
        # OMNI_OBST_RATIO_KEY: the per-episode promotion ratios of the window's obstacle-tile episodes
        # (a RingMean is used only as the ring buffer; its median is taken in means()).
        self._obst_ratio = RingMean(capacity, device)

    #: keys reported only once their group is non-empty (see OMNI_LAZY_KEYS)
    lazy_keys = OMNI_LAZY_KEYS

    @staticmethod
    def emit_keys(n_sectors: int, field_mode: bool = False) -> list[str]:
        """Every key :meth:`means` reports (forward keys first, then the omni five, then -- field
        mode only -- :data:`OMNI_FIELD_KEYS`)."""
        keys = ExposureLedger.emit_keys(int(n_sectors)) + list(OMNI_EXTRA_KEYS)
        return keys + list(OMNI_FIELD_KEYS) if field_mode else keys

    def push_episodes(
        self,
        per_ep: dict[str, torch.Tensor],
        spawn_kind: torch.Tensor,
        is_obst: torch.Tensor,
        failed: torch.Tensor,
        ep_steps: torch.Tensor,
        track_ratio: torch.Tensor,
        track_ratio_valid: torch.Tensor,
        heading_err_rms: torch.Tensor,
        block_steps: torch.Tensor,
        r_max: torch.Tensor,
        track_ratio_move: torch.Tensor | None = None,
        track_ratio_move_valid: torch.Tensor | None = None,
        off_tile_frac: torch.Tensor | None = None,
        off_tile_block_steps: torch.Tensor | None = None,
    ) -> None:
        """Push one batch of finished episodes (see :data:`OMNI_EXTRA_KEYS` for grouping).

        ``track_ratio_move`` / ``track_ratio_move_valid`` are OPTIONAL: left out (or ``None``),
        nothing is pushed and ``track_ratio_move_mean`` stays an empty group -- which the
        parkour term's ``_update_metrics`` emits as 0.0, exactly as it does for any other ring
        before its first episode. Keeping them optional means a caller that does not measure
        the slot-gated pair (a test double, an older term) is not forced to fabricate one, and
        a fabricated zero would be indistinguishable from a policy that tracks nothing.
        ``track_ratio_move`` without a validity mask counts every episode.

        ``off_tile_frac`` / ``off_tile_block_steps`` (field mode): pushed over ALL episodes into the
        :data:`OMNI_FIELD_KEYS` rings; ignored by a ring-mode ledger (it has none).
        """
        self.base.push_episodes(per_ep, spawn_kind, is_obst, failed, ep_steps)
        rsi = spawn_kind == SPAWN_RSI
        valid = track_ratio_valid.bool()
        self._extra["track_ratio_mean"].push(track_ratio[valid])
        # OMNI_OBST_RATIO_KEY: the promotion rule's own population (valid = ep_cmd_dist >= min_cmd_dist)
        # on the tile the episode ENDED on; a non-finite ratio (a NaN from a blown-up sim) is dropped
        # so the median can never be NaN.
        obst_r = track_ratio[valid & is_obst.bool()]
        self._obst_ratio.push(obst_r[torch.isfinite(obst_r)])
        # the spawn-kind split (OMNI_SPAWN_SPLIT_KEYS): all terrain, same validity rule
        self._split["track_ratio_mean_rsi"].push(track_ratio[valid & rsi])
        self._split["track_ratio_mean_norsi"].push(track_ratio[valid & ~rsi])
        if track_ratio_move is not None:
            move_valid = (
                torch.ones_like(track_ratio_move, dtype=torch.bool)
                if track_ratio_move_valid is None
                else track_ratio_move_valid.bool()
            )
            self._extra["track_ratio_move_mean"].push(track_ratio_move[move_valid])
            self._split["track_ratio_move_mean_rsi"].push(track_ratio_move[move_valid & rsi])
            self._split["track_ratio_move_mean_norsi"].push(track_ratio_move[move_valid & ~rsi])
        self._extra["heading_err_rms_mean"].push(heading_err_rms)
        self._extra["block_steps_mean"].push(block_steps[is_obst.bool()])
        self._extra["r_max_mean"].push(r_max)
        if self.field_mode:
            if off_tile_frac is not None:
                self._field["off_tile_frac_mean"].push(off_tile_frac)
            if off_tile_block_steps is not None:
                self._field["off_tile_block_steps_mean"].push(off_tile_block_steps)

    def means(self) -> dict[str, float]:
        """Every :meth:`emit_keys` key (NaN while empty), plus each :data:`OMNI_LAZY_KEYS` key whose
        group has received at least one episode (absent, not NaN, before that)."""
        out = self.base.means()
        out.update({k: r.mean() for k, r in self._extra.items()})
        out.update({k: r.mean() for k, r in self._split.items() if r.n > 0})
        out.update({k: r.mean() for k, r in self._field.items()})
        if self._obst_ratio.n > 0:
            # true (interpolated) median, the statistic p2_read's calibrate() takes on a dump
            out[OMNI_OBST_RATIO_KEY] = float(torch.quantile(self._obst_ratio.buf[: self._obst_ratio.n], 0.5))
        return out


# --------------------------------------------------------------------- spawn reporting


def note_spawn_xy_to_parkour(env, env_ids, xy: torch.Tensor, kind=SPAWN_PLATFORM) -> None:
    """Tell every parkour term that implements ``note_spawn_xy`` where env_ids were placed.

    The xy twin of ``parkour_isaaclab.envs.mdp.events.note_spawn_to_parkour`` (same lookup
    through ``env.parkour_manager``, same tolerance of an env without one): the corridor
    only ever needed the spawn's ``x`` because promotion measured downrange distance, while
    the ring measures a radius and needs both coordinates.

    ``kind`` is an ``exposure_stats.SPAWN_*`` int (or an ``(n,)`` long tensor) and drives
    the ledger's non-RSI / RSI split -- the ring reset passes ``SPAWN_PLATFORM``, the RSI
    reset ``SPAWN_RSI``.
    """
    pm = getattr(env, "parkour_manager", None)
    if pm is None:
        return
    for name in pm.active_terms:
        term = pm.get_term(name)
        if hasattr(term, "note_spawn_xy"):
            term.note_spawn_xy(env_ids, xy, kind)

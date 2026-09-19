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

    def __init__(self, n_sectors: int, capacity: int = 1024, device: str | torch.device = "cpu"):
        self.n_sectors = int(n_sectors)
        self.base = ExposureLedger(self.n_sectors, capacity=capacity, device=device)
        self._extra: dict[str, RingMean] = {k: RingMean(capacity, device) for k in OMNI_EXTRA_KEYS}

    @staticmethod
    def emit_keys(n_sectors: int) -> list[str]:
        """Every key :meth:`means` reports (forward keys first, then the omni five)."""
        return ExposureLedger.emit_keys(int(n_sectors)) + list(OMNI_EXTRA_KEYS)

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
    ) -> None:
        """Push one batch of finished episodes (see :data:`OMNI_EXTRA_KEYS` for grouping).

        ``track_ratio_move`` / ``track_ratio_move_valid`` are OPTIONAL: left out (or ``None``),
        nothing is pushed and ``track_ratio_move_mean`` stays an empty group -- which the
        parkour term's ``_update_metrics`` emits as 0.0, exactly as it does for any other ring
        before its first episode. Keeping them optional means a caller that does not measure
        the slot-gated pair (a test double, an older term) is not forced to fabricate one, and
        a fabricated zero would be indistinguishable from a policy that tracks nothing.
        ``track_ratio_move`` without a validity mask counts every episode.
        """
        self.base.push_episodes(per_ep, spawn_kind, is_obst, failed, ep_steps)
        valid = track_ratio_valid.bool()
        self._extra["track_ratio_mean"].push(track_ratio[valid])
        if track_ratio_move is not None:
            move_valid = (
                torch.ones_like(track_ratio_move, dtype=torch.bool)
                if track_ratio_move_valid is None
                else track_ratio_move_valid.bool()
            )
            self._extra["track_ratio_move_mean"].push(track_ratio_move[move_valid])
        self._extra["heading_err_rms_mean"].push(heading_err_rms)
        self._extra["block_steps_mean"].push(block_steps[is_obst.bool()])
        self._extra["r_max_mean"].push(r_max)

    def means(self) -> dict[str, float]:
        out = self.base.means()
        out.update({k: r.mean() for k, r in self._extra.items()})
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

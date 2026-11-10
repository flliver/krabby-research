"""Construction-site sub-terrains (site teacher, ``crab_hex_omni_task/experiments/2026-10-28_site_teacher/
DESIGN.md`` §3).

Two decorated bodies, both installed only under ``KRABBY_FLAT_TERRAIN_GEOM=site``
(``config/omni_site.py``):

``omni_site_field_terrain`` (``OmniSiteFieldTerrainCfg``)
    the legacy field's ``field_layout`` + padding + roughness, byte-identical heights to
    ``omni_terrains.omni_field_terrain`` at the same RNG state, plus the site layers (kind
    ``FIELD_BLOCK`` where the stamp is raised, ``FIELD_PIT`` where it is sunk; level 0; no
    crossing radius, no site difficulty).
``omni_site_terrain`` (``OmniSiteTerrainCfg``)
    one hazard kind per column (``cfg.kind``), laid out by :func:`site_layout` in the tile-centre
    frame on the ladder ``lo + (hi - lo) * d_site`` (:data:`LADDERS`, clamped by the plant caps), every
    feature inside the +-4.80 m square and the walkable level 0 on every pixel with
    ``max(|x|, |y|) >= 4.40`` (the seam rule: tiles meet at level 0, so no seam cliff).

Both return the 5-value raw-body form ``(z, goals, goal_heights, riser, layers)`` that
``terrains/utils.py`` turns into the 8-tuple (DESIGN §2.2); ``layers`` follows the emission
contract of ``crab_hex_omni_task/mdp/site_raster.py`` (:func:`_site_raster`): ``kind`` int8,
``level`` int16 (walkable level: no features, no roughness), ``surface0`` int16 (level + every
feature, ramps included, before roughness), ``material`` int8 zeros, and ``tile = {cross_r, d,
centre_level_m}``. ``riser`` is ``surface0`` with the ramp cells zeroed, so every feature edge is a
vertical wall (``convert_height_field_to_mesh(riser_field=...)``, when ``cfg.vertical_edges``) and
the ramps get none. Roughness is added after ``surface0`` / ``riser`` are taken, so it never enters
the raster or raises a wall.

Site difficulty (DESIGN §3.2, :func:`site_difficulty`): the generator hands each tile its ``d`` inside
``gen_d_range``; a site kind maps it to ``u = (d - D_lo) / (D_hi - D_lo)`` (0 on a degenerate band)
and ``d_site = lo + (hi - lo) u`` inside ``site_d_range`` (``KRABBY_SITE_DIFF``), so in curriculum mode
every kind spans its whole ladder over the 10 levels whatever the legacy band is.

Bottom rung (PLAN_rev1 B4, ``results/replan/probe_curriculum.md`` §2; knob ``KRABBY_SITE_LADDER``): a named ladder
(:data:`SITE_LADDER_BOTTOMS`, today only ``b4``) replaces the ladder BELOW the knee :data:`LADDER_KNEE_D` = 0.6 by a
straight line from its own value at ``d_site = 0`` to today's value at the knee; at and above the knee today's
expression runs verbatim, so every ``d_site >= 0.6`` layout (the eval band, training levels 6-9) is byte-identical.
The pits' cover target ramps the same way (:data:`SITE_LADDER_PIT_COVER0` -> :data:`PIT_COVER`). The ladder reaches
the body as the sub-terrain's ``function`` (:data:`SITE_LADDER_BODIES`: ``omni_site_terrain_b4``, written by the site
applier only when the knob is set), so the config dump records it and a blank knob leaves the cfg untouched.
:func:`primary_metres` is the ladder's primary hazard parameter in metres per kind (:data:`PRIMARY_PARAM`), the
telemetry's ``d -> metres`` reading.

Geometry (DESIGN §3.3; radii are the table's literals, tile-centre frame, metres):

=========== ================================================================== ================
kind        layout                                                             R_cross
=========== ================================================================== ================
pits        square pits (yaw U(0, pi), side s*U(0.9, 1.1)) inside 2.0-3.6,      4.1
            count round(0.40 * annulus / s^2) (<= 120), >= 2 px apart
trench      one annulus centred 2.4, width w                                   2.4 + w/2 + 0.5
ledge_down  the pad disc (r <= R_pad) RAISED by h                              R_pad + 1.5
ledge_up    the pad disc SUNK by h                                             R_pad + 1.5
terraces    risers at 1.7 / 2.3 / 2.9 / 3.5, levels 0, -a, -2a, -a, 0 (down)   4.0
            or 0, +a, +2a, +a, 0 (up), variant p = 0.5
snags       two 0.16 m rings at 2.2 and 3.0, height h                         3.5
straddle    one ring at 2.6, width U(0.25, 0.45), height h                     2.6 + w/2 + 0.5
chests      4 + round(3 d) boxes (chest / table p 0.3), centre 2.4-4.2,       NaN
            footprint clearance >= 2.5 m
walls       2 + round(2 d) tangential 0.20 m segments at r U(2.6, 3.4),        NaN
            angular cover <= 0.5, gaps >= 2.5 m: never a closed ring
rubble      the 2.0-3.6 band, 0.32 m cells of U(0, h_max)                      4.1
ramps       berm 2.0-3.6 (slopes 0.6 m, plateau 0.4 m), H = 0.6 tan(alpha)    4.1
=========== ================================================================== ================

The ring radii assume the default 1.7 m pad (the first hazard 0.3 m beyond it): :func:`site_layout`
refuses a pad above :data:`PAD_R_M`, and the site applier refuses ``KRABBY_FIELD_PAD_R`` above it.

Pure numpy at module scope, like ``omni_terrains.py`` (whose helpers -- ``_oriented_box_mask``,
``_yawed_half_extents``, ``_dilate_square``, ``_tile_pixel_dims``, ``_sector_anchor_goals`` and
``field_layout`` -- are imported, not copied; ``omni_terrains.py`` itself is golden-pinned and never
edited). Draws use the generator's ambient ``np.random`` exactly as ``field_layout`` does (a seeded
generator reproduces; tests pass a ``RandomState``).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ..extreme_parkour.extreme_parkour_terrians import padding_height_field_raw, random_uniform_terrain
from ..utils import parkour_field_to_mesh_xy
from . import omni_terrains as _ot

if TYPE_CHECKING:
    from . import omni_terrains_cfg

__all__ = [
    "PAD_R_M", "HALF_WIDTH_M", "SEAM_R_M", "RAMP_MAX_DEG",
    "DROP_CAP_M", "RISE_CAP_M", "GAP_CAP_M", "STRADDLE_CAP_M", "LADDERS", "CAPS",
    "CROSS_R", "PIT_COVER", "PIT_MAX_COUNT", "PIT_GAP_PX", "BLOCKER_CLEARANCE_M", "WALL_COVER_MAX",
    "SiteLayout", "site_difficulty", "ladder_value", "ramp_cap_deg", "site_layout", "site_height_field",
    "site_field_height_field", "omni_site_field_terrain", "omni_site_terrain",
    "LADDER_KNEE_D", "SITE_LADDER_BOTTOMS", "SITE_LADDER_PIT_COVER0", "PRIMARY_PARAM", "SITE_LADDER_BODIES",
    "pit_cover", "primary_metres", "ladder_of", "omni_site_terrain_b4",
]

#: Spawn pad radius (m): toe reach from the pad centre is <= 1.46 m (``KRABBY_FIELD_PAD_R`` may only lower it).
PAD_R_M = 1.7
#: Half-width of the square every feature pixel lies in (m): ``0.5 * 9.92 - 0.16``, the field's rule.
HALF_WIDTH_M = 4.80
#: Seam rule (m): the walkable level is 0 on every pixel with ``max(|x|, |y|) >= SEAM_R_M``.
SEAM_R_M = 4.40
#: Ramp slope cap (deg) at the nominal friction floor; the site applier lowers it with the site foot
#: friction (:func:`ramp_cap_deg`, DESIGN §3.3).
RAMP_MAX_DEG = 20.0

# Plant caps (m), DESIGN §3.9: ``min(<PLAN ladder top>, <FK limit>)``, written ONCE from
# ``experiments/2026-10-28_site_teacher/fk_reach.json`` (``crab_hex_omni_task/mdp/site_reach.py``, 2026-10-28,
# inputs sha256 d424d3bd...: drop_max 0.460, rise_max 0.822, gap min(1.049, 0.880), step_over_max 1.133 m).
# Every FK limit is above its ladder top, so no cap clamps a ladder (``caps.*.clamped`` = false in the JSON);
# ``tests/unit/test_site_reach.py`` recomputes the limits and fails if a cap ever exceeds one.
DROP_CAP_M = 0.35
RISE_CAP_M = 0.30
GAP_CAP_M = 0.45
STRADDLE_CAP_M = 0.70

#: ``(lo, hi)`` ladder endpoints per kind and parameter (DESIGN §3.3), before the plant caps clamp them.
LADDERS = {
    "pits": {"side_m": (0.30, 0.70), "depth_m": (0.10, 0.40)},
    "trench": {"width_m": (0.15, 0.45), "depth_m": (0.10, 0.45)},
    "ledge_down": {"height_m": (0.05, 0.35)},
    "ledge_up": {"height_m": (0.05, 0.30)},
    "terraces": {"riser_m": (0.05, 0.20)},
    "snags": {"height_m": (0.03, 0.20)},
    "straddle": {"height_m": (0.15, 0.70)},
    "chests": {"height_m": (0.60, 1.20)},
    "walls": {"length_m": (1.0, 4.0), "height_m": (0.80, 2.00)},
    "rubble": {"height_max_m": (0.05, 0.25)},
    "ramps": {"slope_deg": (5.0, 20.0)},
}
#: The ladder parameters the plant caps clamp: ``(kind, parameter) -> cap`` (DESIGN §3.9).
CAPS = {
    ("ledge_down", "height_m"): DROP_CAP_M,
    ("ledge_up", "height_m"): RISE_CAP_M,
    ("trench", "width_m"): GAP_CAP_M,
    ("straddle", "height_m"): STRADDLE_CAP_M,
}
#: Bottom rungs (PLAN_rev1 B4): below this ``d_site`` a named ladder runs linearly from its d = 0 value to today's
#: value here; at and above it, today's expression verbatim.
LADDER_KNEE_D = 0.6
#: ``{ladder: {(kind, parameter): value at d_site = 0}}`` (m). Parameters a ladder does not name (pits side, walls,
#: ramps) keep today's ladder at every d. ``b4`` = PLAN_rev1's B4 table (probe C §2) plus the chests floor (0.85 m
#: at d 0 -> today's 0.96 m at the knee, so every training chest stays above the 0.70 m straddle cap; the +-0.05 m
#: jitter is unchanged).
SITE_LADDER_BOTTOMS = {
    "b4": {
        ("pits", "depth_m"): 0.03,
        ("trench", "width_m"): 0.05,
        ("trench", "depth_m"): 0.03,
        ("ledge_down", "height_m"): 0.02,
        ("ledge_up", "height_m"): 0.02,
        ("terraces", "riser_m"): 0.02,
        ("snags", "height_m"): 0.02,
        ("straddle", "height_m"): 0.04,
        ("chests", "height_m"): 0.85,
        ("rubble", "height_max_m"): 0.02,
    },
}
#: The pits' cover target at ``d_site = 0`` per ladder (linear to :data:`PIT_COVER` at the knee).
SITE_LADDER_PIT_COVER0 = {"b4": 0.05}
#: Each metres-tagged hazard kind's primary ladder parameter (PLAN_rev1 Telemetry / T1): pit and trench depth,
#: straddle height, terrace riser, ledge drop / rise, snag height, rubble max. Chests, walls and ramps have none.
PRIMARY_PARAM = {
    "pits": "depth_m",
    "trench": "depth_m",
    "ledge_down": "height_m",
    "ledge_up": "height_m",
    "terraces": "riser_m",
    "snags": "height_m",
    "straddle": "height_m",
    "rubble": "height_max_m",
}

#: Fixed crossing radii (m) of the kinds whose ``R_cross`` does not depend on the draw (DESIGN §3.3).
CROSS_R = {"pits": 4.1, "terraces": 4.0, "snags": 3.5, "rubble": 4.1, "ramps": 4.1}
#: Hazard band of the annulus kinds (pits, rubble, ramps): ``R_IN <= r <= R_OUT`` (m).
BAND_R_IN_M, BAND_R_OUT_M = 2.0, 3.6
#: The first hazard starts this far beyond the pad (m): ``BAND_R_IN_M - PAD_R_M``.
PAD_CLEARANCE_M = 0.3
PIT_COVER = 0.40
PIT_MAX_COUNT = 120
#: Clear ground pixels between any two pits (Chebyshev, diagonals included).
PIT_GAP_PX = 2
PIT_SIDE_JITTER = (0.9, 1.1)
#: A pit gets this multiple of ``cfg.max_tries`` centre draws (the field's ``FIELD_PIT_TRIES_MULT`` rule).
PIT_TRIES_MULT = 4
#: Ray-angle bins of the pits' gap-seeking placement (the forced-crossing test's 720 rays, 0.5 deg apart).
PIT_ANGLE_BINS = 720
TRENCH_R_M = 2.4
TERRACE_RISER_R_M = (1.7, 2.3, 2.9, 3.5)
TERRACE_STEPS = (1, 2, 1)          # the three inner bands' level in risers (x the variant's sign)
SNAG_R_M = (2.2, 3.0)
SNAG_WIDTH_M = 0.16
STRADDLE_R_M = 2.6
STRADDLE_WIDTH_M = (0.25, 0.45)
CHEST_R_M = (2.4, 4.2)
CHEST_SIDES_M = ((0.6, 1.2), (0.6, 0.8))
TABLE_SIDES_M = ((1.2, 2.0), (0.8, 0.8))
TABLE_P = 0.3
BLOCKER_H_JITTER_M = 0.05
#: Pairwise footprint clearance between blockers (m): the passage a 1.2 m-wide robot walks through.
BLOCKER_CLEARANCE_M = 2.5
WALL_R_M = (2.6, 3.4)
WALL_THICKNESS_M = 0.20
WALL_LENGTH_JITTER_M = 0.2
#: Upper bound on the walls' angular coverage at their radius (sum of the segments' angles / 2 pi).
WALL_COVER_MAX = 0.5
RUBBLE_CELL_PX = 4
RAMP_RUN_M = 0.6
RAMP_PLATEAU_M = (2.6, 3.0)
#: Ramp cap below the friction angle (deg): ``min(20, atan(mu_floor) - 3)``.
RAMP_FRICTION_MARGIN_DEG = 3.0

_SITE_RASTER_PY = (Path(__file__).resolve().parents[3] / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task"
                   / "mdp" / "site_raster.py")
_SITE_RASTER_DOTTED = "parkour_tasks.crab_hex_omni_task.mdp.site_raster"
_SITE_RASTER_STANDALONE = "crab_hex_omni_site_raster"


def _site_raster():
    """The kind constants' module (``crab_hex_omni_task/mdp/site_raster.py``), imported lazily: the
    package object when it is already loaded, else a standalone path-load (its module scope is
    numpy / torch only). Never imported at module scope: the task package's ``__init__`` chain
    imports this library's cfgs, so a module-scope import would be circular under Kit."""
    mod = sys.modules.get(_SITE_RASTER_DOTTED) or sys.modules.get(_SITE_RASTER_STANDALONE)
    if mod is None:
        spec = importlib.util.spec_from_file_location(_SITE_RASTER_STANDALONE, _SITE_RASTER_PY)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_SITE_RASTER_STANDALONE] = mod
        spec.loader.exec_module(mod)
    return mod


class SiteLayout(NamedTuple):
    """One site tile in the raw body's pixel frame (vertical units for the int arrays).

    ``raw`` the stamped heights before padding / roughness (float, = ``surface0``), ``riser`` the int16
    riser field (stamped feature + level heights, ramp cells 0), ``kind`` int8 pixel kinds, ``level``
    int16 walkable level, ``surface0`` int16 pre-roughness surface, ``cross_r`` the crossing radius (m,
    NaN for blockers). ``n_placed`` / ``n_skipped`` count the drawn objects of the scatter kinds (pits,
    chests, walls) that were stamped / given up after ``max_tries`` (0 / 0 for the ring kinds).
    """

    raw: np.ndarray
    riser: np.ndarray
    kind: np.ndarray
    level: np.ndarray
    surface0: np.ndarray
    cross_r: float
    n_placed: int = 0
    n_skipped: int = 0


# ============================================================================ difficulty and ladders
def site_difficulty(difficulty: float, cfg) -> float:
    """The tile's site difficulty (DESIGN §3.2): ``u = (d - D_lo) / (D_hi - D_lo)`` inside
    ``cfg.gen_d_range`` (0 when the band is degenerate, < 1e-9; clipped into [0, 1]), then
    ``lo + (hi - lo) u`` inside ``cfg.site_d_range``."""
    d_lo, d_hi = (float(v) for v in cfg.gen_d_range)
    lo, hi = (float(v) for v in cfg.site_d_range)
    span = d_hi - d_lo
    u = 0.0 if abs(span) < 1e-9 else (float(difficulty) - d_lo) / span
    u = min(1.0, max(0.0, u))
    return lo + (hi - lo) * u


def _check_ladder(ladder: str | None) -> None:
    if ladder is not None and ladder not in SITE_LADDER_BOTTOMS:
        raise ValueError(f"unknown site ladder {ladder!r}; valid: None (today's), {', '.join(SITE_LADDER_BOTTOMS)}")


def ladder_value(kind: str, param: str, d_site: float, ladder: str | None = None) -> float:
    """``lo + (hi - lo) * d_site`` of :data:`LADDERS` ``[kind][param]``, clamped to its plant cap
    (:data:`CAPS`) where one applies.

    ``ladder`` (``KRABBY_SITE_LADDER``; None = today's): below :data:`LADDER_KNEE_D`, a parameter the ladder names
    in :data:`SITE_LADDER_BOTTOMS` is ``bottom + (today(knee) - bottom) * d_site / knee``; every other case runs
    today's expression unchanged (so ``d_site >= knee`` is byte-identical)."""
    if ladder is not None:
        _check_ladder(ladder)
        bottom = SITE_LADDER_BOTTOMS[ladder].get((kind, param))
        d = float(d_site)
        if bottom is not None and d < LADDER_KNEE_D:
            knee = ladder_value(kind, param, LADDER_KNEE_D)
            return bottom + (knee - bottom) * (d / LADDER_KNEE_D)
    lo, hi = LADDERS[kind][param]
    value = lo + (hi - lo) * float(d_site)
    cap = CAPS.get((kind, param))
    return value if cap is None else min(value, cap)


def pit_cover(d_site: float, ladder: str | None = None) -> float:
    """The pits' cover target: :data:`PIT_COVER` at every d (today's), or under ``ladder`` below the knee
    ``cover0 + (PIT_COVER - cover0) * d_site / knee`` (:data:`SITE_LADDER_PIT_COVER0`)."""
    if ladder is not None:
        _check_ladder(ladder)
        cover0 = SITE_LADDER_PIT_COVER0.get(ladder)
        d = float(d_site)
        if cover0 is not None and d < LADDER_KNEE_D:
            return cover0 + (PIT_COVER - cover0) * (d / LADDER_KNEE_D)
    return PIT_COVER


def primary_metres(kind: str, d_site: float, ladder: str | None = None) -> float:
    """The primary hazard parameter (m, :data:`PRIMARY_PARAM`) of ``kind`` at ``d_site`` on ``ladder`` -- exactly
    the value the layout stamps from (before the vertical-scale rounding). ``KeyError`` for a kind without one
    (chests, walls, ramps, the flat / legacy-field columns)."""
    return ladder_value(kind, PRIMARY_PARAM[kind], d_site, ladder)


def ramp_cap_deg(mu_floor: float | None) -> float:
    """``min(20, degrees(atan(mu_floor)) - 3)`` -- the ramp cap "by friction" (DESIGN §3.3); 20 when the
    site foot friction is not armed (``mu_floor`` None: the nominal floor 0.8 holds 20 deg)."""
    if mu_floor is None:
        return RAMP_MAX_DEG
    return min(RAMP_MAX_DEG, math.degrees(math.atan(float(mu_floor))) - RAMP_FRICTION_MARGIN_DEG)


# ============================================================================ the tile canvas
class _Canvas:
    """Pixel-centre grid (tile-centre frame, metres) and the four layers being stamped (int, vertical
    units): ``kind``, ``level``, ``feature`` (heights on top of the level), ``ramp`` (cells whose
    feature is a slope, excluded from the riser field)."""

    def __init__(self, cfg):
        self.w, self.ell = _ot._tile_pixel_dims(cfg)
        self.hs = float(cfg.horizontal_scale)
        self.vs = float(cfg.vertical_scale)
        self.cx = 0.5 * cfg.size[0]
        self.cy = 0.5 * cfg.size[1]
        self.xs = (np.arange(self.w) + 0.5) * self.hs - self.cx
        self.ys = (np.arange(self.ell) + 0.5) * self.hs - self.cy
        self.dx, self.dy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.r = np.sqrt(self.dx * self.dx + self.dy * self.dy)
        self.kind = np.zeros((self.w, self.ell), dtype=np.int8)
        self.level = np.zeros((self.w, self.ell), dtype=np.int32)
        self.feature = np.zeros((self.w, self.ell), dtype=np.int32)
        self.ramp = np.zeros((self.w, self.ell), dtype=bool)

    def px(self, metres: float) -> int:
        return int(round(float(metres) / self.vs))

    def window(self, bx: float, by: float, ex: float, ey: float) -> tuple[slice, slice]:
        """Pixel slices covering the axis-aligned box ``bx +- ex``, ``by +- ey`` (one pixel of slack)."""
        i0 = max(0, int(math.floor((bx - ex + self.cx) / self.hs - 0.5)) - 1)
        i1 = min(self.w, int(math.ceil((bx + ex + self.cx) / self.hs - 0.5)) + 2)
        j0 = max(0, int(math.floor((by - ey + self.cy) / self.hs - 0.5)) - 1)
        j1 = min(self.ell, int(math.ceil((by + ey + self.cy) / self.hs - 0.5)) + 2)
        return slice(i0, max(i0, i1)), slice(j0, max(j0, j1))

    def box(self, bx: float, by: float, yaw: float, length: float, width: float):
        """``(si, sj, mask)``: the oriented box's pixels inside its window (``_oriented_box_mask``)."""
        ex, ey = _ot._yawed_half_extents(yaw, length, width)
        si, sj = self.window(bx, by, ex, ey)
        mask = _ot._oriented_box_mask(self.dx[si, sj] - bx, self.dy[si, sj] - by, yaw, length, width)
        return si, sj, mask

    def stamp_box(self, si, sj, mask, kind: int, feature_px: int) -> np.ndarray:
        """Stamp a box (raised: ``max``; sunk: ``min``) and return its full-tile bool mask."""
        full = np.zeros((self.w, self.ell), dtype=bool)
        full[si, sj] = mask
        self.feature[full] = (np.maximum if feature_px > 0 else np.minimum)(self.feature[full], feature_px)
        self.kind[full] = kind
        return full


def _check_cfg(cfg) -> tuple[float, float]:
    """``(pad_r, half)`` after the bounds the layouts assume: the half-width inside the tile minus its
    margin (``field_layout``'s rule) and the pad at most :data:`PAD_R_M` (the ring radii start
    :data:`PAD_CLEARANCE_M` beyond the default pad)."""
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    half = float(cfg.ring_outer_radius)
    margin = float(getattr(cfg, "field_margin", 0.0))
    if half > min(cx, cy) - margin + 1e-9:
        raise ValueError(f"site half-width ring_outer_radius={half} exceeds the tile's half-size {min(cx, cy):.4f} "
                         f"minus field_margin={margin}")
    pad_r = float(cfg.clear_radius)
    if not 0.0 < pad_r <= PAD_R_M + 1e-9:
        raise ValueError(f"site clear_radius={pad_r} must be in (0, {PAD_R_M}] m: the hazard radii start "
                         f"{PAD_CLEARANCE_M} m beyond the default pad (DESIGN §3.3)")
    return pad_r, half


# ============================================================================ the kinds (DESIGN §3.3)
def _ring(c: _Canvas, r_c: float, width: float) -> np.ndarray:
    return np.abs(c.r - r_c) <= 0.5 * width


def _band(c: _Canvas) -> np.ndarray:
    return (c.r >= BAND_R_IN_M) & (c.r <= BAND_R_OUT_M)


def _layout_pits(rng, d, c: _Canvas, cfg, pad_r, ladder=None):
    sr = _site_raster()
    side = ladder_value("pits", "side_m", d, ladder)
    depth_px = c.px(ladder_value("pits", "depth_m", d, ladder))
    area = math.pi * (BAND_R_OUT_M ** 2 - BAND_R_IN_M ** 2)
    n = min(PIT_MAX_COUNT, int(round(pit_cover(d, ladder) * area / (side * side))))
    max_tries = int(getattr(cfg, "max_tries", 64))
    near = np.zeros((c.w, c.ell), dtype=bool)       # pixels within PIT_GAP_PX of a stamped pit
    placed = 0
    covered = np.zeros(PIT_ANGLE_BINS, dtype=bool)   # ray angles (0.5 deg bins) a stamped pit already blocks
    bin_w = 2.0 * np.pi / PIT_ANGLE_BINS
    sector = 2.0 * np.pi / max(n, 1)
    for i in range(n):
        yaw = rng.uniform(0.0, np.pi)
        s = side * rng.uniform(*PIT_SIDE_JITTER)
        # area-uniform centre in the annulus shrunk by the INSCRIBED half-side; the pixels themselves are then
        # checked against the band (a circumscribed shrink would leave a 0.5 m centre band for 0.77 m pits)
        r_lo, r_hi = BAND_R_IN_M + 0.5 * s, BAND_R_OUT_M - 0.5 * s
        hit = None
        for t in range(max_tries * PIT_TRIES_MULT):
            r_c = math.sqrt(rng.uniform(r_lo * r_lo, r_hi * r_hi))
            # angle: pit i's first max_tries centres in its own 2 pi / n sector (the pits ring the pad evenly),
            # the next max_tries in the WIDEST run of ray angles no pit blocks yet, the rest anywhere -- so a
            # straight walk out of the pad cannot slip through a wide angular gap (DESIGN §3.5)
            if t < max_tries:
                theta = (i + rng.uniform(0.0, 1.0)) * sector
            else:
                lo_a, hi_a = _widest_open_run(covered) if t < 2 * max_tries else (0, PIT_ANGLE_BINS)
                theta = rng.uniform(lo_a * bin_w, hi_a * bin_w)
            si, sj, mask = c.box(r_c * math.cos(theta), r_c * math.sin(theta), yaw, s, s)
            if not mask.any() or (mask & near[si, sj]).any():
                continue
            r_px = c.r[si, sj][mask]
            if r_px.min() < BAND_R_IN_M or r_px.max() > BAND_R_OUT_M:
                continue
            hit = (si, sj, mask)
            break
        if hit is None:
            continue
        full = c.stamp_box(*hit, sr.KIND_PIT, -depth_px)
        near |= _ot._dilate_square(full, PIT_GAP_PX)
        ang = np.arctan2(c.dy[full], c.dx[full]) % (2.0 * np.pi)
        half = 0.5 * c.hs / c.r[full]               # each pixel's angular half-width seen from the centre
        for a0, a1 in zip(np.floor((ang - half) / bin_w).astype(int), np.floor((ang + half) / bin_w).astype(int)):
            covered[np.arange(a0, a1 + 1) % PIT_ANGLE_BINS] = True
        placed += 1
    return CROSS_R["pits"], placed, n - placed


def _widest_open_run(covered: np.ndarray) -> tuple[int, int]:
    """``(lo, hi)`` bin bounds (``hi`` exclusive, may exceed the bin count: circular) of the widest run of
    False in the circular ``covered``; the whole circle when nothing is covered or everything is."""
    n = covered.size
    if covered.all() or not covered.any():
        return 0, n
    start = int(np.argmax(covered))                 # rotate so the scan starts on a covered bin
    rolled = np.roll(covered, -start)
    best, best_lo, run_lo = 0, 0, None
    for k in range(n + 1):
        free = k < n and not rolled[k]
        if free and run_lo is None:
            run_lo = k
        elif not free and run_lo is not None:
            if k - run_lo > best:
                best, best_lo = k - run_lo, run_lo
            run_lo = None
    return start + best_lo, start + best_lo + best


def _layout_trench(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    w = ladder_value("trench", "width_m", d, ladder)
    mask = _ring(c, TRENCH_R_M, w)
    c.feature[mask] = -c.px(ladder_value("trench", "depth_m", d, ladder))
    c.kind[mask] = sr.KIND_TRENCH
    return TRENCH_R_M + 0.5 * w + 0.5, 0, 0


def _layout_ledge(sign: int, kind: str):
    def layout(rng, d, c, cfg, pad_r, ladder=None):
        sr = _site_raster()
        pad = c.r <= pad_r
        c.level[pad] = sign * c.px(ladder_value(kind, "height_m", d, ladder))
        c.kind[pad] = sr.KIND_LEDGE
        return pad_r + 1.5, 0, 0
    return layout


def _layout_terraces(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    a_px = c.px(ladder_value("terraces", "riser_m", d, ladder))
    sign = -1 if rng.uniform(0.0, 1.0) < 0.5 else 1     # down variant with p = 0.5
    for (r0, r1), steps in zip(zip(TERRACE_RISER_R_M[:-1], TERRACE_RISER_R_M[1:]), TERRACE_STEPS):
        c.level[(c.r >= r0) & (c.r < r1)] = sign * steps * a_px
    c.kind[c.level != 0] = sr.KIND_TERRACE
    return CROSS_R["terraces"], 0, 0


def _layout_snags(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    h_px = c.px(ladder_value("snags", "height_m", d, ladder))
    for r_c in SNAG_R_M:
        mask = _ring(c, r_c, SNAG_WIDTH_M)
        c.feature[mask] = h_px
        c.kind[mask] = sr.KIND_SNAG
    return CROSS_R["snags"], 0, 0


def _layout_straddle(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    w = rng.uniform(*STRADDLE_WIDTH_M)
    mask = _ring(c, STRADDLE_R_M, w)
    c.feature[mask] = c.px(ladder_value("straddle", "height_m", d, ladder))
    c.kind[mask] = sr.KIND_STRADDLE
    return STRADDLE_R_M + 0.5 * w + 0.5, 0, 0


def _clear_of(c: _Canvas, si, sj, mask, others: list[np.ndarray], clearance: float) -> bool:
    """True when the box pixels keep ``clearance`` (m, footprint edge to edge: pixel-centre distance
    minus one pixel) to every pixel of every earlier blocker (``others``: (k, 2) pixel-centre coords)."""
    if not others:
        return True
    pts = np.stack([c.dx[si, sj][mask], c.dy[si, sj][mask]], axis=-1)
    prev = np.concatenate(others, axis=0)
    d2 = ((pts[:, None, :] - prev[None, :, :]) ** 2).sum(-1)
    return math.sqrt(float(d2.min())) - c.hs >= clearance


def _inside_half(c: _Canvas, si, sj, mask, half: float) -> bool:
    return bool((np.maximum(np.abs(c.dx[si, sj][mask]), np.abs(c.dy[si, sj][mask])) <= half).all())


def _layout_chests(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    half = float(cfg.ring_outer_radius)
    h_lad = ladder_value("chests", "height_m", d, ladder)
    lo_h, hi_h = LADDERS["chests"]["height_m"]
    n = 4 + int(round(3.0 * d))
    max_tries = int(getattr(cfg, "max_tries", 64))
    stamped: list[np.ndarray] = []
    for _ in range(n):
        yaw = rng.uniform(0.0, np.pi)
        sides = TABLE_SIDES_M if rng.uniform(0.0, 1.0) < TABLE_P else CHEST_SIDES_M
        length = rng.uniform(*sides[0])
        width = rng.uniform(*sides[1])
        h = min(hi_h, max(lo_h, h_lad + rng.uniform(-BLOCKER_H_JITTER_M, BLOCKER_H_JITTER_M)))
        hit = None
        for _try in range(max_tries):
            r_c = math.sqrt(rng.uniform(CHEST_R_M[0] ** 2, CHEST_R_M[1] ** 2))
            theta = rng.uniform(0.0, 2.0 * np.pi)
            si, sj, mask = c.box(r_c * math.cos(theta), r_c * math.sin(theta), yaw, length, width)
            if not mask.any() or not _inside_half(c, si, sj, mask, half):
                continue
            if not _clear_of(c, si, sj, mask, stamped, BLOCKER_CLEARANCE_M):
                continue
            hit = (si, sj, mask)
            break
        if hit is None:
            continue
        full = c.stamp_box(*hit, sr.KIND_BLOCKER_CHEST, c.px(h))
        stamped.append(np.stack([c.dx[full], c.dy[full]], axis=-1))
    return math.nan, len(stamped), n - len(stamped)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _layout_walls(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    half = float(cfg.ring_outer_radius)
    l_lad = ladder_value("walls", "length_m", d, ladder)
    h_px = c.px(ladder_value("walls", "height_m", d, ladder))
    k = 2 + int(round(2.0 * d))
    max_tries = int(getattr(cfg, "max_tries", 64))
    stamped: list[np.ndarray] = []
    arcs: list[tuple[float, float]] = []        # (centre angle, half-angle) of every stamped segment
    for _ in range(k):
        length = l_lad + rng.uniform(-WALL_LENGTH_JITTER_M, WALL_LENGTH_JITTER_M)
        hit = None
        for _try in range(max_tries):
            r_w = rng.uniform(*WALL_R_M)
            theta = rng.uniform(0.0, 2.0 * np.pi)
            phi = math.atan2(0.5 * length, r_w)
            # angular cover of every segment at its radius, and never overlapping another in angle
            if (sum(2.0 * p for _t, p in arcs) + 2.0 * phi) / (2.0 * math.pi) > WALL_COVER_MAX + 1e-12:
                continue
            if any(abs(_wrap(theta - t)) <= phi + p for t, p in arcs):
                continue
            si, sj, mask = c.box(r_w * math.cos(theta), r_w * math.sin(theta), theta + 0.5 * math.pi,
                                 length, WALL_THICKNESS_M)
            if not mask.any() or not _inside_half(c, si, sj, mask, half):
                continue
            if not _clear_of(c, si, sj, mask, stamped, BLOCKER_CLEARANCE_M):
                continue
            hit = (si, sj, mask, theta, phi)
            break
        if hit is None:
            continue
        full = c.stamp_box(*hit[:3], sr.KIND_BLOCKER_WALL, h_px)
        stamped.append(np.stack([c.dx[full], c.dy[full]], axis=-1))
        arcs.append((hit[3], hit[4]))
    return math.nan, len(stamped), k - len(stamped)


def _layout_rubble(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    h_max = ladder_value("rubble", "height_max_m", d, ladder)
    n_i = -(-c.w // RUBBLE_CELL_PX)
    n_j = -(-c.ell // RUBBLE_CELL_PX)
    cells = np.rint(rng.uniform(0.0, h_max, size=(n_i, n_j)) / c.vs).astype(np.int32)
    per_px = np.repeat(np.repeat(cells, RUBBLE_CELL_PX, axis=0), RUBBLE_CELL_PX, axis=1)[: c.w, : c.ell]
    band = _band(c)
    c.feature[band] = per_px[band]
    c.kind[band] = sr.KIND_RUBBLE
    return CROSS_R["rubble"], 0, 0


def _layout_ramps(rng, d, c, cfg, pad_r, ladder=None):
    sr = _site_raster()
    alpha = min(ladder_value("ramps", "slope_deg", d, ladder), float(getattr(cfg, "ramp_max_deg", RAMP_MAX_DEG)))
    t = math.tan(math.radians(alpha))
    r = c.r
    up = (r >= BAND_R_IN_M) & (r < RAMP_PLATEAU_M[0])
    top = (r >= RAMP_PLATEAU_M[0]) & (r < RAMP_PLATEAU_M[1])
    down = (r >= RAMP_PLATEAU_M[1]) & (r < BAND_R_OUT_M)
    z = np.zeros_like(r)
    z[up] = (r[up] - BAND_R_IN_M) * t
    z[top] = RAMP_RUN_M * t
    z[down] = (BAND_R_OUT_M - r[down]) * t
    berm = up | top | down
    c.feature[berm] = np.rint(z[berm] / c.vs).astype(np.int32)
    c.kind[berm] = sr.KIND_RAMP
    c.ramp |= berm
    return CROSS_R["ramps"], 0, 0


_LAYOUTS = {
    "pits": _layout_pits,
    "trench": _layout_trench,
    "ledge_down": _layout_ledge(+1, "ledge_down"),
    "ledge_up": _layout_ledge(-1, "ledge_up"),
    "terraces": _layout_terraces,
    "snags": _layout_snags,
    "straddle": _layout_straddle,
    "chests": _layout_chests,
    "walls": _layout_walls,
    "rubble": _layout_rubble,
    "ramps": _layout_ramps,
}


def site_layout(rng, d_site: float, cfg: "omni_terrains_cfg.OmniSiteTerrainCfg",
                ladder: str | None = None) -> SiteLayout:
    """Pure body: dispatch on ``cfg.kind`` to the kind's layout (DESIGN §3.3) at site difficulty
    ``d_site``. ``rng`` is ``np.random`` (or a seeded ``RandomState``); ``cfg.size`` is the decorator's
    already-reduced tile size, as for every terrain body. ``ladder`` is the bottom rung
    (:data:`SITE_LADDER_BOTTOMS`; None = today's ladder): it moves values only, never a draw.

    Raises ``ValueError`` for an unknown kind or ladder, a ``d_site`` outside [0, 1], a pad above :data:`PAD_R_M`,
    a half-width outside the tile, and -- a construction check, never expected to fire -- a feature
    pixel outside the +-``ring_outer_radius`` square or a non-zero level at ``max(|x|, |y|) >=``
    :data:`SEAM_R_M`.
    """
    kind = str(cfg.kind)
    if kind not in _LAYOUTS:
        raise ValueError(f"unknown site kind {kind!r}; valid: {', '.join(_LAYOUTS)}")
    d = float(d_site)
    if not 0.0 <= d <= 1.0:
        raise ValueError(f"site difficulty must be in [0, 1], got {d_site!r}")
    _check_ladder(ladder)
    pad_r, half = _check_cfg(cfg)
    c = _Canvas(cfg)
    cross_r, n_placed, n_skipped = _LAYOUTS[kind](rng, d, c, cfg, pad_r, ladder)

    cheb = np.maximum(np.abs(c.dx), np.abs(c.dy))
    if ((c.kind != 0) & (cheb > half + 1e-9)).any():
        raise ValueError(f"site kind {kind!r}: a feature pixel lies outside the +-{half} m square")
    if (c.level[cheb >= SEAM_R_M] != 0).any():
        raise ValueError(f"site kind {kind!r}: non-zero level beyond the {SEAM_R_M} m seam")

    surface0 = c.level + c.feature
    riser = np.where(c.ramp, c.level, surface0)
    return SiteLayout(
        raw=surface0.astype(np.float64),
        riser=riser.astype(np.int16),
        kind=c.kind.copy(),
        level=c.level.astype(np.int16),
        surface0=surface0.astype(np.int16),
        cross_r=float(cross_r),
        n_placed=int(n_placed),
        n_skipped=int(n_skipped),
    )


# ============================================================================ the raw bodies
def _finish(raw: np.ndarray, difficulty: float, cfg, num_goals: int):
    """Padding + roughness exactly as ``omni_terrains.field_height_field`` applies them, plus the sector
    anchors: ``(height_field_raw, goals (m, corner frame), goal_heights)``."""
    goals, goal_heights = _ot._sector_anchor_goals(cfg, num_goals)
    height_field_raw = padding_height_field_raw(raw.copy(), cfg)
    if cfg.apply_roughness:
        height_field_raw = random_uniform_terrain(difficulty, cfg, height_field_raw)
    return height_field_raw, goals * cfg.horizontal_scale, goal_heights


def site_field_height_field(rng, difficulty: float, cfg: "omni_terrains_cfg.OmniSiteFieldTerrainCfg",
                            num_goals: int):
    """Pure body of :func:`omni_site_field_terrain`: ``omni_terrains.field_height_field`` op for op (the
    same draws, padding and roughness, so byte-identical heights at the same RNG state), plus the site
    layers. Returns ``(z, goals, goal_heights, stamped, layers)``; ``stamped`` is the field's 4th value
    (the riser field when ``cfg.vertical_edges``)."""
    sr = _site_raster()
    raw, _occupancy, _n_placed = _ot.field_layout(rng, difficulty, cfg)
    stamped = np.rint(raw).astype(np.int16)
    height_field_raw, goals, goal_heights = _finish(raw, difficulty, cfg, num_goals)
    kind = np.where(stamped > 0, sr.KIND_FIELD_BLOCK, np.where(stamped < 0, sr.KIND_FIELD_PIT, sr.KIND_GROUND))
    layers = {
        "kind": kind.astype(np.int8),
        "level": np.zeros_like(stamped),
        "surface0": stamped.copy(),
        "material": np.zeros(stamped.shape, dtype=np.int8),
        "tile": {"cross_r": math.nan, "d": math.nan, "centre_level_m": 0.0},
    }
    return height_field_raw, goals, goal_heights, stamped, layers


def site_height_field(rng, difficulty: float, cfg: "omni_terrains_cfg.OmniSiteTerrainCfg", num_goals: int,
                      ladder: str | None = None):
    """Pure body of :func:`omni_site_terrain` (``ladder`` None) and :func:`omni_site_terrain_b4` (``"b4"``):
    :func:`site_difficulty`, :func:`site_layout` on ``ladder``, then padding + roughness (at the GENERATOR
    difficulty, as the field) and the sector anchors. Returns ``(z, goals, goal_heights, riser, layers)``;
    ``layers["tile"]`` carries ``cross_r``, ``d`` (= d_site) and ``centre_level_m`` (the pad's level, m: what
    the spawn adds under ``spawn_level_z``)."""
    d_site = site_difficulty(difficulty, cfg)
    lay = site_layout(rng, d_site, cfg, ladder)
    height_field_raw, goals, goal_heights = _finish(lay.raw, difficulty, cfg, num_goals)
    w, ell = lay.level.shape
    layers = {
        "kind": lay.kind,
        "level": lay.level,
        "surface0": lay.surface0,
        "material": np.zeros(lay.kind.shape, dtype=np.int8),
        "tile": {"cross_r": lay.cross_r, "d": float(d_site),
                 "centre_level_m": float(lay.level[w // 2, ell // 2]) * float(cfg.vertical_scale)},
    }
    return height_field_raw, goals, goal_heights, lay.riser, layers


@parkour_field_to_mesh_xy
def omni_site_field_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniSiteFieldTerrainCfg", num_goals: int):
    return site_field_height_field(np.random, difficulty, cfg, num_goals)


@parkour_field_to_mesh_xy
def omni_site_terrain(difficulty: float, cfg: "omni_terrains_cfg.OmniSiteTerrainCfg", num_goals: int):
    return site_height_field(np.random, difficulty, cfg, num_goals)


@parkour_field_to_mesh_xy
def omni_site_terrain_b4(difficulty: float, cfg: "omni_terrains_cfg.OmniSiteTerrainCfg", num_goals: int):
    """:func:`omni_site_terrain` on the ``b4`` bottom rung (``KRABBY_SITE_LADDER=b4``; PLAN_rev1 B4)."""
    return site_height_field(np.random, difficulty, cfg, num_goals, "b4")


#: ``{ladder: decorated body}`` -- what the site applier writes into every hazard sub-terrain's ``function`` when
#: ``KRABBY_SITE_LADDER`` is set (the dump then names the ladder); a blank knob leaves :func:`omni_site_terrain`.
SITE_LADDER_BODIES = {"b4": omni_site_terrain_b4}
_LADDER_OF_BODY = {"omni_site_terrain": None, **{fn.__name__: name for name, fn in SITE_LADDER_BODIES.items()}}


def ladder_of(function) -> str | None:
    """The ladder a hazard sub-terrain's ``function`` stamps on: None for :func:`omni_site_terrain` (today's),
    its :data:`SITE_LADDER_BODIES` name otherwise (matched by ``__name__``, so a path-loaded twin of this module
    reads the same). ``ValueError`` for any other function (not a site hazard body)."""
    name = getattr(function, "__name__", None)
    if name not in _LADDER_OF_BODY:
        raise ValueError(f"{function!r} is not a site hazard body (one of {', '.join(_LADDER_OF_BODY)})")
    return _LADDER_OF_BODY[name]

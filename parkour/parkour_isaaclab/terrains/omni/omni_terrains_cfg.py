from dataclasses import MISSING

from isaaclab.utils import configclass

from ..extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
from . import omni_terrains, site_terrains


@configclass
class OmniRingTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """A clear central disc out to `clear_radius`, then an annulus (out to
    `ring_outer_radius`) scattered with `n_blocks` randomly placed, randomly yawed bars
    and slabs -- see `omni_terrains.ring_height_field` for the exact geometry. The height
    formulas are the recal2b2 strings (`crab_hex_env_cfg.py:400-406`), carried over
    verbatim so the omni ring's obstacle heights match the forward/sideways campaigns'
    calibration to the measured plant.
    """

    function = omni_terrains.omni_ring_terrain

    clear_radius: float = 2.0
    ring_outer_radius: float = 4.5
    n_blocks: int = 8
    bar_fraction: float = 0.5

    # recal2b2 (crab_hex_env_cfg.py:400-406), shared by both block types: bars sample a
    # height from this (lo, hi) range per block; slabs use `step_height` verbatim (a
    # single value at the given difficulty, not a sampled range).
    hurdle_height_range: str = "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
    step_height: str = "0.02 + 0.06 * difficulty"


@configclass
class OmniFlatTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """No blocks -- flat (plus roughness noise) inside the same ring geometry, so the
    8 sector anchors land at the same radius as `OmniRingTerrainCfg`'s regardless of
    which sub-terrain a tile actually is."""

    function = omni_terrains.omni_flat_terrain

    clear_radius: float = 2.0
    ring_outer_radius: float = 4.5


@configclass
class OmniFieldTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Omni phase-2 plan, step 2 (2026-10-06): a small clear spawn PAD, then bars, slabs and pits
    scattered uniformly over the whole tile square -- see ``omni_terrains.field_layout`` for the
    exact placement rule. A SIBLING of :class:`OmniRingTerrainCfg`, not a subclass, so the ring
    guard's ``isinstance`` classification can never mistake a field tile for a ring tile (or the
    reverse).

    The two radius fields keep the ring's NAMES because the sector anchors
    (``_sector_anchor_goals``), the knob layer and the parkour term's
    ``clear_radius_m`` / ``ring_outer_radius_m`` sync all read them by name:

    * ``clear_radius`` is the spawn pad (1.7 m: toe reach <= 1.46 m from foot FK, hardware max
      1.55 m, plus margin; knob ``KRABBY_FIELD_PAD_R``, refused below 1.6 m);
    * ``ring_outer_radius`` is the HALF-WIDTH of the square blocks are confined to,
      ``0.5 * 9.92 - field_margin`` = 4.80 m (9.92 m = the decorator-reduced 10 m tile), so a
      2-pixel (0.16 m) strip along every tile edge stays block-free and the 1-px ``pad_width``
      padding never clips a block.

    Sizes are the ring's distributions (bars U(1.0, 2.0) x U(0.16, 0.32) m, slabs and pits
    U(0.6, 1.2) m square); heights are the recal2b2 strings verbatim, and the pit depth reuses the
    slab formula (a pit is a slab pushed down). Densities: 14 (``omni_field_sparse``) and 28
    (``omni_field_dense``) blocks per tile, i.e. ~0.17 / 0.34 per m2 of the ~82 m2 of the 9.6 m
    square outside the pad -- within ~8 % of the ring's 0.157 / 0.313 per m2 of annulus
    (~8.5 % / 17 % cover), on +75 % blocks per tile.

    ``vertical_edges`` (knob ``KRABBY_FIELD_EDGES=wall``) forces vertical risers on every block
    edge -- without it, every block under 0.12 m renders as a one-pixel ramp, exactly like the
    ring and the forward corridors (``slope_threshold`` 1.5 at hs 0.08 / vs 0.005). Built and
    tested, not trained in phase 2.
    """

    function = omni_terrains.omni_field_terrain

    clear_radius: float = 1.7
    ring_outer_radius: float = 4.80
    field_margin: float = 0.16
    n_blocks: int = 14
    bar_fraction: float = 0.4
    pit_fraction: float = 0.2
    min_gap: float = 0.0
    max_tries: int = 64
    bar_length_range: tuple[float, float] = (1.0, 2.0)
    bar_thickness_range: tuple[float, float] = (0.16, 0.32)
    slab_side_range: tuple[float, float] = (0.6, 1.2)

    # recal2b2 (crab_hex_env_cfg.py:400-406), as on OmniRingTerrainCfg; pits sink by the slab
    # formula (0.02 + 0.06 * d).
    hurdle_height_range: str = "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
    step_height: str = "0.02 + 0.06 * difficulty"
    pit_depth: str = "0.02 + 0.06 * difficulty"

    vertical_edges: bool = False


@configclass
class OmniSiteFieldTerrainCfg(OmniFieldTerrainCfg):
    """Site teacher (``crab_hex_omni_task/experiments/2026-10-28_site_teacher/DESIGN.md`` §3.1): the
    legacy field on a SITE generator -- ``field_layout`` + padding + roughness, byte-identical heights
    to :class:`OmniFieldTerrainCfg` at the same RNG state, plus the site layers (kind ``FIELD_BLOCK``
    / ``FIELD_PIT``, level 0, no crossing radius). A SUBCLASS on purpose: the parkour term's
    ``field_mode`` (exact occupancy, seam-aware lookups, field keys) is chosen by ``isinstance`` and
    switches on for every site generator. Installed only under ``KRABBY_FLAT_TERRAIN_GEOM=site``.
    """

    function = site_terrains.omni_site_field_terrain


@configclass
class OmniSiteTerrainCfg(ExtremeParkourRoughTerrainCfg):
    """Site teacher (DESIGN §3.1-§3.6): one construction-site hazard kind per column, laid out around
    the spawn pad in the tile-centre frame by ``site_terrains.site_layout`` on the kind's ladder
    ``lo + (hi - lo) * d_site``. ``kind`` is a ``SITE_KINDS`` hazard name (``pits`` ... ``ramps``).

    The two radius fields keep the ring / field NAMES (the sector anchors and the parkour-term radius
    sync read them by name): ``clear_radius`` is the spawn pad, ``ring_outer_radius`` the half-width of
    the square every feature lies in. ``site_d_range`` / ``gen_d_range`` are written by the site
    applier (``KRABBY_SITE_DIFF`` and the generator's ``difficulty_range``); ``ramp_max_deg`` is the
    friction-capped ramp slope. Installed only under ``KRABBY_FLAT_TERRAIN_GEOM=site``.
    """

    function = site_terrains.omni_site_terrain

    kind: str = MISSING
    clear_radius: float = 1.7
    ring_outer_radius: float = 4.80
    field_margin: float = 0.16
    noise_range: tuple[float, float] = (0.02, 0.02)
    vertical_edges: bool = True
    site_d_range: tuple[float, float] = (0.0, 1.0)
    gen_d_range: tuple[float, float] = (0.0, 1.0)
    ramp_max_deg: float = 20.0
    max_tries: int = 64


@configclass
class OmniSiteRtTerrainCfg(OmniSiteTerrainCfg):
    """Rough-terrain teacher (``crab_hex_omni_task/experiments/2026-11-10_rough_terrain/PLAN.md`` step 2): one rt
    hazard column of ``OMNI_SITE_RT_TERRAINS_CFG`` (``kind`` one of ``site_terrains.RT_KINDS``), laid out on the
    rough-terrain ladder by ``site_terrains.omni_site_terrain_rt``. A SUBCLASS of :class:`OmniSiteTerrainCfg`, so every
    site check (class, radius fields, d mapping, caps) applies unchanged. Installed only under
    ``KRABBY_FLAT_TERRAIN_GEOM=site`` with ``KRABBY_SITE_LADDER=rt``.
    """

    function = site_terrains.omni_site_terrain_rt

    tile_col: int | None = None
    """The tile's column in the generator grid, set per tile by ``ParkourTerrainGenerator._get_terrain_mesh``: the
    slopes layout alternates its up / down variant by column parity (even column up), so every level holds both.
    None (a direct call of the body) falls back to a per-tile 0.5 / 0.5 draw."""

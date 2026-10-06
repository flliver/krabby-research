from isaaclab.utils import configclass

from ..extreme_parkour.extreme_parkour_terrains_cfg import ExtremeParkourRoughTerrainCfg
from . import omni_terrains


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

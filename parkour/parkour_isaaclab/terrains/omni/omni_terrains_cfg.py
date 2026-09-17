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

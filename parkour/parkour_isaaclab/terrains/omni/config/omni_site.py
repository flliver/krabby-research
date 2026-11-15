"""Pre-defined terrain configuration for the omni construction-SITE generator (site teacher,
``crab_hex_omni_task/experiments/2026-10-28_site_teacher/DESIGN.md`` §3.1).

Installed by ``crab_hex_omni_task/mdp/omni_terrain_knobs.install_omni_terrain_generator`` (a deep
copy) when ``KRABBY_FLAT_TERRAIN_GEOM=site``; the ring generator (``omni_ring.py``) stays the scene
default. Same tile size, scales, border, rows and slope threshold as the field. The sub-terrains are
in ``site_raster.SITE_SUBTERRAIN_KEYS`` order, one column kind each; the proportions here are the
default mix (``DEFAULT_SITE_MIX``, 20 columns: flat 0.2, legacy field 0.2, hazards 0.6), and the site
applier rewrites them and ``num_cols`` from ``KRABBY_SITE_MIX`` (zero-count keys stay declared with
proportion 0). ``curriculum=True`` is only the class value: the effective mode follows
``KRABBY_FLAT_TERRAIN_CURRICULUM``, as on the field.
"""
from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.omni import *

OMNI_SITE_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(10, 10),
    border_width=20,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0, 1),
    curriculum=True,
    sub_terrains={
        # "parkour_flat" is load-bearing (shared masks / knob loops key off it) -- as in the ring / field.
        "parkour_flat": OmniFlatTerrainCfg(proportion=0.20, clear_radius=1.7, ring_outer_radius=4.80),
        "site_field_sparse": OmniSiteFieldTerrainCfg(proportion=0.10, n_blocks=14),
        "site_field_dense": OmniSiteFieldTerrainCfg(proportion=0.10, n_blocks=28),
        "site_pits": OmniSiteTerrainCfg(proportion=0.10, kind="pits"),
        "site_trench": OmniSiteTerrainCfg(proportion=0.05, kind="trench"),
        "site_ledge_down": OmniSiteTerrainCfg(proportion=0.05, kind="ledge_down"),
        "site_ledge_up": OmniSiteTerrainCfg(proportion=0.05, kind="ledge_up"),
        "site_terraces": OmniSiteTerrainCfg(proportion=0.05, kind="terraces"),
        "site_snags": OmniSiteTerrainCfg(proportion=0.05, kind="snags"),
        "site_straddle": OmniSiteTerrainCfg(proportion=0.05, kind="straddle"),
        "site_chests": OmniSiteTerrainCfg(proportion=0.05, kind="chests"),
        "site_walls": OmniSiteTerrainCfg(proportion=0.05, kind="walls"),
        "site_rubble": OmniSiteTerrainCfg(proportion=0.05, kind="rubble"),
        "site_ramps": OmniSiteTerrainCfg(proportion=0.05, kind="ramps"),
    },
)

#: The rough-terrain generator (``crab_hex_omni_task/experiments/2026-11-10_rough_terrain/PLAN.md`` step 2), installed
#: instead of :data:`OMNI_SITE_TERRAINS_CFG` when ``KRABBY_SITE_LADDER=rt``. Same tile size, scales, border, rows and
#: slope threshold; the sub-terrains are ``omni_terrain_knobs.RT_SUBTERRAIN_KEYS`` in order, which is the column order
#: (gravel next to the flat / field columns, so a low-mu env never walks onto a slope or hill neighbour). The
#: proportions are the training mix (``RT_TRAIN_MIX``); the site applier rewrites them and ``num_cols`` from the
#: REQUIRED ``KRABBY_SITE_MIX``. The gravel column builds its own roughness (``apply_roughness=False`` is the dump's
#: record of that; the body ignores the cfg roughness on gravel either way).
OMNI_SITE_RT_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(10, 10),
    border_width=20,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.08,
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0, 1),
    curriculum=True,
    sub_terrains={
        "parkour_flat": OmniFlatTerrainCfg(proportion=0.15, clear_radius=1.7, ring_outer_radius=4.80),
        "rt_gravel": OmniSiteRtTerrainCfg(proportion=0.10, kind="gravel", apply_roughness=False),
        "site_field_sparse": OmniSiteFieldTerrainCfg(proportion=0.10, n_blocks=14),
        "site_field_dense": OmniSiteFieldTerrainCfg(proportion=0.05, n_blocks=28),
        "rt_slopes": OmniSiteRtTerrainCfg(proportion=0.10, kind="slopes"),
        "rt_hills": OmniSiteRtTerrainCfg(proportion=0.10, kind="hills"),
        "site_ledge_up": OmniSiteRtTerrainCfg(proportion=0.10, kind="ledge_up"),
        "site_ledge_down": OmniSiteRtTerrainCfg(proportion=0.10, kind="ledge_down"),
        "site_pits": OmniSiteRtTerrainCfg(proportion=0.10, kind="pits"),
        "site_rubble": OmniSiteRtTerrainCfg(proportion=0.10, kind="rubble"),
    },
)

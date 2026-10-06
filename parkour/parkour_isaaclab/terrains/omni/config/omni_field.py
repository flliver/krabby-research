"""Pre-defined terrain configuration for the omni FIELD generator (omni phase-2 plan, step 2).

Installed by ``crab_hex_omni_task/mdp/omni_terrain_knobs.install_omni_terrain_generator`` (a deep
copy) when ``KRABBY_FLAT_TERRAIN_GEOM=field``; the ring generator (``omni_ring.py``) stays the
scene default. Same tile size, scales, border, rows/cols and slope threshold as the ring, so the
two geometries differ ONLY in the sub-terrains -- and the flat tiles share the field's pad /
half-width so their sector anchors sit where the field tiles' do.
"""
from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.omni import *

OMNI_FIELD_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
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
        # "parkour_flat" is load-bearing (shared masks / knob loops key off it) -- as in the ring.
        "parkour_flat": OmniFlatTerrainCfg(proportion=0.5, clear_radius=1.7, ring_outer_radius=4.80),
        "omni_field_sparse": OmniFieldTerrainCfg(proportion=0.25, n_blocks=14),
        "omni_field_dense": OmniFieldTerrainCfg(proportion=0.25, n_blocks=28),
    },
)

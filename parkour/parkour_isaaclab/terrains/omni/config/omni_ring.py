from parkour_isaaclab.terrains.parkour_terrain_generator_cfg import ParkourTerrainGeneratorCfg
from parkour_isaaclab.terrains.omni import *

OMNI_RING_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
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
        # PLAN (composed-moth, D8): "parkour_flat" is load-bearing beyond this module --
        # shared masks/obs flags/knob loops elsewhere key off this exact string -- keep it.
        "parkour_flat": OmniFlatTerrainCfg(proportion=0.5),
        "omni_ring_sparse": OmniRingTerrainCfg(proportion=0.25, n_blocks=8),
        "omni_ring_dense": OmniRingTerrainCfg(proportion=0.25, n_blocks=16),
    },
)
